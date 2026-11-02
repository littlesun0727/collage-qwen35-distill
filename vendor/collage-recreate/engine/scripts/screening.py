"""One offline admission pass. Never edits extracted pixels or calls a model."""
from copy import deepcopy
from pathlib import Path
from common import read, save, sha, verify_source


def retain_uncertain_frames(answer, records):
    """Automatic rejection needs photographic evidence; dirt alone never removes a frame."""
    from asset_gate import HARD_ISSUES
    value=deepcopy(answer);by_id={r['id']:r for r in records};notes=[]
    for decision in value['decisions']:
        record=by_id.get(decision['id'],{})
        available=bool(record.get('candidate')) and not (set(record.get('issues',[])) & HARD_ISSUES)
        if decision['action']=='drop' and available and decision.get('photo_content')!='present':
            decision['action']='keep'
            decision['reason']='未确认原照片残留，保留可用候选；'+decision['reason']
            notes.append(decision['id']+'：没有明确原照片残留证据，保留候选。')
        elif decision['action'] in ['keep','backing'] and not available:
            decision['action']='drop'
            decision.pop('embedded_ids',None)
            decision['reason']='候选缺失或为空，使用程序相框；'+decision['reason']
            notes.append(decision['id']+'：无可用像素，程序相框补位。')
    return value,notes


def screen(run, path, dry_run=False, rescreen=False):
    from scene import load_scene
    from asset_gate import publish, HARD_ISSUES
    from validate import analysis_check
    from reveal_assets import contains
    from render import render
    run=Path(run);scene=load_scene(run);result=read(run/'result.json');plan=read(path)
    if set(plan)-{'render_id','decisions'} or plan.get('render_id')!=result.get('render_id'):
        raise ValueError('Screen needs the current result.render_id')
    if result.get('scene_sha256')!=sha(run/'scene.json') or result.get('final_sha256')!=sha(run/'final.png'):
        raise ValueError('Screen result is stale; render first')
    if scene.get('screening') and not rescreen:
        raise ValueError('Screen already applied; use explicit rescreen to reassess the current candidates')
    if scene.get('revision') or scene.get('generated_groups') or any(o.get('generated') for o in scene['objects']):
        raise ValueError('Screen before later manual revisions or generation')
    decisions=plan.get('decisions')
    if not isinstance(decisions,list):raise ValueError('Screen decisions must be a list')
    report=deepcopy(read(verify_source(scene['asset_gate'])))
    records={r['id']:r for r in report['records']};objects={o['id']:o for o in scene['objects']}
    analysis=deepcopy(read(run/'analysis.json'));analysis_objects={o['id']:o for o in analysis['objects']}
    seen=set();claims=set()
    for d in decisions:
        if not isinstance(d,dict) or set(d)-{'id','action','reason','style','embedded_ids','photo_content'}:
            raise ValueError('Unknown screen decision fields')
        if 'photo_content' in d and d['photo_content'] not in ['present','absent','uncertain']:
            raise ValueError('Unknown photographic evidence classification')
        oid=d.get('id');o=objects.get(oid)
        local_draw=o and o.get('method')=='local' and d.get('action')=='draw'
        if oid in seen or not o or o['kind'] not in ['overlay','frame'] or (oid not in records and not local_draw):
            raise ValueError('Screen needs a unique extracted overlay/frame, or an existing local overlay for draw: '+str(oid))
        seen.add(oid)
        if o['kind'] != 'frame' and d.get('action') != 'keep':
            raise ValueError('Automatic screening preserves non-frame decorations: '+oid)
        if d.get('action') not in ['keep','drop','draw','backing'] or not isinstance(d.get('reason'),str) or not d['reason'].strip():
            raise ValueError('Screen needs keep/drop/draw and a reason: '+oid)
        if o.get('recovery_owner') or any(oid in g['member_ids'] for g in scene.get('recovered_groups',[])):
            raise ValueError('Screen does not split recovered groups')
        if 'style' in d and d['action']!='draw':raise ValueError('style is only for draw')
        if d['action']=='draw':
            if o['kind']=='frame':raise ValueError('Extracted frame cannot be replaced by a local draw: '+oid)
            style=d.get('style',o.get('style',{}))
            if oid not in analysis_objects or not isinstance(style,dict) or not style.get('shape'):
                raise ValueError('draw needs an explicit supported primitive: '+oid)
            analysis_objects[oid]['style']=deepcopy(style);analysis_objects[oid]['method']='local'
        r=records.get(oid,{})
        if d['action'] in ['keep','backing'] and (not r.get('candidate') or set(r.get('issues',[])) & HARD_ISSUES):
            raise ValueError('Cannot keep unavailable or failed pixels: '+oid)
        ids=d.get('embedded_ids',[])
        if not isinstance(ids,list) or (ids and d['action'] not in ['keep','backing']):raise ValueError('Only kept pixels may own embedded content')
        for child in ids:
            c=objects.get(child,{})
            owner=c.get('embedded_in') or c.get('parent_id')
            candidate=any(x['id']==child for x in r.get('text_candidates',[]))
            if child in claims or c.get('kind') not in ['text','overlay'] or c.get('text_origin') or child==oid:
                raise ValueError('Invalid embedded member: '+str(child))
            if owner not in [None,oid] or not (owner==oid or candidate) or contains(o['bbox'],c['bbox'])<.9:
                raise ValueError('Embedded member lacks a compatible spatial/owner relation: '+child)
            claims.add(child)
    analysis_check(analysis)
    if dry_run:return {'valid':True}
    previous = scene.get('screening')
    if previous:
        previous_path = verify_source(previous)
        archive = run/'assets/screening-history'/(previous['sha256']+'.json')
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_bytes(Path(previous_path).read_bytes())
        scene.setdefault('screening_history', []).append(
            {'file':str(archive.resolve()),'sha256':sha(archive),'render_id':result['render_id']})
    for d in decisions:
        oid=d['id'];o=objects[oid];r=records.get(oid);action=d['action']
        if r is None:
            # An existing local primitive has no extracted candidate to reject.
            o.update(method='local',style=deepcopy(analysis_objects[oid]['style']))
            continue
        for child in objects.values():
            if child.get('embedded_owner')==oid:child.pop('embedded_owner')
        for field in ['recovered','gate_local','gate_embedded_ids','reveal_warning','frame_role']:
            o.pop(field,None)
        r.update(status='accepted' if action in ['keep','backing'] else 'rejected',basis='visual' if action!='draw' else 'local_requested',
                 semantic_verified=action in ['keep','backing'],decision=deepcopy(d),embedded_ids=d.get('embedded_ids',[]),
                 use_local=action=='draw',local_fallback=action=='draw')
        if action in ['keep','backing']:
            o['recovered']=deepcopy(r['candidate']);o['gate_embedded_ids']=r['embedded_ids']
            for child in r['embedded_ids']:objects[child]['embedded_owner']=oid
            # The decision describes the pixels; preserve the supplied layer order.
            if o['kind']=='frame':
                o['frame_role']='backing' if action=='backing' else 'border'
        elif action=='draw':
            o.update(method='local',style=deepcopy(analysis_objects[oid]['style']),gate_local=True)
            r['local_style']=deepcopy(o['style'])
        else:o['reveal_warning']='Screen rejected extracted layer: '+d['reason']
        o['gate']={k:r[k] for k in ['status','basis','input_key']}
    for child,owners in list(scene.get('reveal_unconfirmed_text',{}).items()):
        remaining=[oid for oid in owners if objects[oid].get('recovered') and objects[child].get('embedded_owner')!=oid]
        if remaining:scene['reveal_unconfirmed_text'][child]=remaining
        else:scene['reveal_unconfirmed_text'].pop(child)
    from frame_fallback import align_frames
    align_frames(scene)
    receipt=run/'screening.json';save(receipt,plan)
    scene['screening']={'file':str(receipt.resolve()),'sha256':sha(receipt)}
    publish(run,scene,list(records.values()));save(run/'scene.json',scene)
    return render(run)
