"""Explicit, offline recovery of decoration fragments. Never infer pixel ownership."""
from copy import deepcopy
from pathlib import Path
from collage.frame_links import photo_ids
import math
from PIL import Image, ImageChops, ImageDraw
from common import read, save, sha, verify_source


def check_plan(plan, scene):
    if not isinstance(plan, dict) or set(plan)-{'schema_version','reference_sha256','groups','windows','omit','edits','decisions'}:
        raise ValueError('Unknown recovery plan fields')
    if plan.get('schema_version')!='collage-recovery-v1':raise ValueError('Invalid recovery schema')
    if plan.get('reference_sha256')!=scene['reference']['sha256']:
        raise ValueError('Recovery plan reference does not match')
    objects={o['id']:o for o in scene['objects']};w,h=scene['reference_size']
    groups=plan.get('groups',[]);windows=plan.get('windows',[]);omitted=plan.get('omit',[])
    if not all(isinstance(x,list) for x in [groups,windows,omitted]):raise ValueError('Recovery entries must be lists')
    def reason(item):
        if not isinstance(item.get('reason'),str) or not item['reason'].strip():
            raise ValueError('Recovery requires a visual reason')
    def number(v):return type(v) in (int,float) and math.isfinite(v)
    members_seen=set()
    for group in groups:
        if not isinstance(group,dict) or set(group)-{'primary','member_ids','parts','reason'}:
            raise ValueError('Unknown recovery group fields')
        reason(group);members=group.get('member_ids',[]);parts=group.get('parts',[])
        if not isinstance(members,list) or not members or any(not isinstance(i,str) for i in members):
            raise ValueError('Recovery group needs member_ids')
        if len(set(members))!=len(members) or members_seen.intersection(members):
            raise ValueError('Recovery members cannot appear twice')
        if group.get('primary') not in members:raise ValueError('Recovery primary must be a member')
        for oid in members:
            o=objects.get(oid)
            if not o or o['kind']!='overlay':raise ValueError('Only known overlay layers may be grouped')
            if o.get('photo_id') or any(p['kind']=='photo' and p.get('parent_id')==oid for p in objects.values()):
                raise ValueError('Photo-bearing frames use window repair, not decoration fusion')
        if not isinstance(parts,list) or not parts:raise ValueError('Recovery group needs ordered parts')
        for part in parts:
            if not isinstance(part,dict) or set(part)-{'source_id','bbox','polygon'}:
                raise ValueError('Unknown recovery part fields')
            if part.get('source_id') not in members:raise ValueError('Part source must belong to the group')
            if 'bbox' in part and 'polygon' in part:raise ValueError('Use bbox or polygon, not both')
            if 'bbox' in part:
                b=part['bbox']
                if not isinstance(b,list) or len(b)!=4 or not all(number(v) for v in b) or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):
                    raise ValueError('Part bbox must be inside the reference canvas')
            if 'polygon' in part:
                points=part['polygon']
                if not isinstance(points,list) or len(points)<3 or any(not isinstance(p,list) or len(p)!=2 or not all(number(v) for v in p) or not (0<=p[0]<=w and 0<=p[1]<=h) for p in points):
                    raise ValueError('Part polygon must contain reference coordinates')
                area=sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1]))
                if abs(area)<1:raise ValueError('Part polygon has no area')
        members_seen.update(members)
    windows_seen=set()
    for item in windows:
        if not isinstance(item,dict) or set(item)-{'overlay_id','photo_id','padding','reason'}:
            raise ValueError('Unknown window repair fields')
        reason(item);oid=item.get('overlay_id');pid=item.get('photo_id')
        if not isinstance(oid,str) or not isinstance(pid,str):raise ValueError('Window IDs must be strings')
        o=objects.get(oid);p=objects.get(pid)
        if not o or o['kind'] not in ['overlay','frame'] or not p or p['kind']!='photo' or p.get('mode','cover')!='cover':
            raise ValueError('Window repair requires an overlay/frame and a cover photo')
        if pid not in photo_ids(o) and p.get('parent_id')!=oid:
            raise ValueError('Window repair requires an explicit photo relationship')
        if (oid,pid) in windows_seen or oid in members_seen:raise ValueError('Conflicting window repairs')
        pad=item.get('padding',0)
        if not number(pad) or not 0<=pad<=16:raise ValueError('Window padding must be 0..16 reference pixels')
        windows_seen.add((oid,pid))
    omitted_seen=set()
    for item in omitted:
        if not isinstance(item,dict) or set(item)-{'id','reason'} or not isinstance(item.get('id'),str):
            raise ValueError('Invalid omitted layer')
        reason(item);oid=item['id']
        if oid not in objects or objects[oid]['kind'] not in ['overlay','frame']:raise ValueError('Only a known overlay/frame can be omitted')
        if oid in omitted_seen or oid in members_seen or any(w[0]==oid for w in windows_seen):
            raise ValueError('Conflicting omitted layer')
        omitted_seen.add(oid)
    from asset_gate import check_extensions
    check_extensions(plan,scene)
    return plan


def load_plan(run, scene):
    path=Path(run)/'recovery.json'
    if not path.exists():return {'groups':[],'windows':[]}
    plan=check_plan(read(path),scene)
    scene['recovery_plan']={'file':str(path.resolve()),'sha256':sha(path)}
    return plan


def selected_part(image, part):
    """Restrict alpha only when explicitly requested; never crop, move or stretch RGB."""
    if 'bbox' not in part and 'polygon' not in part:return image.copy()
    mask=Image.new('L',image.size);draw=ImageDraw.Draw(mask)
    if 'bbox' in part:
        l,t,r,b=map(round,part['bbox']);draw.rectangle((l,t,r-1,b-1),fill=255)
    else:draw.polygon([tuple(map(round,p)) for p in part['polygon']],fill=255)
    result=image.copy();result.putalpha(ImageChops.multiply(image.getchannel('A'),mask))
    return result


def compose_groups(run, scene, assets, plan):
    """Use only acquired design layers. Background 00 and photo auxiliaries are absent."""
    assets=dict(assets);original=dict(assets);owners={};records=[]
    objects={o['id']:o for o in scene['objects']};size=tuple(scene['reference_size'])
    folder=Path(run)/'assets/reveal/recovered-groups';folder.mkdir(parents=True,exist_ok=True)
    for group in plan.get('groups',[]):
        sources={part['source_id'] for part in group['parts']}
        missing=sources-original.keys()
        if missing:raise ValueError('Missing recovery source layers: '+', '.join(sorted(missing)))
        canvas=Image.new('RGBA',size);dependencies=[]
        for part in group['parts']:
            source=original[part['source_id']]
            with Image.open(verify_source(source)) as raw:image=raw.convert('RGBA').resize(size,Image.Resampling.LANCZOS)
            tile=selected_part(image,part)
            if not tile.getchannel('A').getbbox():raise ValueError('Selected recovery part is empty: '+part['source_id'])
            canvas=Image.alpha_composite(canvas,tile)
            dependencies.append({'source_id':part['source_id'],'source':deepcopy(source),'selection':deepcopy(part)})
        primary=group['primary'];file=folder/(primary+'.png');canvas.save(file)
        manifest=folder/(primary+'.json')
        save(manifest,{'group':group,'parts':dependencies,'reference':scene['reference']})
        assets[primary]={'file':str(file.resolve()),'sha256':sha(file),'original_bbox':objects[primary]['bbox'],
                         'bbox':list(canvas.getchannel('A').getbbox()),'dependencies':dependencies,
                         'manifest':{'file':str(manifest.resolve()),'sha256':sha(manifest)},'recovery_group':group}
        for member in group['member_ids']:
            if member!=primary:owners[member]=primary
        records.append({'primary':primary,'member_ids':group['member_ids'],'reason':group['reason'],
                        'source':assets[primary]})
    scene['recovered_groups']=records
    return assets,owners


def verify_asset(source):
    verify_source(source)
    if source.get('raw_file'):verify_source({'file':source['raw_file'],'sha256':source['raw_sha256']})
    if source.get('manifest'):verify_source(source['manifest'])
    for part in source.get('dependencies',[]):verify_asset(part['source'])


def recover(run, path):
    """Recompile from downloaded layers with network disabled, retaining the first render."""
    from scene import load_scene, compile_scene
    from render import render
    run=Path(run);scene=load_scene(run);plan=check_plan(read(path),scene)
    if any(o.get('generated') for o in scene['objects']) or scene.get('generated_groups') or scene.get('revision',0):
        raise ValueError('Recover before apply/generate revisions; rebuilding would discard those edits')
    config=read(run/'reveal-config.json')
    if not config.get('enabled'):raise ValueError('Recovery requires downloaded Reveal layers')
    config['remote']=False
    # Detect missing donors before replacing the active recovery plan.
    index=read(verify_source(scene['reveal_index']))
    available={r['id'] for r in index['records'] if r.get('source')}
    needed={p['source_id'] for g in plan.get('groups',[]) for p in g['parts']}
    needed.update(w['overlay_id'] for w in plan.get('windows',[]))
    needed.update(o['id'] for o in plan.get('omit',[]))
    needed.update(o['id'] for o in plan.get('edits',[]))
    if needed-available:raise ValueError('Missing downloaded recovery sources: '+', '.join(sorted(needed-available)))
    save(run/'recovery.json',plan)
    compile_scene(run,config)
    return render(run)
