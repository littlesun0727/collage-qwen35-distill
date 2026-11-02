"""Deterministic admission of extracted layers; semantic uncertainty is explicit."""
from collage.object_roles import is_customer_photo
from pathlib import Path
import math
import hashlib
import numpy as np
from PIL import Image, ImageDraw
from common import read, save, sha, fingerprint, verify_source

VERSION = 'asset-gate-v4-screen-switch'
HARD_ISSUES = {'empty_layer','repair_failed','explicitly_omitted',
               }


def check_extensions(plan, scene):
    objects={o['id']:o for o in scene['objects']};w,h=scene['reference_size']
    for field in ['edits','decisions']:
        if not isinstance(plan.get(field,[]),list):raise ValueError(field+' must be a list')
    seen=set()
    for d in plan.get('decisions',[]):
        if set(d)-{'id','input_key','verdict','reason','embedded_ids'}:raise ValueError('Unknown gate decision field')
        if d.get('id') not in objects or objects[d['id']]['kind'] not in ['overlay','frame']:raise ValueError('Decision needs a known overlay/frame')
        if d['id'] in seen:raise ValueError('Duplicate gate decision')
        seen.add(d['id'])
        if d.get('verdict') not in ['accepted','rejected','local']:raise ValueError('Invalid gate verdict')
        if objects[d['id']]['kind']=='frame' and d['verdict']=='local':raise ValueError('Extracted frames cannot use local fallback')
        if not isinstance(d.get('reason'),str) or not d['reason'].strip():raise ValueError('Decision needs a reason')
        key=d.get('input_key','')
        if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Decision needs current gate input_key')
        ids=d.get('embedded_ids',[])
        if not isinstance(ids,list) or any(i not in objects or (is_customer_photo(objects[i]) or objects[i]['kind']=='frame') for i in ids):raise ValueError('Invalid embedded_ids')
        if ids and d['verdict']!='accepted':raise ValueError('Only accepted layers can own embedded content')
        for child in ids:
            owner=objects[child].get('embedded_in') or objects[child].get('parent_id')
            if owner!=d['id']:raise ValueError('Embedded acceptance requires the existing explicit owner')
    for e in plan.get('edits',[]):
        if set(e)-{'id','operation','bbox','polygon','reason','fill','backend','texts','all_text','padding'}:raise ValueError('Unknown repair edit field')
        if e.get('id') not in objects or objects[e['id']]['kind'] not in ['overlay','frame']:raise ValueError('Edit needs a known overlay/frame')
        if e.get('operation') not in ['erase','fill','inpaint','remove_text']:raise ValueError('Invalid edit operation')
        if not isinstance(e.get('reason'),str) or not e['reason'].strip():raise ValueError('Edit needs a visual reason')
        if ('bbox' in e)==('polygon' in e):raise ValueError('Edit needs exactly one bbox or polygon')
        def num(x):return type(x) in (int,float) and math.isfinite(x)
        if 'bbox' in e:
            b=e['bbox']
            if not isinstance(b,list) or len(b)!=4 or not all(num(x) for x in b) or not 0<=b[0]<b[2]<=w or not 0<=b[1]<b[3]<=h:raise ValueError('Edit bbox outside reference')
        else:
            p=e['polygon']
            if not isinstance(p,list) or len(p)<3 or any(not isinstance(q,list) or len(q)!=2 or not all(num(x) for x in q) or not 0<=q[0]<=w or not 0<=q[1]<=h for q in p):raise ValueError('Invalid edit polygon')
            if abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(p,p[1:]+p[:1])))<1:raise ValueError('Empty edit polygon')
        if e['operation']=='fill' or (e['operation']=='remove_text' and e.get('backend')=='fill'):
            from PIL import ImageColor
            try:ImageColor.getrgb(e['fill'])
            except (ValueError,KeyError,TypeError):raise ValueError('Fill repair needs an explicit color')
        if e['operation']=='remove_text':
            if e.get('backend','lama') not in ['lama','fill']:raise ValueError('Invalid text repair backend')
            if type(e.get('all_text',False)) is not bool:raise ValueError('all_text must be boolean')
            texts=e.get('texts',[])
            if not isinstance(texts,list) or any(not isinstance(t,str) or not t.strip() for t in texts):raise ValueError('Invalid expected text')
            if not texts and not e.get('all_text'):raise ValueError('Specify texts or explicitly authorize all_text inside selection')
            if not num(e.get('padding',2)) or not 0<=e.get('padding',2)<=8:raise ValueError('OCR padding must be 0..8')
        if any(x['id']==e['id'] for x in plan.get('omit',[])):raise ValueError('Cannot edit an omitted layer')


def inspect(image, target, scene, source, repair, checks, text_candidates, repair_errors):
    """No pixel changes here. Decisions bind to the current candidate and context."""
    screen_enabled=scene.get('workflow_options',{}).get('screen',False)
    a=np.asarray(image.getchannel('A'));visible=a>16;count=int(visible.sum())
    key=fingerprint({'version':VERSION,'code':sha(__file__),'pixels':hashlib.sha256(image.tobytes()).hexdigest(),
        'source':source,'target':target,'sources':scene['sources'],'screen':screen_enabled,
        'edits':[e for e in repair.get('edits',[]) if e['id']==target['id']],
        'windows':[e for e in repair.get('windows',[]) if e['overlay_id']==target['id']],
        'checks':checks,'text_candidates':text_candidates,'repair_errors':repair_errors})
    issues=[];hard=[];warnings=[]
    if not count:hard.append('empty_layer')
    if repair_errors:hard.append('repair_failed')
    if any(x['id']==target['id'] for x in repair.get('omit',[])):hard.append('explicitly_omitted')
    if target.get('kind')=='frame':
        from collage.frame_links import photo_ids
        ids = photo_ids(target)
        linked=[c for c in checks if c.get('photo_id') in ids]
        if set(c.get('photo_id') for c in linked) != set(ids) or any(c.get('interior_visible_fraction') is None for c in linked):
            warnings.append('frame_window_unverified')
        elif any(c['interior_visible_fraction']>.5 for c in linked):
            issues.append('opaque_frame_window_needs_semantic_check')
        elif any(c['interior_visible_fraction']>.05 for c in linked):
            issues.append('frame_interior_needs_visual_check')
    if any(c.get('review_required') and c['visible_fraction']>.5 for c in checks):issues.append('occupied_foreground_window')
    elif any(c.get('review_required') for c in checks):warnings.append('foreground_window_edge_content')
    if text_candidates:warnings.append('unconfirmed_embedded_content')
    if not repair_errors and any(e['id']==target['id'] and e['operation'] in ['inpaint','remove_text'] for e in repair.get('edits',[])):
        issues.append('repaired_content_needs_visual_check')
    # Use a rotated envelope plus slack; never delete outliers or select a largest component.
    l,t,r,b=target['bbox'];cx=(l+r)/2;cy=(t+b)/2
    angle=math.radians(target.get('rotation',0));cw=abs(math.cos(angle));sw=abs(math.sin(angle))
    ew=(r-l)*cw+(b-t)*sw;eh=(r-l)*sw+(b-t)*cw
    slack=max(8,max(image.size)*.012)
    shadow=target.get('style',{}).get('shadow',{})
    slack+=3*shadow.get('blur',0)+max([abs(v) for v in shadow.get('offset',[0,0])])
    box=[max(0,int(cx-ew/2-slack)),max(0,int(cy-eh/2-slack)),min(image.width,math.ceil(cx+ew/2+slack)),min(image.height,math.ceil(cy+eh/2+slack))]
    inside=int(visible[box[1]:box[3],box[0]:box[2]].sum());outside=(count-inside)/max(count,1)
    if outside>.02 and count-inside>32 and not source.get('recovery_group'):
        (issues if outside>.5 else warnings).append('pixels_outside_expected_extent')
    decision=next((d for d in repair.get('decisions',[]) if d['id']==target['id']),None)
    valid=decision and decision['input_key']==key
    if decision and not valid:issues.append('stale_visual_decision')
    # A disabled screen accepts recovered pixels at build time too. Heuristics
    # remain diagnostic warnings, so skipping the later VLM cannot strand frames
    # in pending status and replace them with local approximations.
    if target.get('kind') != 'frame' or not screen_enabled:
        warnings.extend(issues);issues=[]
    status='rejected' if hard else 'pending' if issues else 'accepted';basis='program';embedded=[];use_local=False
    if target.get('kind')!='frame' and status=='accepted':
        embedded=[o['id'] for o in scene.get('objects',[]) if o.get('embedded_in')==target['id']
                  and not is_customer_photo(o) and o['kind']!='frame']
    if valid:
        if decision['verdict']=='rejected' and target.get('kind')=='frame':status='rejected';basis='visual'
        elif decision['verdict']=='accepted' and not hard:
            status='accepted';basis='visual';embedded=decision.get('embedded_ids',[])
        elif decision['verdict']=='local' and target.get('kind')!='frame':
            use_local=True;status='rejected';basis='local_requested'
    return {'id':target['id'],'status':status,'basis':basis,'input_key':key,
        'issues':hard+issues,'warnings':warnings,'repair_errors':repair_errors,'window_checks':checks,
        'text_candidates':text_candidates,'outside_fraction':round(outside,6),
        'embedded_ids':embedded,'use_local':use_local,
        'decision':decision if valid else None,'semantic_verified':basis=='visual' and status=='accepted'}


def publish(run, scene, records):
    folder=Path(run)/'assets/gate';folder.mkdir(parents=True,exist_ok=True)
    counts={s:sum(r['status']==s for r in records) for s in ['accepted','pending','rejected']}
    report={'version':VERSION,'sources':scene['sources'],'counts':counts,'records':records,
            'note':'Program admission is not semantic or whole-composition acceptance'}
    file=folder/'report.json';save(file,report)
    scene['asset_gate']={'file':str(file.resolve()),'sha256':sha(file),'version':VERSION}
    sheets=[]
    selected=[r for r in records if (r['status']!='accepted' or r.get('warnings')) and r.get('candidate')]
    from PIL import ImageFont
    from text import font_file
    font=ImageFont.truetype(font_file({'font':'sans'},''),15)
    for start in range(0,len(selected),6):
        sheet=Image.new('RGB',(1080,960),'#dddddd');draw=ImageDraw.Draw(sheet)
        for i,rec in enumerate(selected[start:start+6]):
            x=(i%2)*540;y=(i//2)*320
            draw.text((x+6,y+4),rec['id']+' '+rec['status'],font=font,fill='black')
            draw.text((x+6,y+24),', '.join(rec['issues']+rec.get('warnings',[]))[:65],font=font,fill='#803000')
            for j,field in enumerate(['reference_crop','foreground']):
                path=rec.get(field)
                if not path:continue
                with Image.open(path) as im:tile=im.convert('RGBA');tile.thumbnail((258,260))
                px=x+6+270*j;py=y+52
                draw.rectangle((px,py,px+258,py+260),fill='#aaaaaa')
                sheet.paste(tile,(px+(258-tile.width)//2,py+(260-tile.height)//2),tile)
        path=Path(run)/'previews'/f'gate-{start//6+1}.jpg';path.parent.mkdir(exist_ok=True);sheet.save(path)
        sheets.append(str(path))
    scene['asset_gate_summary']={**counts,'warning_objects':sum(bool(r.get('warnings')) for r in records),'sheets':sheets}


def verify(scene):
    if scene.get('screening'):verify_source(scene['screening'])
    if not scene.get('asset_gate'):
        if any(o.get('gate') for o in scene['objects']):raise ValueError('Missing asset gate report')
        return
    report=read(verify_source(scene['asset_gate']))
    if report['sources']!=scene['sources']:raise ValueError('Gate inputs changed; rebuild')
    records={r['id']:r for r in report['records']}
    for r in records.values():
        if r.get('candidate'):verify_source(r['candidate'])
    for o in scene['objects']:
        if o.get('recovered'):
            r=records.get(o['id'])
            if not r or r['status']!='accepted' or r.get('candidate',{}).get('sha256')!=o['recovered']['sha256']:raise ValueError('Recovered layer has no valid gate admission: '+o['id'])
        if o.get('gate'):
            r=records.get(o['id'])
            if not r or o['gate']!={k:r[k] for k in ['status','basis','input_key']}:raise ValueError('Gate state changed: '+o['id'])
            if bool(o.get('gate_local'))!=bool(r.get('local_fallback')):raise ValueError('Unapproved local fallback: '+o['id'])
            if r.get('local_style') is not None and o.get('style')!=r['local_style']:raise ValueError('Local fallback style changed: '+o['id'])
