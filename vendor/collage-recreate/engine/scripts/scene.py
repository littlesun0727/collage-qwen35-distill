from collage.object_roles import is_customer_photo, is_photo_background
from copy import deepcopy
from collage.frame_links import photo_ids
import math
import os
from pathlib import Path
from common import read, save, sha, fingerprint, verify_source
from validate import analysis_check, bindings_check, review_check
from prepare import boxes
from effects import resolve_style
from collage.workflow.options import load_options


def compile_scene(run, reveal_config=None):
    run=Path(run);inputs=read(run/'input.json');verify_source(inputs['reference'])
    analysis=analysis_check(read(run/'analysis.json'),inputs)
    catalog=read(run/'prepared/catalog.json');bindings=bindings_check(read(run/'bindings.json'),analysis,catalog)
    assets={a['id']:a for a in catalog['assets']};bound={b['slot_id']:b for b in bindings['photos']};texts={t['object_id']:t for t in bindings['texts']}
    scale=inputs['output_width']/analysis['reference_size'][0]
    window_plan = None; windows = {}
    if (run/'photo-windows.json').exists():
        from photo_windows import check, configure, enforce_canvas_base
        window_plan = read(run/'photo-windows.json')
        enforce_canvas_base(analysis, window_plan)
        window_plan = check(window_plan, analysis)
        windows = {p['photo_id']:p for p in window_plan['photos']}
    objects=[]
    for original in analysis['objects']:
        o=deepcopy(original);o['style']=resolve_style(o)
        if o['id'] in texts:o['text']=texts[o['id']]['text'];o['text_origin']=texts[o['id']]['origin']
        o['text_unresolved']=o.get('text_status')=='unreadable' and o['id'] not in texts
        if is_customer_photo(o):
            o.setdefault('mode','cover')
            if o['mode']=='stroke' and window_plan is None:
                raise ValueError(o['id'] + ': stroke mode requires photo-windows.json parameters')
            if window_plan is not None:configure(o, windows[o['id']])
            o['binding']=bound[o['id']];o['source']=assets[o['binding']['asset_id']]
        objects.append(o)
    from photo_links import attach as attach_photo_links
    attach_photo_links(objects)
    scene={'schema_version':'collage-scene-v1','reference':inputs['reference'],'reference_size':analysis['reference_size'],'canvas_size':[inputs['output_width'],round(analysis['reference_size'][1]*scale)],'objects':objects,'layer_order':analysis['layer_order'],'cutout_model':inputs.get('cutout_model'),'sources':{'analysis':sha(run/'analysis.json'),'bindings':sha(run/'bindings.json'),'input':sha(run/'input.json'),'catalog':sha(run/'prepared/catalog.json')},'revision':0}
    scene['workflow_options']=load_options(run)
    if window_plan is not None:scene['sources']['photo_windows']=sha(run/'photo-windows.json')
    config_file=run/'reveal-config.json'
    if os.getenv('COLLAGE_BEST_EFFORT') == '1':
        scene['best_effort'] = True
    if reveal_config is not None:save(config_file,reveal_config)
    config=read(config_file) if config_file.exists() else {}
    if config.get('enabled'):
        from reveal_assets import attach
        attach(run,scene,config)
    from frame_fallback import align_frames
    align_frames(scene)
    if config_file.exists():scene['reveal_config_sha256']=sha(config_file)
    save(run/'scene.json',scene);boxes(run,analysis)
    return scene


def load_scene(run):
    run=Path(run);s=read(run/'scene.json')
    plan_path=run/'photo-windows.json';expected=s['sources'].get('photo_windows')
    if (expected is not None and (not plan_path.is_file() or sha(plan_path)!=expected)) or (expected is None and plan_path.exists()):
        raise ValueError('photo-windows.json changed: run build to compile new inputs')
    for name,p in [('analysis','analysis.json'),('bindings','bindings.json'),('input','input.json'),('catalog','prepared/catalog.json')]:
        if sha(run/p)!=s['sources'][name]:raise ValueError(f'{p} changed: run build to compile new inputs')
    verify_source(s['reference'])
    if s.get('reveal_config_sha256') and sha(run/'reveal-config.json')!=s['reveal_config_sha256']:
        raise ValueError('Reveal configuration changed; build again')
    from reveal_assets import verify
    verify(s)
    from photo_links import attach as attach_photo_links
    attach_photo_links(s['objects'])
    return s


def checked_review(run, path):
    run=Path(run);s=load_scene(run);r=review_check(read(path),s)
    result=read(run/'result.json')
    if r.get('render_id'):
        if r['render_id']!=result.get('render_id'):raise ValueError('Review is stale: render_id changed')
        r.setdefault('scene_sha256',result['scene_sha256']);r.setdefault('final_sha256',result['final_sha256'])
    if r['scene_sha256']!=sha(run/'scene.json') or r['final_sha256']!=sha(run/'final.png'):
        raise ValueError('Review is stale: use current scene_sha256 and final_sha256 from result.json')
    return s,r


def move_local_photos(scene, before, carrier, explicit_geometry):
    """Move a local carrier's photo windows with one similarity transform."""
    related=[p for p in scene['objects'] if is_customer_photo(p) and
             (p['id'] in photo_ids(carrier) or p.get('parent_id')==carrier['id'])]
    if not related:return
    if any(p['id'] in explicit_geometry for p in related):
        raise ValueError('Adjust carrier geometry only; linked photos follow automatically')
    if any(p.get('photo_window') or p.get('window_bbox') or p.get('generated') for p in related):
        raise ValueError('Fixed extracted/legacy windows cannot follow a local carrier')
    a,b=before['bbox'],carrier['bbox']
    sx=(b[2]-b[0])/(a[2]-a[0]);sy=(b[3]-b[1])/(a[3]-a[1])
    # Allow integer-coordinate rounding, but not a distorted shared window.
    scale=(sx+sy)/2
    if max(abs((b[2]-b[0])-scale*(a[2]-a[0])),abs((b[3]-b[1])-scale*(a[3]-a[1])))>1:
        raise ValueError('Linked carrier resize must preserve aspect ratio (uniform scale)')
    angle=carrier.get('rotation',0)-before.get('rotation',0)
    c,s=math.cos(math.radians(angle)),math.sin(math.radians(angle))
    ax,ay=(a[0]+a[2])/2,(a[1]+a[3])/2;bx,by=(b[0]+b[2])/2,(b[1]+b[3])/2
    for photo in related:
        box=photo['bbox'];dx=((box[0]+box[2])/2-ax)*scale;dy=((box[1]+box[3])/2-ay)*scale
        cx,cy=bx+c*dx-s*dy,by+s*dx+c*dy
        w,h=(box[2]-box[0])*scale,(box[3]-box[1])*scale
        photo['bbox']=[round(cx-w/2),round(cy-h/2),round(cx+w/2),round(cy+h/2)]
        photo['rotation']=(photo.get('rotation',0)+angle+180)%360-180


def remove_objects(s, removed):
    removed=set(removed);by_id={o['id']:o for o in s['objects']}
    if not removed:return
    if not removed<=by_id.keys():raise ValueError('Unknown object to remove')
    if len(removed)==len(by_id):raise ValueError('Keep at least one visible scene object')
    for obj in s['objects']:
        if obj['kind']=='frame' and obj['id'] not in removed and photo_ids(obj) and set(photo_ids(obj)) <= removed:
            raise ValueError('Remove the linked frame with its photo: '+obj['id'])
        if obj['id'] not in removed and obj.get('photo_window_spec',{}).get('linked_to') in removed:
            raise ValueError('Remove the pop-out foreground with its base photo: '+obj['id'])
    for obj in s['objects']:
        for key in ['embedded_owner','recovery_owner']:
            owner=obj.get(key)
            if owner and ((obj['id'] in removed)!=(owner in removed)):
                raise ValueError('Remove the complete merged layer: '+obj['id']+' and '+owner)
    for group in s.get('generated_groups',[]):
        members=set(group['member_ids'])
        if removed & members and not members<=removed:raise ValueError('Remove the complete generated group')
    s['objects']=[o for o in s['objects'] if o['id'] not in removed]
    s['layer_order']=[oid for oid in s['layer_order'] if oid not in removed]
    s['generated_groups']=[g for g in s.get('generated_groups',[]) if not set(g['member_ids'])<=removed]
    for obj in s['objects']:
        for key in ['photo_id','parent_id','embedded_in','reveal_frame_id']:
            value = obj.get(key)
            if isinstance(value, list):
                obj[key] = [identifier for identifier in value if identifier not in removed]
            elif value in removed:
                obj.pop(key)
    s['reveal_unconfirmed_text']={k:[v for v in values if v not in removed]
        for k,values in s.get('reveal_unconfirmed_text',{}).items() if k not in removed}


def apply_review(run, path, dry_run=False):
    run=Path(run);s,r=checked_review(run,path);by_id={o['id']:o for o in s['objects']};pending=[]
    # Validate all changes in memory before writing a revision or altering assets.
    explicit_geometry={i['id'] for i in r['items'] if i['action']=='adjust' and
                       {'bbox','rotation'} & set(i['changes'])}
    moving_carriers=[by_id[i] for i in explicit_geometry if by_id[i]['kind']=='overlay' and
                     by_id[i].get('method','local')=='local' and not by_id[i].get('recovered') and
                     not by_id[i].get('generated')]
    claimed=[]
    for carrier in moving_carriers:
        claimed.extend(p['id'] for p in s['objects'] if is_customer_photo(p) and
                       (p['id'] in photo_ids(carrier) or p.get('parent_id')==carrier['id']))
    if len(claimed)!=len(set(claimed)):
        raise ValueError('A photo cannot follow multiple adjusted carriers')
    removed={i['id'] for i in r['items'] if i['action']=='remove'}
    remove_objects(s,removed)
    if 'layer_order' in r:s['layer_order']=list(r['layer_order'])
    for item in r['items']:
        o=by_id[item['id']]
        if item['action']=='adjust':
            before=deepcopy(o)
            protected=o.get('recovered') or o.get('embedded_owner') or o.get('recovery_owner') or o.get('photo_window')
            changes=item['changes']
            translates_extraction=False
            if o['kind'] in ['overlay','frame'] and o.get('recovered') and set(changes)=={'bbox'}:
                linked=bool(o.get('photo_id') or o.get('photo_window') or
                    any(is_customer_photo(p) and p.get('parent_id')==o['id'] for p in s['objects']))
                grouped=bool(o.get('embedded_owner') or o.get('recovery_owner') or
                    o['recovered'].get('source',{}).get('recovery_group') or
                    any(p.get('embedded_owner')==o['id'] or p.get('recovery_owner')==o['id'] for p in s['objects']))
                old=o['bbox'];new=changes['bbox'];dx=new[0]-old[0];dy=new[1]-old[1]
                translates_extraction=not linked and not grouped and new[2]-old[2]==dx and new[3]-old[3]==dy
                if translates_extraction:
                    offset=o.get('extracted_offset',[0,0]);o['extracted_offset']=[offset[0]+dx,offset[1]+dy]
            allowed_photo={'crop_center','source_crop','mirror_x','asset_id'} if o['kind'] in {'photo','background'} else set()
            if protected and not translates_extraction and set(changes)-allowed_photo:
                raise ValueError('Recovered geometry/text belongs to the extraction: update analysis and build again')
            if any(o['id'] in g['member_ids'] for g in s.get('generated_groups',[])):
                raise ValueError('Fused generated member cannot be adjusted separately; recompile analysis to split or regenerate the whole group')
            # Bind a replacement first so placement does not depend on JSON key order.
            ordered=sorted(changes,key=lambda key:key!='asset_id')
            for key in ordered:
                value=changes[key]
                if key=='asset_id':
                    if o['kind'] not in {'photo','background'}:raise ValueError('asset_id only applies to customer photos or backgrounds')
                    catalog={a['id']:a for a in read(run/'prepared/catalog.json')['assets']}
                    if value not in catalog:raise ValueError('Unknown customer asset: '+value)
                    verify_source(catalog[value])
                    if o['kind']=='background' and not is_photo_background(o):
                        if o.get('embedded_owner') or o.get('recovery_owner') or any(
                            p.get('embedded_owner')==o['id'] or p.get('recovery_owner')==o['id'] for p in s['objects']):
                            raise ValueError('Cannot replace a background merged with other objects')
                        # Keep the layer ID/position/order; replace its pixels with a verified customer image.
                        for old_key in ['recovered','generated','gate','gate_local','photo_window','window_bbox','text','text_status','extracted_offset']:
                            o.pop(old_key,None)
                        o.update(kind='background',mode='cover',method='local',style={},binding={'slot_id':o['id']},
                                 bbox=[0,0,*s['reference_size']],rotation=0,
                                 text_unresolved=False,customer_background=True,background_type='photo')
                    o['source']=deepcopy(catalog[value]);o['binding']['asset_id']=value
                    for placement in ['source_crop','crop_center','mirror_x']:
                        if placement not in changes:o['binding'].pop(placement,None)
                elif key=='text':
                    if o['kind']!='text' or o.get('method')=='extract' or o.get('generated'):
                        raise ValueError('Only independent local text can be edited')
                    if not value.strip():raise ValueError('Text must not be blank')
                    o.update(text=value,text_status='known',text_unresolved=False,text_origin='customer_revision')
                elif key=='style':
                    if o.get('photo_window_spec') and {'corner_radius','feather','outline_width','outline_color','card','stroke','stroke_width'} & set(value):
                        raise ValueError('Photo window effects belong to photo-windows.json; update the plan and build again')
                    for name,setting in value.items():
                        if isinstance(setting,dict):o['style'][name]={**o['style'].get(name,{}),**setting}
                        else:o['style'][name]=setting
                elif key in ['crop_center','source_crop','mirror_x']:
                    if not is_customer_photo(o):raise ValueError('Photo placement parameters only apply to photos')
                    o['binding'][key]=value
                else:o[key]=value
            if o in moving_carriers:
                move_local_photos(s,before,o,explicit_geometry)
        elif item['action']=='generate':pending.append(o['id'])
    w,h=s['reference_size']
    for o in s['objects']:
        l,t,rr,b=o['bbox']
        if not (0<=l<rr<=w and 0<=t<b<=h):raise ValueError('Adjusted bbox is out of bounds: '+o['id'])
    s['pending_generation']=pending;s['revision']+=1
    if not dry_run:
        save(run/'reviews'/f'revision-{s["revision"]}.json',r);save(run/'scene.json',s)
    return s
