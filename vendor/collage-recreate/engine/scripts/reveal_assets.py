"""Compile recovered full-canvas layers and shared customer-photo windows."""
from collage.object_roles import is_customer_photo, is_photo_background
from copy import deepcopy
from collage.frame_links import photo_ids
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageFilter, ImageOps
from common import read, save, sha, fingerprint, verify_source
from effects import photo_layout, rounded_mask
from reveal import acquire, request_object
from recovery import load_plan, compose_groups, verify_asset
from asset_gate import inspect as inspect_asset, publish as publish_gate
from asset_repair import apply_edits


def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])


def contains(a,b):
    return area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])/max(1,area(b))


def frame_hint(o):
    return (o.get('kind')=='frame' or o.get('style',{}).get('shape')=='frame' or o.get('photo_id')
            or any(s in o.get('label','').lower() for s in ['相框','相纸','拍立得','polaroid','frame']))


def targets(scene):
    result=[]
    for o in scene['objects']:
        if o['kind'] in ['overlay','frame']:result.append({**o,'request':o.get('method')!='local'})
        elif is_customer_photo(o) and not o.get('photo_window_spec') and (o.get('appearance')=='polaroid' or 'card' in o['style']):
            result.append({'id':'frame_'+fingerprint(o['id'])[:16],'kind':'overlay','bbox':o['bbox'],
                           'photo_id':o['id'],'label':o['label']+' 相纸','description':'Derived photo frame',
                           'derived':True,'request':True,'style':{},'rotation':o.get('rotation',0)})
    return result


def make_window(canvas_size, box, angle=0, radius=0, spec=None):
    if spec is not None:
        from photo_windows import canvas_mask
        return canvas_mask(canvas_size,box,angle,spec)
    l,t,r,b=box;tile=rounded_mask((r-l,b-t),radius)
    if angle:tile=tile.rotate(-angle,Image.Resampling.BICUBIC,expand=True)
    result=Image.new('L',canvas_size)
    result.paste(tile,(round((l+r-tile.width)/2),round((t+b-tile.height)/2)))
    return result


def has_ink(image,box):
    patch=np.asarray(image.crop(box));visible=patch[:,:,3]>128
    if visible.mean()<.45:return False
    values=patch[:,:,:3].mean(2)[visible]
    return bool(len(values) and np.percentile(values,95)-np.percentile(values,5)>55)


def window_coverage(image, photo, size):
    """Visible alpha inside this photo's window, not across the carrier canvas."""
    mask=make_window(size,photo.get('window_bbox',photo['bbox']),
                     photo.get('rotation',0),photo.get('style',{}).get('corner_radius',0),photo.get('photo_window_spec'))
    selected=np.asarray(mask)>128
    return float((np.asarray(image.getchannel('A'))[selected]>16).mean()) if selected.any() else 1.0


def frame_window_check(image, photo, size):
    """Inspect the shaped, rotated opening without erasing returned pixels."""
    box=photo.get('window_bbox',photo['bbox'])
    mask=make_window(size,box,photo.get('rotation',0),
                     photo.get('style',{}).get('corner_radius',0),photo.get('photo_window_spec'))
    # Inward tolerance avoids confusing rough frame edges with the old photo.
    inset=max(1,round(min(box[2]-box[0],box[3]-box[1])*.02))
    bounds=mask.getbbox()
    if bounds is None:
        return {'interior_visible_fraction':None,'interior_pixels':0,'edge_tolerance':inset}
    tile=mask.crop(bounds)
    core=ImageOps.expand(tile,border=inset,fill=0).filter(ImageFilter.MinFilter(2*inset+1))
    core=core.crop((inset,inset,inset+tile.width,inset+tile.height))
    selected=np.asarray(core)>128
    count=int(selected.sum())
    fraction=float((np.asarray(image.getchannel('A').crop(bounds))[selected]>16).mean()) if count else None
    return {'interior_visible_fraction':round(fraction,6) if fraction is not None else None,
            'interior_pixels':count,'edge_tolerance':inset}


def attach(run,scene,config):
    run=Path(run);folder=run/'assets/reveal';folder.mkdir(parents=True,exist_ok=True)
    repair=load_plan(run,scene)
    requested=targets(scene);save(folder/'targets.json',requested)
    assets,receipts=acquire(run,scene,requested,config)
    assets,merged_owners=compose_groups(run,scene,assets,repair)
    window_repairs={(w['overlay_id'],w['photo_id']):w for w in repair.get('windows',[])}
    omitted={o['id']:o['reason'] for o in repair.get('omit',[])}
    missing={w['overlay_id'] for w in repair.get('windows',[])}-assets.keys()
    if missing:raise ValueError('Missing window repair sources: '+', '.join(sorted(missing)))
    by_id={o['id']:o for o in scene['objects']};photos=[o for o in scene['objects'] if is_customer_photo(o)]
    width,height=scene['reference_size'];records=[];recovered={};owners={};used_photos=set();gate_records=[]
    # A clean background is the residual canvas, never a giant foreground extraction target.
    for o in scene['objects']:
        if o['kind']!='background' or o.get('background_type')!='reference_clean':continue
        oid=o['id'];source=assets.get(oid)
        rec={'id':oid,'input_bbox':o['bbox'],'action':'background_cleanup'};records.append(rec)
        def fallback_background(reason, gate=None):
            o['reveal_warning']=reason
            o.pop('recovered',None)
            o['gate_local']=True
            gate=gate or {'id':oid,'status':'rejected','basis':'program',
                         'input_key':fingerprint([scene['sources'],o,reason]),'issues':[reason]}
            gate.update(use_local=True,local_fallback=True,semantic_verified=False)
            gate_records.append(gate);o['gate']={k:gate[k] for k in ['status','basis','input_key']}
            rec.update(action='local_background_fallback',error=reason,gate=gate)
        if source is None:
            chain_path=folder/'chain-report.json'
            detail=read(chain_path).get('error') if chain_path.exists() else None
            fallback_background(detail or 'Reference background cleanup incomplete; inspect assets/reveal/chain-report.json')
            continue
        try:
            with Image.open(verify_source(source)) as raw:image=raw.convert('RGBA')
        except (OSError,ValueError) as exc:
            fallback_background('Reference background source unavailable: '+str(exc))
            continue
        file=folder/(oid+'.png');image.save(file)
        reference_crop=folder/(oid+'-reference.png');foreground=folder/(oid+'-preview.png')
        with Image.open(scene['reference']['file']) as ref:ref.crop(o['bbox']).save(reference_crop)
        image.crop(o['bbox']).save(foreground)
        candidate={'file':str(file),'sha256':sha(file),'source':source,'reference_crop':str(reference_crop),
                   'foreground':str(foreground),'action':'background_cleanup'}
        gate=inspect_asset(image,o,scene,source,repair,[],[],[])
        gate.update(candidate=candidate,reference_crop=str(reference_crop),foreground=str(foreground),semantic_verified=False)
        if gate['status']=='accepted':
            gate_records.append(gate);o['gate']={k:gate[k] for k in ['status','basis','input_key']}
            o['recovered']=candidate
        else:fallback_background('; '.join(gate['issues']) or 'Reference background was not admitted',gate)
        rec.update(source=source,clean=candidate,gate=gate)
    for target in requested:
        oid=target['id'];o=target if target.get('derived') else by_id[oid]
        expected=request_object(target,scene['reference_size'],config.get('padding',.1))
        rec={'id':oid,'input_bbox':target['bbox'],'request_bbox':expected['bbox'],'padding':expected['padding']};records.append(rec)
        if oid in merged_owners:
            o['recovery_owner']=merged_owners[oid]
            rec.update(action='grouped_member',owner=merged_owners[oid],source=assets.get(oid))
            continue
        def fallback(reason):
            rec.update(action='local_fallback',reason=reason)
            obj=by_id[target['photo_id']] if target.get('derived') else o
            obj['reveal_warning']=reason
            if not is_customer_photo(obj):
                if obj['kind']!='frame':
                    obj['method']='local' if obj.get('style',{}).get('shape') else 'placeholder'
                gate={'id':oid,'status':'rejected','basis':'program','input_key':fingerprint([scene['sources'],target,reason]),
                      'issues':[reason],'use_local':False}
                gate_records.append(gate);obj['gate']={k:gate[k] for k in ['status','basis','input_key']}
                if obj['kind']!='frame' and obj['method']=='local':obj['gate_local']=True;gate['local_fallback']=True
        if oid not in assets:
            if target.get('request'):fallback('Reveal target unavailable; local fallback requires review')
            else:rec['action']='local_primitive'
            continue
        source=assets[oid]
        # The legacy service widens very thin boxes; no arbitrary layout remapping.
        tolerance=max(2,np.ceil(8*max(width,height)/1024))
        if any(abs(a-b)>tolerance for a,b in zip(source['original_bbox'],target['bbox'])):
            fallback('Cached target box differs from current design; request a matching layer');continue
        verified=verify_source(source)
        try:
            with Image.open(verified) as raw:
                api_size=raw.size;image=raw.convert('RGBA').resize((width,height),Image.Resampling.LANCZOS)
        except OSError:
            fallback('Unreadable extracted image');continue
        rec.update(source=source,api_size=list(api_size),scale_to_reference=[width/api_size[0],height/api_size[1]])
        if image.getchannel('A').getextrema()[1]<16:
            fallback('Reveal layer is empty or nearly transparent');continue
        enclosed=[p for p in photos if p.get('mode')=='cover' and contains(o['bbox'],p['bbox'])>.94 and area(p['bbox'])<area(o['bbox'])]
        rec['action']='recover_group' if source.get('recovery_group') else 'reuse_layer'
        notes=[]
        if oid in omitted:
            image.putalpha(0);rec.update(action='omit_after_review',reason=omitted[oid])
            o['reveal_warning']='Returned layer omitted after visual review: '+omitted[oid]
        if len(enclosed)>1 or (area(o['bbox'])/(width*height)>.22 and sum(contains(o['bbox'],p['bbox'])>.35 for p in photos)>1):
            notes.append('Layer spans multiple photos; inspect actual pixels, retained without automatic rejection')
        related=[p for p in photos if p['id'] in photo_ids(o) or p.get('parent_id')==oid]
        if oid in omitted:related=[]
        if frame_hint(o) and not related:notes.append('Frame has no explicit photo relationship; returned pixels retained')
        for photo in related:
            if photo['id'] in used_photos or (photo.get('mode')!='cover' and not photo.get('photo_window_spec')):
                notes.append('Photo window not attached: ambiguous ownership or unsupported photo mode')
                continue
            if target.get('derived'):
                l,t,r,b=photo['bbox'];inset=photo_layout((r-l,b-t),photo['style'])
                box=photo.get('window_bbox',[l+inset[0],t+inset[1],l+inset[2],t+inset[3]])
                if photo.get('rotation',0):
                    # Shift the window center around the outer card's rotation center.
                    cx,cy=(l+r)/2,(t+b)/2;wx,wy=(box[0]+box[2])/2,(box[1]+box[3])/2
                    a=math.radians(photo['rotation']);dx=(wx-cx)*math.cos(a)-(wy-cy)*math.sin(a)+cx-wx
                    dy=(wx-cx)*math.sin(a)+(wy-cy)*math.cos(a)+cy-wy
                    box=[round(box[0]+dx),round(box[1]+dy),round(box[2]+dx),round(box[3]+dy)]
                # Legacy outer-card coordinates convert once to an inner window.
                photo['bbox']=box
                photo['style']={k:v for k,v in photo['style'].items() if k not in ['card','shadow','outline_width','outline_color']}
            else:box=photo.get('window_bbox',photo['bbox'])
            mask=make_window((width,height),box,photo.get('rotation',0),photo['style'].get('corner_radius',0),photo.get('photo_window_spec'))
            mp=folder/(oid+'-'+photo['id']+'-window.png');mask.save(mp)
            photo['photo_window']={'file':str(mp),'sha256':sha(mp)}
            photo['reveal_frame_id']=oid
            selected=np.asarray(mask)>128;alpha=np.asarray(image.getchannel('A'))
            fraction=float((alpha[selected]>16).mean()) if selected.any() else 0
            item={'photo_id':photo['id'],'window':photo['photo_window'],'window_bbox':box,
                  'original_visible_fraction':round(fraction,6),'action':'preserve_returned_pixels'}
            operation=window_repairs.get((oid,photo['id']))
            if operation:
                pad=operation.get('padding',0)
                clear_box=[round(box[0]-pad),round(box[1]-pad),round(box[2]+pad),round(box[3]+pad)]
                clear_mask=make_window((width,height),clear_box,photo.get('rotation',0),photo['style'].get('corner_radius',0),photo.get('photo_window_spec'))
                image.putalpha(ImageChops.multiply(image.getchannel('A'),ImageChops.invert(clear_mask)))
                cp=folder/(oid+'-'+photo['id']+'-clear.png');clear_mask.save(cp)
                item.update(action='clear_photo_window',reason=operation['reason'],padding=pad,
                            clear_mask={'file':str(cp),'sha256':sha(cp)})
                rec['action']='clear_photo_window'
            elif fraction>.05:
                notes.append('Window '+photo['id']+' contains visible pixels; inspect for old photos or intentional decoration')
            rec.setdefault('windows',[]).append(item);used_photos.add(photo['id'])
        image,edit_records,repair_errors=apply_edits(run,image,oid,repair)
        if edit_records:rec['edits']=edit_records
        # Relationships do not imply foreground placement. Preserve the explicit
        # analysis order, including opaque backings below their customer photos.
        coverage={p['id']:window_coverage(image,p,(width,height)) for p in related
                  if p.get('mode')=='cover' or (o['kind']=='frame' and p.get('photo_window_spec'))}
        if target.get('derived'):
            # Legacy cards have no explicit overlay position. Only add their frame
            # above the photo when every associated opening is substantially clear.
            foreground=bool(coverage) and all(v<=.05 for v in coverage.values())
            scene['layer_order'].insert(scene['layer_order'].index(target['photo_id'])+int(foreground),oid)
            rec['layer_placement']='derived_above_clear_window' if foreground else 'derived_foreground_blocked'
            if not foreground:notes.append('Derived frame foreground placement blocked: photo window is not sufficiently transparent; inspect for repair')
        order=scene['layer_order']
        for p in related:
            if p['id'] not in coverage:continue
            above=order.index(oid)>order.index(p['id'])
            occupied=coverage[p['id']]>.05
            rec.setdefault('layer_checks',[]).append({'photo_id':p['id'],
                'position':'above_photo' if above else 'below_photo',
                'visible_fraction':round(coverage[p['id']],6),
                'review_required':above and occupied,
                'action':'preserve_analysis_order' if not target.get('derived') else rec['layer_placement']})
            if o['kind']=='frame':
                rec['layer_checks'][-1].update(frame_window_check(image,p,(width,height)))
            if above and occupied:
                notes.append('Foreground carrier window '+p['id']+' still contains visible pixels; inspect for occlusion or intentional decoration before clearing')
        if notes:rec['notes']=notes;o['reveal_notes']=notes
        file=folder/(oid+'.png');image.save(file)
        reference_crop=folder/(oid+'-reference.png');foreground=folder/(oid+'-preview.png')
        with Image.open(scene['reference']['file']) as ref:ref.crop(source['bbox']).save(reference_crop)
        image.crop(source['bbox']).save(foreground)
        candidate={'file':str(file),'sha256':sha(file),'source':source,'reference_crop':str(reference_crop),
                   'foreground':str(foreground),'action':rec['action']}
        text_candidates=[]
        for child in scene['objects']:
            if child['id']==oid or child['kind'] not in ['text','overlay']:continue
            owner=child.get('embedded_in') or child.get('parent_id')
            if owner in merged_owners:owner=merged_owners[owner]
            if owner==oid:text_candidates.append({'id':child['id'],'relation':'explicit'})
            elif child['kind']=='text' and not owner and contains(o['bbox'],child['bbox'])>.95 and has_ink(image,child['bbox']):
                text_candidates.append({'id':child['id'],'relation':'possible'})
        gate=inspect_asset(image,target,scene,source,repair,rec.get('layer_checks',[]),text_candidates,repair_errors)
        gate.update(candidate=candidate,reference_crop=str(reference_crop),foreground=str(foreground))
        gate_records.append(gate);rec['gate']=gate;o['gate']={k:gate[k] for k in ['status','basis','input_key']}
        rec['clean']=candidate
        if gate['status']=='accepted':
            o['recovered']=candidate;recovered[oid]=image;owners[oid]=o;o['gate_embedded_ids']=gate['embedded_ids']
        else:
            o['reveal_warning']='Asset gate '+gate['status']+': '+', '.join(gate['issues'])
            if o['kind']!='frame' and o.get('style',{}).get('shape') and not source.get('recovery_group'):
                o['gate_local']=True;o['method']='local';gate['local_fallback']=True
            elif gate['use_local']:
                gate['issues'].append('local_fallback_not_supported')
        if target.get('derived'):
            obj={k:v for k,v in o.items() if k not in ['derived','request']};obj['text_unresolved']=False
            scene['objects'].append(obj);by_id[oid]=obj
    suppressed={};candidates={}
    for gate in gate_records:
        for child in gate.get('text_candidates',[]):
            if child['id'] not in gate.get('embedded_ids',[]):candidates.setdefault(child['id'],[]).append(gate['id'])
    for obj in scene['objects']:
        if is_customer_photo(obj) or obj.get('text_origin'):continue
        owner=obj.get('embedded_in') or obj.get('parent_id')
        if owner in merged_owners:owner=merged_owners[owner]
        if owner in by_id and by_id[owner].get('reveal_frame_id'):owner=by_id[owner]['reveal_frame_id']
        if owner in recovered and obj['id'] in by_id[owner].get('gate_embedded_ids',[]) and contains(owners[owner]['bbox'],obj['bbox'])>.9:
            obj['embedded_owner']=owner;suppressed[obj['id']]=owner
        elif obj['kind']=='text':
            possible=[oid for oid,image in recovered.items() if contains(owners[oid]['bbox'],obj['bbox'])>.95 and has_ink(image,obj['bbox'])]
            if possible:candidates[obj['id']]=possible
    index={'reference':scene['reference'],'records':records,'embedded_content':suppressed,
           'unconfirmed_embedded_text':candidates,'service_tasks':receipts,'config':config,
           'recovered_groups':scene.get('recovered_groups',[]),'postprocess':'preserve-unless-explicit-repair-v1'}
    save(folder/'index.json',index)
    scene['reveal_index']={'file':str(folder/'index.json'),'sha256':sha(folder/'index.json')}
    scene['reveal_unconfirmed_text']=candidates
    publish_gate(run,scene,gate_records)
    return index


def verify(scene):
    from asset_gate import verify as verify_gate
    verify_gate(scene)
    if scene.get('reveal_index'):verify_source(scene['reveal_index'])
    if scene.get('recovery_plan'):verify_source(scene['recovery_plan'])
    for obj in scene['objects']:
        if obj.get('recovered'):
            verify_source(obj['recovered']);verify_asset(obj['recovered']['source'])
        if obj.get('photo_window'):verify_source(obj['photo_window'])
