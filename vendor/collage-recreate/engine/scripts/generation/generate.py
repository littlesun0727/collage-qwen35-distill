"""Optional yibu generation. No analysis/selection/review model APIs."""
from collage.object_roles import is_customer_photo
import math
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image
from common import read, save, sha, now, event, fingerprint
from scene import load_scene


def remove_key(im, key=(0,255,0)):
    """Chroma key for synthetic backgrounds, never applied to customer photos."""
    a=np.asarray(im.convert('RGBA')).astype('float32')
    green=a[:,:,1]-np.maximum(a[:,:,0],a[:,:,2])
    mask=np.clip((green-15)/65,0,1)
    a[:,:,3]*=1-mask
    a[:,:,1]=np.minimum(a[:,:,1],np.maximum(a[:,:,0],a[:,:,2])+15)
    result=Image.fromarray(np.rint(a).astype('uint8'))
    if not result.getchannel('A').getbbox():raise ValueError('Generated image has no foreground after keying')
    return result


def request_one(run,scene,obj,args):
    key=fingerprint([obj['id'],obj['bbox'],obj['description'],obj.get('text'),scene['reference'],obj.get('member_ids')])
    existing=obj.get('generated')
    if existing and existing.get('input_key')==key and Path(existing['file']).exists() and sha(existing['file'])==existing['sha256']:
        return obj['id'],existing
    folder=Path(run)/'generation'/(obj['id']+'-'+uuid.uuid4().hex[:10]);folder.mkdir(parents=True)
    with Image.open(scene['reference']['file']) as ref:ref.crop(obj['bbox']).convert('RGB').save(folder/'reference.png')
    opaque=obj['kind']=='background'
    prompt=f"Recreate only this design element: {obj['label']}. {obj['description']}\nKeep the crop's full coordinate layout, scale, and margins. Exclude reference customer photos and unrelated neighbors."
    if obj.get('text'):prompt+='\nExact text: '+obj['text']
    prompt+='\nUse an opaque full background.' if opaque else '\nPlace the element on a flat pure #00FF00 chroma-key background. No shadows on the green, no green in the element. Preserve holes and spacing.'
    (folder/'prompt.txt').write_text(prompt,encoding='utf-8')
    box=obj['bbox'];ratio=(box[2]-box[0])/(box[3]-box[1]);ratios=['1:1','2:3','3:2','3:4','4:3','4:5','5:4','9:16','16:9','21:9']
    nearest=min(ratios,key=lambda s:abs(math.log((int(s.split(':')[0])/int(s.split(':')[1]))/ratio)))
    cmd=['node',str(Path(__file__).with_name('yibu.mjs')),'--image',str(folder/'reference.png'),'--prompt',str(folder/'prompt.txt'),'--output',str(folder/'request'),'--credentials',args.credentials,'--ratio',nearest,'--timeout-seconds',str(args.timeout)]
    if args.dry_run:cmd.append('--dry-run')
    event(run,'generation_request_started',object_id=obj['id'],request=str(folder))
    with (folder/'stdout.txt').open('wb') as stdout,(folder/'stderr.txt').open('wb') as stderr:
        proc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=args.timeout+30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    call=read(folder/'request/call.json')
    if proc.returncode or call['status'] not in ['completed','dry_run_verified']:
        event(run,'generation_request_failed',object_id=obj['id'],request=str(folder),status=call.get('status'))
        raise ValueError(f'Image generation failed: {folder}/request/call.json; inspect before retry')
    if args.dry_run:return obj['id'],None
    with Image.open(folder/'request/original-image.bin') as raw:raw.load();image=raw.convert('RGBA')
    if not opaque:image=remove_key(image)
    image.save(folder/'asset.png');record={'file':str(folder/'asset.png'),'sha256':sha(folder/'asset.png'),'request':str(folder),'resource_type':'generated_overlay','input_key':key}
    event(run,'generation_request_finished',object_id=obj['id'],request=str(folder),elapsed_seconds=call.get('elapsed_seconds'))
    return obj['id'],record


def generate(run,args):
    if not args.allow_remote and not args.dry_run:raise ValueError('generate requires --allow-remote')
    if not 1<=args.workers<=4:raise ValueError('workers must be 1..4')
    scene=load_scene(run);objects={o['id']:o for o in scene['objects']}
    if any(set(args.ids)&set(g['member_ids']) for g in scene.get('recovered_groups',[])):
        raise ValueError('Remove the decoration recovery group with recover before generating its members')
    if len(args.ids)!=len(set(args.ids)):raise ValueError('Duplicate generation IDs')
    for oid in args.ids:
        if oid not in objects or is_customer_photo(objects[oid]):raise ValueError('Only known non-photo objects may be generated')
        if objects[oid]['kind']=='frame':raise ValueError('Frames require extraction with their linked photo, not generation')
        if objects[oid].get('text_unresolved'):raise ValueError('Resolve exact text before generation')
    groups=scene.get('generated_groups',[])
    for group in groups:
        overlap=set(args.ids)&set(group['member_ids'])
        if overlap and (overlap!=set(group['member_ids']) or not getattr(args,'group',False)):
            raise ValueError('A fused group must be regenerated with all members and --group; recompile inputs to split it')
    jobs_objects=[objects[oid] for oid in args.ids];new_group=None
    if getattr(args,'group',False):
        if len(args.ids)<2 or any(o['kind'] not in ['text','overlay'] for o in jobs_objects):raise ValueError('A group requires at least two text/overlay objects')
        positions=[scene['layer_order'].index(i) for i in args.ids]
        box=[min(o['bbox'][0] for o in jobs_objects),min(o['bbox'][1] for o in jobs_objects),max(o['bbox'][2] for o in jobs_objects),max(o['bbox'][3] for o in jobs_objects)]
        from overlays import intersect
        for oid in scene['layer_order'][min(positions):max(positions)+1]:
            if oid not in args.ids and intersect(box,objects[oid]['bbox']):raise ValueError('Cannot fuse across an overlapping interleaved layer')
        primary=scene['layer_order'][max(positions)]
        new_group={'primary':primary,'member_ids':list(args.ids),'bbox':box}
        composite={'id':primary,'kind':'overlay','bbox':box,'label':' + '.join(o['label'] for o in jobs_objects),'description':'\n'.join(o['description'] for o in jobs_objects),'text':'\n'.join(o['text'] for o in jobs_objects if o.get('text')),'member_ids':list(args.ids)}
        old=next((g for g in groups if set(g['member_ids'])==set(args.ids)),None)
        if old:composite['generated']=old['source']
        jobs_objects=[composite]
    errors=[];completed=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(request_one,run,scene,obj,args):obj['id'] for obj in jobs_objects}
        for f in as_completed(jobs):
            try:
                oid,record=f.result()
                if record:
                    if new_group:new_group['source']=record
                    else:objects[oid]['generated']=record
                completed.extend(args.ids if new_group else [oid])
            except Exception as exc:errors.append({'id':jobs[f],'error':str(exc)})
    if not args.dry_run:
        if new_group and new_group.get('source'):
            scene['generated_groups']=[g for g in groups if not set(g['member_ids'])&set(args.ids)]+[new_group]
        scene['revision']+=1;scene['pending_generation']=[i for i in scene.get('pending_generation',[]) if i not in completed];save(Path(run)/'scene.json',scene)
        from render import render
        render(run)
    return {'status':'partial' if errors else 'dry_run_verified' if args.dry_run else 'generated','completed':completed,'errors':errors}
