"""Execute crop batches, retain local raw layers, restore to reference coordinates."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import time
from PIL import Image
from common import cache_directory,read,save,sha,fingerprint,event
from reveal_plan import plan,preview


def resize_mapping_matches(source_size, target_size, box_mapping):
    """Verify the service's independent x/y resize against every returned box."""
    if box_mapping is None:
        return False
    original, resized = box_mapping
    if not isinstance(original, list) or not isinstance(resized, list) or not original or len(original) != len(resized):
        return False
    rw, rh = source_size
    cw, ch = target_size
    for before, after in zip(original, resized):
        if not isinstance(before, list) or not isinstance(after, list) or len(before) != 4 or len(after) != 4:
            return False
        for i, (a, b) in enumerate(zip(before, after)):
            if (type(a) not in (int, float) or type(b) not in (int, float)
                    or not math.isfinite(a) or not math.isfinite(b)
                    or abs(a * (rw / cw if i % 2 == 0 else rh / ch) - b) > 2):
                return False
    return True


def restore(raw, crop, reference_size, destination, box_mapping=None):
    with Image.open(raw) as source:
        if 'A' not in source.getbands():raise ValueError('Reveal layer lacks alpha')
        rw,rh=source.size;cw,ch=crop[2]-crop[0],crop[3]-crop[1]
        if abs(rw/rh-cw/ch)>.02*max(rw/rh,cw/ch):
            # The service may round dimensions independently. Accept this only
            # when its returned box coordinates prove the same x/y transform.
            if not resize_mapping_matches((rw, rh), (cw, ch), box_mapping):
                raise ValueError('Unexpected returned crop aspect ratio: no consistent box mapping')
        tile=source.convert('RGBA').resize((cw,ch),Image.Resampling.LANCZOS)
    out=Image.new('RGBA',tuple(reference_size));out.paste(tile,(crop[0],crop[1]))
    out.save(destination)
    return [cw/rw,ch/rh]


def acquire(run,scene,targets,config):
    from reveal import fetch_batch,api_key,downloads_ready,reserve_submission
    run=Path(run);reference=Path(scene['reference']['file']);reference_hash=sha(reference)
    p=plan(scene,targets,config.get('padding',.1),config.get('layout','full'),
           config.get('padding_mode','ratio'))
    preview(reference,p,run/'assets/reveal')
    if p['blocked']:
        receipts=[{'error':b['reason'],'requested_ids':b['ids'],'blocked':True} for b in p['blocked']]
        save(run/'assets/reveal/grouped-receipts.json',receipts)
        return {},receipts
    if len(p['batches'])>3:raise ValueError('Hard limit: at most 3 Reveal requests')
    root=run/'assets/reveal/grouped';root.mkdir(parents=True,exist_ok=True)
    external=Path(config['cache']) if config.get('cache') else None
    if external and not (external/'request-plan.json').exists():
        raise ValueError('Grouped cache requires request-plan.json; use layout legacy for old caches')
    def work(batch):
        start=time.monotonic();identity={'reference_sha256':reference_hash,'planner':p['version'],'batch':batch}
        local=cache_directory(root,'batch',identity,'group.json');task=local
        if external:
            task=cache_directory(external/'grouped','batch',identity,'group.json',create=False) or local
        readonly=not task.is_relative_to(root);task.mkdir(parents=True,exist_ok=True)
        receipt={'task':str(task),'batch_id':batch['id'],'requested_ids':batch['primary_ids'],
                 'count':batch['count'],'crop_box':batch['crop_box'],'estimated_sampling_gain':batch['estimated_sampling_gain']}
        found={}
        try:
            if (task/'group.json').exists() and read(task/'group.json')!=identity:raise ValueError('Batch cache identity changed')
            if not (task/'group.json').exists():save(task/'group.json',identity)
            ip=task/'input.png'
            with Image.open(reference) as im:crop=im.convert('RGB').crop(batch['crop_box'])
            if ip.exists():
                with Image.open(ip) as im:
                    if im.convert('RGB').tobytes()!=crop.tobytes() or im.size!=crop.size:raise ValueError('Cached crop changed')
            else:crop.save(ip)
            receipt['cache_hit']=downloads_ready(task)
            if not receipt['cache_hit']:
                if not config.get('remote') or readonly:
                    receipt['error']='No complete matching grouped cache'
                else:
                    event(run,'reveal_request_started',task=str(task),targets=batch['primary_ids'])
                    reserve_submission(task,run/'assets/reveal')
                    try:
                        fetch_batch(task,ip,batch['objects'],api_key(config.get('key_file')),config.get('timeout',360))
                    except Exception as exc:
                        receipt['error']=type(exc).__name__+': '+str(exc)
                qp=task/'query_response.json'
                if not qp.exists() or read(qp).get('status')!='done':return found,receipt
            meta=read(task/'request_meta.json')
            if meta['reference_sha256']!=sha(ip) or meta['objects']!=batch['objects']:raise ValueError('Request metadata changed')
            q=read(task/'query_response.json');out=q['output'];indices=out.get('boxes_mapping_index')
            if not isinstance(indices,list) or len(set(indices))!=len(indices):raise ValueError('Invalid Reveal mapping')
            boxes=[o['bbox'] for o in batch['objects']]
            if any(type(i) is not int or not 0<=i<len(boxes) for i in indices):raise ValueError('Invalid Reveal target index')
            if out.get('image_boxes') not in [None,boxes,[boxes[i] for i in indices]]:raise ValueError('Service box echo differs')
            receipt.update(task_id=q.get('task_id') or read(task/'submit_response.json').get('task_id'),
                           status=q['status'],returned=len(indices),timings=read(task/'timing.json') if (task/'timing.json').exists() else {})
            restored=local/'restored';restored.mkdir(parents=True,exist_ok=True)
            for j,i in enumerate(indices,1):
                o=batch['objects'][i]
                if o['role']!='asset':continue
                dest=restored/(fingerprint(o['id'])[:20]+'.png')
                errors=[]
                for raw in [task/f'layers_aug_{j:02d}.png',task/f'layers_base_{j:02d}.png']:
                    if not raw.exists():continue
                    try:
                        scale=restore(raw,batch['crop_box'],scene['reference_size'],dest,
                                      (out.get('image_boxes',boxes),out.get('resized_image_boxes')))
                        break
                    except (OSError, ValueError) as exc:errors.append(type(exc).__name__+': '+str(exc))
                else:
                    receipt.setdefault('object_errors',{})[o['id']]='; '.join(errors) or 'Layer download missing'
                    continue
                found[o['id']]={'file':str(dest.resolve()),'sha256':sha(dest),'bbox':o['reference_bbox'],
                               'original_bbox':o['original_bbox'],'padding':o['padding'],'pixel_scale':scale,
                               'raw_file':str(raw.resolve()),'raw_sha256':sha(raw),'crop_box':batch['crop_box'],'task':str(task)}
                if 'padding_mode' in o:found[o['id']]['padding_mode']=o['padding_mode']
                event(run,'material_extracted',object_id=o['id'],raw_file=str(raw),
                      processed_file=str(dest),cache_hit=receipt['cache_hit'])
            receipt['missing_ids']=[oid for oid in batch['primary_ids'] if oid not in found]
            receipt['available_ids']=list(found)
            if receipt['missing_ids']:
                receipt.setdefault('error','Some target layers are unavailable; resume existing downloads')
        except Exception as exc:
            receipt['error']=type(exc).__name__+': '+str(exc)
        finally:
            receipt['elapsed_seconds']=round(time.monotonic()-start,3)
            event(run,'reveal_group_finished',**receipt)
        return found,receipt
    found={};receipts=[]
    workers=max(1,min(2,int(config.get('workers',2))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for assets,receipt in pool.map(work,p['batches']):found.update(assets);receipts.append(receipt)
    receipts += [{'error':b['reason'],'requested_ids':b['ids'],'blocked':True} for b in p['blocked']]
    save(run/'assets/reveal/grouped-receipts.json',receipts)
    return found,receipts
