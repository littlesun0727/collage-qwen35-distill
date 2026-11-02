"""Reveal poster adapter. Cache reads are offline; --reveal enables new requests."""
import base64
import math
import os
import time
import threading
import json as jsonlib
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from pathlib import Path
import numpy as np
from PIL import Image
from common import cache_directory, read, save, sha, fingerprint, event, now
from collage.config import Config, env_value

BASE = 'https://api.research.360.cn/v1'
_submission_budget_lock=threading.Lock()
SUBMIT_RETRY_CODES={502,503,504}
QUERY_RETRY_CODES={408,429,500,502,503,504}


def reserve_submission(task, budget_root, limit=None):
    """Reserve each new submission durably, including retries and interrupted submits."""
    from reveal_plan import MAX_REQUESTS
    task=Path(task);budget_root=Path(budget_root)
    if limit is None:
        budget_file=budget_root/'submission-budget.json'
        limit=read(budget_file)['limit'] if budget_file.exists() else MAX_REQUESTS
    if type(limit) is not int or limit < 1:raise ValueError('Reveal submission limit must be a positive integer')
    with _submission_budget_lock:
        marker=task/'budget-reserved.json'
        if marker.exists():
            reservation=read(marker)
            if 'budget_root' not in reservation:
                save(marker,{**reservation,'budget_root':str(budget_root.resolve())})
            return
        used={p.parent for name in ['budget-reserved.json','submission_started.json','submit_response.json'] for p in budget_root.rglob(name)}
        if task not in used and len(used)>=limit:raise ValueError(f'Hard limit: at most {limit} Reveal submissions per run')
        save(marker,{'reserved_at':now(),'limit':limit,'budget_root':str(budget_root.resolve())})


def http(url, json=None, headers=None, timeout=120):
    """Small standard-library transport, shared with injectable HTTP test doubles."""
    payload=jsonlib.dumps(json).encode('utf-8') if json is not None else None
    request=Request(url,data=payload,headers={**(headers or {}),**({'Content-Type':'application/json'} if payload else {})})
    with urlopen(request,timeout=max(timeout) if isinstance(timeout,tuple) else timeout) as response:
        content=response.read()
    class Response:
        def raise_for_status(self):pass
        def json(self):return jsonlib.loads(self.content)
    result=Response();result.content=content;return result


def same_image(a, b):
    with Image.open(a) as x, Image.open(b) as y:
        return x.size == y.size and np.array_equal(np.asarray(x.convert('RGB')), np.asarray(y.convert('RGB')))


def request_object(target, size, padding=.1, padding_mode='capped'):
    """Expand an original bbox, enforce the API minimum, then clamp to the reference."""
    if not math.isfinite(padding) or not 0 <= padding <= 1:
        raise ValueError('Reveal padding must be a finite fraction in 0..1')
    if padding_mode not in ['capped','ratio']:
        raise ValueError('Unknown Reveal padding mode')
    l,t,r,b=target['bbox'];w,h=size
    if padding_mode=='ratio':
        px=(r-l)*padding;py=(b-t)*padding
    else:
        scale=min(1,1024/max(w,h))
        edge=min(padding*min(r-l,b-t),6/scale)
        px=py=edge
    box=[max(0,math.floor(l-px)),max(0,math.floor(t-py)),
         min(w,math.ceil(r+px)),min(h,math.ceil(b+py))]
    minimum=math.ceil(8*max(w,h)/1024)
    for axis,limit in [(0,w),(1,h)]:
        span=min(limit,minimum)
        if box[axis+2]-box[axis]<span:
            box[axis]=max(0,min(limit-span,math.floor((box[axis]+box[axis+2]-span)/2)))
            box[axis+2]=box[axis]+span
    result={'id':target['id'],'bbox':box,'original_bbox':list(target['bbox']),'padding':padding}
    if padding_mode!='ratio':result['padding_mode']=padding_mode
    return result


def matches_request(saved, desired):
    if saved['id']!=desired['id'] or saved['bbox']!=desired['bbox']:return False
    if saved.get('padding_mode','ratio')!=desired.get('padding_mode','ratio'):return False
    if 'original_bbox' not in saved:return desired['padding']==0
    return saved['original_bbox']==desired['original_bbox'] and saved.get('padding')==desired['padding']


def downloads_ready(task):
    qp=task/'query_response.json'
    if not qp.exists():return False
    result=read(qp)
    if result.get('status')!='done':return False
    return all((task/f'layers_aug_{i:02d}.png').exists() or (task/f'layers_base_{i:02d}.png').exists()
               for i in range(1,len(result['output'].get('boxes_mapping_index',[]))+1))


def load_cache(folder, reference, expected=None, allow_partial=False):
    folder = Path(folder)
    if not same_image(folder/'input.png', reference):
        raise ValueError('Reveal cache reference does not match this task')
    found = {}; receipts = []
    for task in sorted(folder.glob('batch_*')):
        if not (task/'query_response.json').is_file(): continue
        meta = read(task/'request_meta.json'); result = read(task/'query_response.json')
        if result.get('status') != 'done': continue
        if allow_partial and not downloads_ready(task):continue
        relevant=[o for o in meta['objects'] if expected is None or (o['id'] in expected and matches_request(o,expected[o['id']]))]
        if not relevant:continue
        if meta.get('model') != 'ecommerce_layer' or meta.get('version') != 'v1.1':
            raise ValueError('Unsupported Reveal cache model/version')
        output = result['output']; indices = output.get('boxes_mapping_index')
        if not isinstance(indices,list) or len(set(indices)) != len(indices):
            raise ValueError('Missing or duplicate Reveal mapping')
        for j, index in enumerate(indices,1):
            if type(index) is not int or not 0 <= index < len(meta['objects']):
                raise ValueError('Invalid Reveal target index')
            obj = meta['objects'][index]; oid = obj['id']
            if obj not in relevant:continue
            if oid in found: raise ValueError('Duplicate Reveal object ID')
            file = task/f'layers_aug_{j:02d}.png'
            if not file.is_file(): file=task/f'layers_base_{j:02d}.png'
            with Image.open(file) as im:
                if 'A' not in im.getbands(): raise ValueError('Reveal layer lacks alpha')
                im.verify()
            found[oid]={'file':str(file.resolve()),'sha256':sha(file),'bbox':obj['bbox'],
                        'original_bbox':obj.get('original_bbox',obj['bbox']),'padding':obj.get('padding',0)}
            if 'padding_mode' in obj:found[oid]['padding_mode']=obj['padding_mode']
        receipts.append({'task':str(task),'generation_seconds':result.get('generation_time'),
                         'targets':sum(meta['objects'][i] in relevant for i in indices),'requested_ids':[o['id'] for o in relevant],
                         'task_id':result.get('task_id') or (read(task/'submit_response.json').get('task_id') if (task/'submit_response.json').exists() else None),
                         'timings':read(task/'timing.json') if (task/'timing.json').exists() else None})
    return found, receipts


def api_key(path=None):
    path = path or os.getenv('COLLAGE_REVEAL_ENV', Config.reveal_key_file)
    key = os.environ.get('360_API_KEY') or env_value(path, '360_API_KEY')
    if not key: raise ValueError(f'360_API_KEY missing; configure {path} or the 360_API_KEY environment variable')
    return key


def fetch_batch(task, reference, objects, key, timeout=360, post=None, get=None, sleep=time.sleep):
    """Retry explicit gateway failures within the run budget; retain uncertain submits."""
    post=post or http;get=get or http
    task=Path(task);task.mkdir(parents=True,exist_ok=True)
    identity={'reference_sha256':sha(reference),'objects':objects,'model':'ecommerce_layer',
              'version':'v1.1','seed':42,'steps':28}
    digest=fingerprint(identity);mp=task/'request_meta.json'
    if mp.exists() and read(mp)['sha256']!=digest:raise ValueError('Reveal request changed in an existing task')
    save(mp,{**identity,'sha256':digest})
    deadline=time.monotonic()+timeout
    timing_file=task/'timing.json';timings=read(timing_file) if timing_file.exists() else {'started_at':now()}
    def elapsed(stage,started):
        timings[stage]=round(timings.get(stage,0)+time.monotonic()-started,3);timings['updated_at']=now();save(timing_file,timings)
    def request(endpoint,payload):
        left=deadline-time.monotonic()
        if left<=0:raise TimeoutError('Reveal task timeout; resume with the same cache')
        started=time.monotonic()
        try:
            response=post(BASE+'/'+endpoint,json=payload,headers={'Authorization':'Bearer '+key},timeout=(min(30,left),min(120,left)))
            response.raise_for_status();return response.json()
        finally:elapsed('submit_seconds' if endpoint=='submit_task' else 'query_seconds',started)
    def pause_retry(endpoint,attempt,code=None):
        delay=2**attempt
        if deadline-time.monotonic()<=delay:
            raise TimeoutError('Reveal retry deadline reached; resume with the same cache')
        report=task/'http-retries.json'
        records=read(report) if report.exists() else []
        records.append({'endpoint':endpoint,'after_attempt':attempt,'http_status':code,'wait_seconds':delay,'at':now()})
        save(report,records);sleep(delay)
    submitted=task/'submit_response.json'
    if not submitted.exists():
        payload={'model':'ecommerce_layer','version':'v1.1','pipeline_type':'crop','time_out':3600,
                 'seed':42,'steps':28,'image_boxes':[o['bbox'] for o in objects],
                 'input':[{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(Path(reference).read_bytes()).decode()}]}
        marker=task/'submission_started.json'
        reservation=task/'budget-reserved.json'
        budget_root=Path(read(reservation).get('budget_root',task)) if reservation.exists() else task
        while not submitted.exists():
            previous=read(marker) if marker.exists() else None
            attempt=1
            if previous:
                code=previous.get('http_status')
                if code not in SUBMIT_RETRY_CODES:
                    raise ValueError('Reveal submit outcome unknown or non-retryable; do not resubmit automatically')
                attempt=previous.get('attempt',1)+1
                if attempt>3:raise ValueError('Reveal gateway retries exhausted; at most 3 submission attempts')
                # Each new HTTP submit consumes a durable slot, including gateway failures.
                retry_slot=task/'submit-retries'/f'attempt-{attempt:02d}'
                pause_retry('submit_task',attempt-1,code)
                try:reserve_submission(retry_slot,budget_root)
                except ValueError as exc:
                    raise ValueError(f'Reveal submit HTTP {code}; retry stopped: {exc}') from exc
            else:reserve_submission(task,budget_root)
            started={'request_sha256':digest,'attempt':attempt,'started_at':now()}
            save(marker,started)
            history=task/'submit-history'/f'attempt-{attempt:02d}.json'
            save(history,started)
            try:
                response=request('submit_task',payload)
                save(submitted,response)
                save(history,{**started,'finished_at':now(),'task_id':response.get('task_id'),
                              'response_status':response.get('response_status')})
            except (OSError,ValueError) as exc:
                code=getattr(exc,'code',None) if isinstance(exc,HTTPError) else None
                failure={**started,'finished_at':now(),'http_status':code,'error_type':type(exc).__name__}
                save(marker,failure);save(history,failure)
                if code not in SUBMIT_RETRY_CODES or attempt>=3:raise
    result=read(submitted)
    if result.get('response_status')!=0 or not result.get('task_id'):raise ValueError('Reveal submission rejected; inspect saved response')
    query={'model':'ecommerce_layer','version':'v1.1','task_id':result['task_id']}
    qp=task/'query_response.json';result=read(qp) if qp.exists() else {}
    while result.get('status')!='done':
        if result.get('status') in ['failed','not_found'] or result.get('response_status')==-1:
            raise ValueError('Reveal task failed; saved task will not be resubmitted')
        for attempt in range(1,4):
            try:
                result=request('query_task',query);save(qp,result);break
            except (OSError,ValueError) as exc:
                code=getattr(exc,'code',None)
                transient=(isinstance(exc,HTTPError) and code in QUERY_RETRY_CODES) or (
                    isinstance(exc,(URLError,TimeoutError,ConnectionError)) and not isinstance(exc,HTTPError))
                if not transient or attempt>=3:raise
                pause_retry('query_task',attempt,code)
        if result.get('status')!='done':
            started=time.monotonic();sleep(min(5,max(0,deadline-time.monotonic())));elapsed('poll_wait_seconds',started)
    download_errors=[]
    for field in ['layers_base','layers_aug']:
        for i,url in enumerate(result['output'].get(field,[])):
            path=task/f'{field}_{i:02d}.png'
            if path.exists():
                try:
                    with Image.open(path) as im:im.verify()
                    continue
                except (OSError, ValueError):pass
            for attempt in range(3):
                started=time.monotonic()
                try:
                    left=deadline-time.monotonic()
                    if left<=0:raise TimeoutError('Reveal download timeout; resume existing task')
                    response=get(url,timeout=(min(30,left),min(120,left)));response.raise_for_status()
                    tmp=path.with_suffix('.part');tmp.write_bytes(response.content)
                    with Image.open(tmp) as im:im.verify()
                    tmp.replace(path)
                    break
                except (OSError, ValueError) as exc:
                    code=getattr(exc,'code',None)
                    transient=(isinstance(exc,HTTPError) and code in {408,429,500,502,503,504}) or (isinstance(exc,(URLError,TimeoutError,ConnectionError)) and not isinstance(exc,HTTPError))
                    if transient and attempt<2 and deadline-time.monotonic()>attempt+1:
                        sleep(attempt+1)
                        continue
                    download_errors.append({'file':path.name,'error':type(exc).__name__+(f' HTTP {code}' if code else ''),'attempts':attempt+1})
                    break
                finally:elapsed('download_seconds',started)
    save(task/'download-report.json',{'errors':download_errors,'complete':not download_errors,'at':now()})
    if download_errors:
        raise RuntimeError('Reveal download incomplete: '+'; '.join(e['file']+': '+e['error'] for e in download_errors))


def acquire(run, scene, targets, config):
    if any(o.get('background_type')=='reference_clean' for o in scene['objects']):
        config={**config,'layout':'full'}
    # First-class frames require photo context even for legacy-default callers.
    if any(t.get('kind')=='frame' for t in targets) and config.get('layout','legacy')=='legacy':
        config={**config,'layout':'full'}
    if config.get('layout','legacy') == 'full':
        from reveal_chain import acquire as chain_acquire
        return chain_acquire(run,scene,targets,config)
    if config.get('layout','legacy') == 'grouped':
        from reveal_grouped import acquire as grouped_acquire
        return grouped_acquire(run,scene,targets,config)
    if config.get('layout','legacy')!='legacy':raise ValueError('Unknown Reveal layout')
    run=Path(run);reference=scene['reference']['file'];found={};receipts=[];errors=[];attempted=set()
    expected={t['id']:request_object(t,scene['reference_size'],config.get('padding',.1),
                                     config.get('padding_mode','ratio')) for t in targets}
    save(run/'assets/reveal/request-targets.json',list(expected.values()))
    def ingest(folder):
        existing,previous=load_cache(folder,reference,expected,allow_partial=Path(folder)==run/'assets/reveal/api');found.update(existing)
        receipts[:]=[r for r in receipts if Path(r['task']).parent!=Path(folder)]+previous
        attempted.update(oid for r in previous for oid in r['requested_ids'])
    if config.get('cache'):ingest(config['cache'])
    root=run/'assets/reveal/api'
    def dispatch(task,objects):
        start=time.monotonic();event(run,'reveal_request_started',task=str(task),targets=[o['id'] for o in objects])
        try:
            reserve_submission(task,run/'assets/reveal')
            fetch_batch(task,root/'input.png',objects,api_key(config.get('key_file')),config.get('timeout',360))
            event(run,'reveal_request_finished',task=str(task),elapsed_seconds=round(time.monotonic()-start,3));return True
        except Exception as exc:
            error=type(exc).__name__+(' HTTP '+str(exc.code) if hasattr(exc,'code') else '')
            event(run,'reveal_request_failed',task=str(task),error=error,elapsed_seconds=round(time.monotonic()-start,3))
            errors.append({'task':str(task),'error':error,'requested_ids':[o['id'] for o in objects]});return False
    if root.exists():
        if not same_image(root/'input.png',reference):raise ValueError('Reveal internal cache reference mismatch')
        for task in sorted(root.glob('batch_*')):
            meta=task/'request_meta.json'
            if not meta.exists():meta=task/'batch-input.json'
            if not meta.exists():continue
            objects=read(meta)['objects']
            relevant=[o for o in objects if o['id'] in expected and matches_request(o,expected[o['id']])]
            if not relevant:continue
            attempted.update(o['id'] for o in relevant)
            if not downloads_ready(task) and config.get('remote'):
                if not dispatch(task,objects):
                    ingest(root);return found,receipts+errors
        ingest(root)
    missing=[expected[t['id']] for t in targets if t.get('request',True) and t['id'] not in found and t['id'] not in attempted]
    if missing and config.get('remote'):
        root.mkdir(parents=True,exist_ok=True)
        if not (root/'input.png').exists():
            with Image.open(reference) as im:im.convert('RGB').save(root/'input.png')
        for start in range(0,len(missing),20):
            chunk=missing[start:start+20]
            task=cache_directory(root,'batch',{'reference_sha256':sha(reference),'objects':chunk},'batch-input.json')
            if not dispatch(task,chunk):break
            ingest(root)
    return found,receipts+errors
