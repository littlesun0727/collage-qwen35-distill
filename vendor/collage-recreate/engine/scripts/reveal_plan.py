"""Full-image batching by capacity, with optional spatial crops. No network access."""
import math
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw
from common import save, fingerprint
from collage.frame_links import photo_ids

VERSION = 'spatial-crop-v2-max3'
FULL_VERSION = 'full-image-v2-chain'
SOFT_LIMIT, HARD_LIMIT = 10, 20
MAX_REQUESTS = 3
CHAIN_MAX_REQUESTS = 16
REQUEST_COST = .12
MIN_CROP_SIDE = 129  # Keep grouped inputs strictly above the API's 128 px minimum.


def crop_geometry(box, size, mode):
    w,h=size;l,t,r,b=box;full=[0,0,w,h]
    margin=max(8,math.ceil(max(r-l,b-t)*.025))
    crop=[max(0,l-margin),max(0,t-margin),min(w,r+margin),min(h,b+margin)]
    if mode=='grouped':
        if min(size)<MIN_CROP_SIDE:
            raise ValueError('Reference is too small to crop both sides above 128 pixels')
        # Extend original-image context; never resize pixels or pad outside it.
        # At an edge, shift the extra context to the opposite side.
        for axis,limit in enumerate(size):
            length=crop[axis+2]-crop[axis]
            if length<MIN_CROP_SIDE:
                start=max(0,min(crop[axis]-(MIN_CROP_SIDE-length)//2,limit-MIN_CROP_SIDE))
                crop[axis],crop[axis+2]=start,start+MIN_CROP_SIDE
    scale=min(1,1024/max(crop[2]-crop[0],crop[3]-crop[1]))
    gain=scale/min(1,1024/max(size));reason='spatial_crop'
    if mode=='full' or gain<1.15:
        crop=full;reason='full_image' if mode=='full' else 'limited_resolution_gain'
        scale=min(1,1024/max(size));gain=1
    return crop,scale,gain,reason


def _partition_original(units,size,mode):
    """Full mode splits only for capacity; grouped mode also scores resolution."""
    if not units:return [],[],None
    if len({o['id'] for u in units for o in u['objects']})>MAX_REQUESTS*HARD_LIMIT:
        return [],[],'More than 60 distinct request boxes cannot fit 3 requests of 20'
    units=sorted(units,key=lambda u:u['ids']);n=len(units)
    total_primary=sum(len(u['ids']) for u in units)
    @lru_cache(None)
    def group(indices):
        objects={}
        for i in indices:
            for o in units[i]['objects']:
                previous=objects.get(o['id'])
                if previous and previous.get('relation')=='explicit':continue
                objects[o['id']]=o
        return {'ids':sorted(oid for i in indices for oid in units[i]['ids']),
                'objects':sorted(objects.values(),key=lambda o:o['id']),
                'bbox':union([units[i]['bbox'] for i in indices]),'notes':[]}
    @lru_cache(None)
    def splits(indices):
        result=set()
        for axis in [0,1]:
            ordered=sorted(indices,key=lambda i:(units[i]['bbox'][axis]+units[i]['bbox'][axis+2],units[i]['ids']))
            for cut in range(1,len(ordered)):
                result.add(tuple(sorted((tuple(sorted(ordered[:cut])),tuple(sorted(ordered[cut:]))))))
        return tuple(sorted(result))
    def score(parts):
        if mode=='full':
            # Among equal-size batch plans, prefer nearby units and less
            # duplicated photo context. Resolution never triggers another batch.
            count=sum(len(group(g)['objects']) for g in parts)
            unique=len({o['id'] for g in parts for o in group(g)['objects']})
            return round(sum(area(group(g)['bbox'])/(size[0]*size[1]) for g in parts)
                         +(count-unique)/HARD_LIMIT,9)
        loss=soft=orphan=0
        for indices in parts:
            g=group(indices);_,scale,_,_=crop_geometry(g['bbox'],size,mode)
            loss+=len(g['ids'])/total_primary*(1-scale)
            soft+=.025*max(0,len(g['objects'])-SOFT_LIMIT)/SOFT_LIMIT
            if len(parts)>1 and len(g['objects'])==1:orphan+=.035
        return round(REQUEST_COST*len(parts)+loss+soft+orphan,9)
    all_indices=tuple(range(n));levels={1:{(all_indices,)}}
    best={}
    for k in range(1,MAX_REQUESTS+1):
        candidates=levels.get(k,set())
        valid=[p for p in candidates if all(len(group(g)['objects'])<=HARD_LIMIT for g in p)]
        if valid:
            best[k]=min(valid,key=lambda p:(score(p),p))
            if mode=='full':break
        if k==MAX_REQUESTS:continue
        nxt=set()
        for parts in candidates:
            for j,indices in enumerate(parts):
                # With one split left, every other group must already fit.
                if k==MAX_REQUESTS-1 and any(len(group(g)['objects'])>HARD_LIMIT for i,g in enumerate(parts) if i!=j):continue
                for pair in splits(indices):nxt.add(tuple(sorted(parts[:j]+parts[j+1:]+pair)))
        levels[k+1]=nxt
    fallback=False
    if not best:
        # Capacity fallback for a feasible arrangement not representable by axis cuts.
        order=sorted(range(n),key=lambda i:(-len(units[i]['objects']),units[i]['ids']))
        visited=set();budget=30000;answer=None
        def search(i,bins):
            nonlocal budget,answer
            key=(i,tuple(sorted(bins)))
            if key in visited or budget<=0:return False
            visited.add(key);budget-=1
            if i==n:answer=tuple(sorted(b for b in bins if b));return True
            unit=order[i];seen=set();choices=[]
            for j,b in enumerate(bins):
                if b in seen:continue
                seen.add(b);merged=tuple(sorted(b+(unit,)))
                if len(group(merged)['objects'])<=HARD_LIMIT:
                    choices.append((area(group(merged)['bbox']),j,merged))
            for _,j,merged in sorted(choices):
                nxt=list(bins);nxt[j]=merged
                if search(i+1,tuple(nxt)):return True
            return False
        if len({o['id'] for u in units for o in u['objects']})<=MAX_REQUESTS*HARD_LIMIT:
            search(0,((),(),()))
        if answer:best[len(answer)]=answer;fallback=True
        else:return [],[],('No feasible grouping found within 3 requests and 20 boxes; associations preserved'+(' (search budget reached)' if budget<=0 else ''))
    chosen=min(best.values(),key=lambda p:(score(p),len(p),p))
    alternatives=[{'requests':k,'score':score(p),'counts':[len(group(g)['objects']) for g in p],
                   'selected':p==chosen} for k,p in sorted(best.items())]
    groups=[group(g) for g in chosen]
    for g in groups:
        g['notes']=['capacity_fallback'] if fallback else [
            'single_full_image' if mode=='full' and len(groups)==1 else
            'capacity_partition' if mode=='full' else 'regional_partition']
    return sorted(groups,key=lambda g:(g['bbox'][1],g['bbox'][0],g['ids'])),alternatives,None


def partition(units, size, mode, max_requests=None):
    """Keep existing small-batch grouping; extend full-image capacity without splitting units."""
    limit = max_requests if max_requests is not None else (CHAIN_MAX_REQUESTS if mode == 'full' else MAX_REQUESTS)
    groups, alternatives, problem = _partition_original(units, size, mode)
    if problem and mode == 'full' and all(len(u['objects']) <= HARD_LIMIT for u in units):
        bins = []
        for unit in sorted(units, key=lambda u: (-len(u['objects']), u['ids'])):
            incoming = {o['id']: o for o in unit['objects']}
            choices = [(len(set(b['objects']) | incoming.keys()), i) for i, b in enumerate(bins)
                       if len(set(b['objects']) | incoming.keys()) <= HARD_LIMIT]
            if choices:
                _, index = min(choices)
                bucket = bins[index]
            else:
                bucket = {'ids': [], 'objects': {}}
                bins.append(bucket)
            bucket['ids'].extend(unit['ids'])
            for oid, obj in incoming.items():
                if bucket['objects'].get(oid, {}).get('relation') != 'explicit':
                    bucket['objects'][oid] = obj
        groups = [{'ids': sorted(b['ids']), 'objects': sorted(b['objects'].values(), key=lambda o: o['id']),
                   'bbox': union([o['bbox'] for o in b['objects'].values()]), 'notes': ['capacity_partition']}
                  for b in bins]
        groups.sort(key=lambda g: (g['bbox'][1], g['bbox'][0], g['ids']))
        alternatives = [{'requests': len(groups), 'counts': [len(g['objects']) for g in groups], 'selected': True}]
        problem = None
    if len(groups) > limit:
        return [], [], f'Material plan needs {len(groups)} requests; configured limit is {limit}'
    return groups, alternatives, problem


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])


def overlap(a,b):return area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])


def gap(a,b,size):
    return math.hypot(max(0,a[0]-b[2],b[0]-a[2])/size[0],
                      max(0,a[1]-b[3],b[1]-a[3])/size[1])


def plan(scene, targets, padding=.1, mode='full', padding_mode='capped', max_requests=None):
    from reveal import request_object
    if mode not in ['grouped','full']:raise ValueError('Unknown grouped request mode')
    if padding_mode not in ['capped','ratio']:raise ValueError('Unknown Reveal padding mode')
    version=FULL_VERSION if mode=='full' else VERSION
    size=scene['reference_size'];w,h=size;full=[0,0,w,h]
    by_id={o['id']:o for o in scene['objects']}
    selected={t['id']:t for t in targets if t.get('request',True)}
    photos={o['id']:o for o in scene['objects'] if o['kind']=='photo'}
    parent={oid:oid for oid in selected}
    def root(oid):
        while parent[oid]!=oid:oid=parent[oid]
        return oid
    for oid,t in selected.items():
        if t.get('kind')=='frame' and (t.get('method')!='extract' or not photo_ids(t) or any(pid not in photos for pid in photo_ids(t))):
            raise ValueError('Frame requires method=extract and a linked photo: '+oid)
        owner=t.get('parent_id') or t.get('embedded_in')
        if owner in selected:parent[root(oid)]=root(owner)
    components={}
    for oid in selected:components.setdefault(root(oid),[]).append(oid)
    units=[];blocked=[]
    for ids in components.values():
        ids=sorted(ids);members=[];contexts={};notes=[]
        for oid in ids:
            t=selected[oid]
            o=request_object(t,size,padding,padding_mode);o['role']='asset';members.append(o)
            required=[]
            if t.get('photo_id'):
                if any(pid not in photos for pid in photo_ids(t)):raise ValueError('Missing linked photo: '+oid)
                required.extend(photo_ids(t))
            required += [pid for pid,p in photos.items() if p.get('parent_id')==oid]
            for pid in required:contexts[pid]='explicit'
            # Large backings/connected ornaments may carry photos in other layers.
            # Only add nearby small windows, never the full-page customer background.
            if area(t['bbox'])>w*h*.35:
                for pid,p in photos.items():
                    if area(p['bbox'])<w*h*.4 and overlap(o['bbox'],p['bbox'])/max(1,area(p['bbox']))>.65:
                        contexts.setdefault(pid,'spatial_context')
        for pid,reason in sorted(contexts.items()):
            box=list(photos[pid]['bbox'])
            cid='__photo_context_'+fingerprint(pid)[:16]
            if cid in by_id:raise ValueError('Reserved context ID collision')
            members.append({'id':cid,'source_id':pid,'role':'photo_context',
                            'bbox':box,'original_bbox':box,'padding':0,'relation':reason})
        if len(members)>HARD_LIMIT:
            blocked.append({'ids':ids,'count':len(members),'reason':'Indivisible association exceeds 20 boxes'});continue
        box=union([o['bbox'] for o in members])
        units.append({'ids':ids,'objects':members,'bbox':box,'notes':notes,
                      'large':area(box)>w*h*.5})
    groups,alternatives,problem=partition(units,size,mode,max_requests) if not blocked else ([],[],None)
    if problem:blocked.append({'ids':sorted(selected),'reason':problem})
    batches=[]
    for group in groups:
        crop,_,gain,reason=crop_geometry(group['bbox'],size,mode)
        local=[];invalid=[]
        for obj in group['objects']:
            box=obj['bbox'];minimum=math.ceil(8*max(crop[2]-crop[0],crop[3]-crop[1])/1024)
            if obj['role']=='photo_context' and min(box[2]-box[0],box[3]-box[1])<minimum:
                invalid.append(obj['source_id'])
            local.append({**obj,'reference_bbox':box,
                          'bbox':[box[0]-crop[0],box[1]-crop[1],box[2]-crop[0],box[3]-crop[1]]})
        if invalid:
            blocked.append({'ids':group['ids'],'reason':'Photo context below API minimum; not expanded','photos':invalid});continue
        batch={'primary_ids':group['ids'],'crop_box':crop,'reference_size':size,'objects':local,
               'count':len(local),'reason':reason,'notes':group['notes'],'estimated_sampling_gain':round(gain,3)}
        identity=[version,mode,padding,batch] if padding_mode=='ratio' else [version,mode,padding_mode,padding,batch]
        batch['id']='batch_'+fingerprint(identity)[:16];batches.append(batch)
    result={'version':version,'mode':mode,'reference_size':size,'padding':padding,
            'soft_limit':SOFT_LIMIT if mode=='grouped' else None,'hard_limit':HARD_LIMIT,
            'max_requests':max_requests if max_requests is not None else (CHAIN_MAX_REQUESTS if mode=='full' else MAX_REQUESTS),
            'status':'blocked' if blocked else 'ready','alternatives':alternatives,
            'batches':[] if blocked else batches,'blocked':blocked}
    if padding_mode!='ratio':result['padding_mode']=padding_mode
    return result


def preview(reference, plan_data, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    save(folder/'request-plan.json',plan_data)
    with Image.open(reference) as src:im=src.convert('RGB')
    draw=ImageDraw.Draw(im);colors=['#ed4141','#10a571','#356ce8','#a82bba','#ce7b00']
    for i,batch in enumerate(plan_data['batches']):
        color=colors[i%len(colors)];box=batch['crop_box'];draw.rectangle(box,outline=color,width=3)
        draw.text((box[0]+3,box[1]+3+i*12),f"G{i+1}: {batch['count']} boxes / {batch['estimated_sampling_gain']}x",fill=color)
    im.save(folder/'groups.png')
    return plan_data
