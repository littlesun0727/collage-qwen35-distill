"""Matched detail crops for visual review, not automatic occlusion verdicts."""
from collage.object_roles import is_customer_photo, is_photo_background
from collage.frame_links import photo_ids
import math
from itertools import combinations
from pathlib import Path
from PIL import Image, ImageDraw


def envelope(obj):
    from editor_scene import bounds
    return list(bounds(obj))


def intersection(a,b):
    return [max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]


def area(box):return max(0,box[2]-box[0])*max(0,box[3]-box[1])


def regions(scene,limit=4):
    w,h=scene['reference_size'];canvas=[0,0,w,h]
    objects=[o for o in scene['objects'] if (is_customer_photo(o) or o['kind'] in ['overlay','frame'])
             and not o.get('embedded_owner') and not o.get('recovery_owner')]
    candidates=[]
    for a,b in combinations(objects,2):
        if not is_customer_photo(a) and not is_customer_photo(b):continue
        # A full-page background photo would otherwise dominate every detail.
        if any(is_customer_photo(o) and o.get('mode','cover')!='cutout'
               and area(o['bbox'])>.85*w*h for o in [a,b]):continue
        overlap=intersection(intersection(envelope(a),envelope(b)),canvas)
        if area(overlap)<16:continue
        linked=b['id'] in photo_ids(a) or a['id'] in photo_ids(b) or a.get('parent_id')==b['id'] or b.get('parent_id')==a['id']
        priority=5 if linked and any(o['kind']=='frame' for o in [a,b]) else 1 if linked else 3 if a['kind']==b['kind'] else 4
        # Photos wholly inside a large paper page are less useful than crossing edges.
        if not linked and any(o['kind']=='overlay' and area(envelope(o))>area(envelope(other))
                              and area(overlap)>.96*area(envelope(other)) for o,other in [(a,b),(b,a)]):priority=2
        if any(o.get('mode')=='cutout' for o in [a,b]):priority+=1
        # Include context around the intersection, at a readable local scale.
        cx=(overlap[0]+overlap[2])/2;cy=(overlap[1]+overlap[3])/2
        cw=min(w,max(w*.2,(overlap[2]-overlap[0])*1.3))
        ch=min(h,max(h*.16,(overlap[3]-overlap[1])*1.3))
        l=max(0,min(w-cw,cx-cw/2));t=max(0,min(h-ch,cy-ch/2))
        box=[math.floor(l),math.floor(t),math.ceil(l+cw),math.ceil(t+ch)]
        candidates.append((priority,area(overlap),{'ids':[a['id'],b['id']],'bbox':box}))
    selected=[]
    for _,__,region in sorted(candidates,key=lambda c:(-c[0],-c[1],c[2]['ids'])):
        if any(area(intersection(region['bbox'],old['bbox']))>.6*min(area(region['bbox']),area(old['bbox'])) for old in selected):continue
        selected.append(region)
        if len(selected)==limit:break
    return selected


def publish(run,scene,reference,final):
    folder=Path(run)/'previews';folder.mkdir(exist_ok=True)
    selected=regions(scene);sx=final.width/reference.width;sy=final.height/reference.height
    for i,region in enumerate(selected,1):
        box=region['bbox'];left=reference.crop(box)
        right=final.crop([round(box[0]*sx),round(box[1]*sy),round(box[2]*sx),round(box[3]*sy)])
        zoom=min(560/left.width,680/left.height);size=(max(1,round(left.width*zoom)),max(1,round(left.height*zoom)))
        sheet=Image.new('RGB',(size[0]*2,size[1]+48),'#eeeeee');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),'REFERENCE',fill='black');draw.text((size[0]+8,5),'CURRENT',fill='black')
        draw.text((8,25),' / '.join(region['ids']),fill='black')
        sheet.paste(left.resize(size,Image.Resampling.LANCZOS),(0,48))
        sheet.paste(right.resize(size,Image.Resampling.LANCZOS),(size[0],48))
        path=folder/f'review-detail-{i}.png';sheet.save(path)
        region['file']=str(path.resolve())
    return selected
