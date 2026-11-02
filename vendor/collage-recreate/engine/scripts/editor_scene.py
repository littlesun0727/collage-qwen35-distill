"""Opt-in manual placement; original analysis and asset coordinates stay intact."""
from collage.object_roles import is_customer_photo, is_photo_background
from collage.frame_links import photo_ids
from copy import deepcopy
import math


def offset(obj):
    value=obj.get('editor_transform',{})
    return value.get('x',0),value.get('y',0)


def transform(obj):
    return {'x':0,'y':0,'rotation':0,'scale':1,**obj.get('editor_transform',{})}


def center(obj):
    l,t,r,b=obj['bbox'];return (l+r)/2,(t+b)/2


def matrix(obj):
    v=transform(obj);cx,cy=center(obj);angle=math.radians(v['rotation'])
    a=v['scale']*math.cos(angle);b=v['scale']*math.sin(angle)
    return a,b,-b,a,cx+v['x']-a*cx+b*cy,cy+v['y']-b*cx-a*cy


def bounds(obj):
    l,t,r,b=obj['bbox'];cx,cy=center(obj);v=transform(obj)
    angle=math.radians(obj.get('rotation',0)+v['rotation'])
    c=abs(math.cos(angle))*v['scale'];s=abs(math.sin(angle))*v['scale']
    w=(r-l)*c+(b-t)*s;h=(r-l)*s+(b-t)*c
    return cx+v['x']-w/2,cy+v['y']-h/2,cx+v['x']+w/2,cy+v['y']+h/2


def capabilities(scene):
    objects={o['id']:o for o in scene['objects']}
    owner={o['id']:o.get('embedded_owner') or o.get('recovery_owner') or o['id'] for o in objects.values()}
    for group in scene.get('generated_groups',[]):
        for identifier in group['member_ids']:owner[identifier]=group['primary']
    def root(identifier):
        seen=set()
        while owner.get(identifier,identifier)!=identifier and identifier not in seen:
            seen.add(identifier);identifier=owner[identifier]
        return identifier
    visible={root(i) for i in objects}
    links={i:{i} for i in visible}
    for o in objects.values():
        related=[o.get('parent_id'),*photo_ids(o),o.get('reveal_frame_id'),o.get('photo_window_spec',{}).get('linked_to')]
        # Only explicit photo/carrier links group layers. Ordinary overlapping
        # stickers or paper backgrounds never become an implicit giant group.
        if not is_customer_photo(o):related=photo_ids(o)
        for other in related:
            if other in objects:
                a,b=root(o['id']),root(other)
                links[a].add(b);links[b].add(a)
    result={}
    for i,o in objects.items():
        group={root(i)};pending=list(group)
        while pending:
            for other in links[pending.pop()]-group:group.add(other);pending.append(other)
        independent=root(i)==i
        movable=independent and o['kind'] in ['photo','overlay','frame','text'] and not any(objects[g]['kind']=='background' for g in group)
        fused=any(i in g['member_ids'] for g in scene.get('generated_groups',[]))
        editable=independent and not fused and o['kind']=='text' and o.get('method','local')!='extract' and not o.get('generated') and not o.get('recovered')
        remove_roots={root(i)}
        if is_customer_photo(o):
            remove_roots.update(root(frame['id']) for frame in objects.values()
                                if frame['kind']=='frame' and i in photo_ids(frame) and len(photo_ids(frame))==1)
            remove_roots.update(root(child['id']) for child in objects.values()
                                if child.get('photo_window_spec',{}).get('linked_to')==i)
        result[i]={'move':movable,'rotate':movable,'scale':movable,'remove':independent,
                   'remove_ids':sorted(k for k in objects if root(k) in remove_roots),
                   'crop':independent and not fused and is_customer_photo(o) and bool(o.get('source')) and not o.get('photo_window_spec',{}).get('linked_to'),
                   'text':editable,'group':sorted(group),'owner':root(i)}
    return result


def apply_edits(scene, changes, layer_order=None):
    if not isinstance(changes,list) or len(changes)>200 or (not changes and layer_order is None):raise ValueError('请提交对象修改或图层顺序')
    scene=deepcopy(scene);objects={o['id']:o for o in scene['objects']};caps=capabilities(scene)
    seen=set();moves={};removed=set();w,h=scene['reference_size']
    for change in changes:
        if not isinstance(change,dict) or set(change)-{'id','transform','text','remove','scope','source_crop','window_crop'}:raise ValueError('不支持的编辑字段')
        identifier=change.get('id')
        if not isinstance(identifier,str) or identifier not in objects or identifier in seen:raise ValueError('对象无效或重复')
        seen.add(identifier);o=objects[identifier];cap=caps[identifier]
        if len(change)==1:raise ValueError('修改内容为空')
        scope=change.get('scope','group')
        if scope not in ['object','group'] or ('scope' in change and 'transform' not in change):
            raise ValueError('变换范围必须为当前对象或关联组合，且需要变换参数')
        if 'remove' in change:
            if change['remove'] is not True or set(change)!={'id','remove'} or not cap['remove']:
                raise ValueError('请选择完整素材删除，不能同时修改被删除对象')
            removed.update(cap['remove_ids']);continue
        for crop_key in ['source_crop','window_crop']:
            if crop_key not in change:continue
            crop=change[crop_key]
            if not cap['crop'] or not isinstance(crop,list) or len(crop)!=4 or any(type(v) not in [int,float] or not math.isfinite(v) or not 0<=v<=1 for v in crop):
                raise ValueError('请选择独立照片，并提供有效的裁剪区域')
            if crop[2]-crop[0]<.01-1e-9 or crop[3]-crop[1]<.01-1e-9:
                raise ValueError('裁剪区域宽高至少为原图的1%')
            if crop_key=='source_crop':o['binding']['source_crop']=list(crop)
            elif crop==[0,0,1,1]:o.pop('window_crop',None)
            else:o['window_crop']=list(crop)
        if 'transform' in change:
            if o.get('shared_photo') and scope=='object':
                raise ValueError('连续出框需要整组变换；裁切请编辑底图照片')
            value=change['transform']
            if not cap['move'] or not isinstance(value,dict) or not {'x','y'}<=value.keys() or set(value)-{'x','y','rotation','scale'}:raise ValueError('该对象不能独立变换')
            if any(type(v) not in [int,float] or not math.isfinite(v) for v in value.values()):raise ValueError('变换参数无效')
            value={'rotation':0,'scale':1,**value}
            if max(abs(value['x']),abs(value['y']))>max(w,h)*2 or abs(value['rotation'])>36000 or not .05<=value['scale']<=10:raise ValueError('位移、角度或缩放超出范围（缩放5%至1000%）')
            old=transform(o);cx,cy=center(o);ratio=value['scale']/old['scale']
            angle=math.radians(value['rotation']-old['rotation']);c=math.cos(angle)*ratio;s=math.sin(angle)*ratio
            for member in ([identifier] if scope=='object' else cap['group']):
                other=objects[member];v=transform(other);mx,my=center(other)
                dx,dy=mx+v['x']-cx-old['x'],my+v['y']-cy-old['y']
                target={'x':round(cx+value['x']+c*dx-s*dy-mx,3),'y':round(cy+value['y']+s*dx+c*dy-my,3)}
                rotation=round(v['rotation']+value['rotation']-old['rotation'],6);size=round(v['scale']*ratio,6)
                if not .05<=size<=10:raise ValueError('组合素材缩放超出范围')
                if rotation or size!=1:target.update(rotation=rotation,scale=size)
                if member in moves and moves[member]!=target:raise ValueError('同一组合收到冲突变换')
                moves[member]=target
        if 'text' in change:
            value=change['text']
            if not cap['text'] or not isinstance(value,str) or not value.strip() or len(value)>2000:raise ValueError('只能修改独立文字，文字不能为空或超过2000字')
            o.update(text=value,text_origin='customer_revision',text_status='known',text_unresolved=False)
    for identifier,value in moves.items():
        if identifier in removed:continue
        o=objects[identifier];o['editor_transform']=value;l,t,r,b=bounds(o)
        if r<=0 or b<=0 or l>=w or t>=h:raise ValueError('对象不能完全移出画布')
    if removed:
        from scene import remove_objects
        for obj in objects.values():
            if obj['kind']=='frame' and photo_ids(obj) and set(photo_ids(obj)) <= removed:
                removed.update(caps[obj['id']]['remove_ids'])
        remove_objects(scene,removed)
    if layer_order is not None:
        ids={o['id'] for o in scene['objects']}
        if not isinstance(layer_order,list) or any(not isinstance(i,str) for i in layer_order) or len(layer_order)!=len(ids) or set(layer_order)!=ids:
            raise ValueError('图层顺序必须包含每个保留对象且不能重复')
        # Preserve photo/window and frame relationships inside each linked unit.
        for cap in caps.values():
            members=set(cap['group']) & ids
            if [i for i in scene['layer_order'] if i in members]!=[i for i in layer_order if i in members]:
                raise ValueError('照片与关联相框的内部顺序不能颠倒，请整组调整')
        scene['layer_order']=list(layer_order)
    scene['revision']=scene.get('revision',0)+1
    return scene


def clip_window(image,obj,scale):
    """Trim the displayed layer without resizing or recentering its pixels."""
    crop=obj.get('window_crop')
    if not crop:return image
    from PIL import Image,ImageDraw,ImageChops
    l,t,r,b=obj['bbox'];w,h=r-l,b-t
    box=[round((l+w*crop[0])*scale),round((t+h*crop[1])*scale),
         round((l+w*crop[2])*scale),round((t+h*crop[3])*scale)]
    mask=Image.new('L',image.size)
    ImageDraw.Draw(mask).rectangle((box[0],box[1],max(box[0],box[2]-1),max(box[1],box[3]-1)),fill=255)
    if obj.get('rotation'):
        mask=mask.rotate(-obj['rotation'],Image.Resampling.BICUBIC,center=((l+r)*scale/2,(t+b)*scale/2))
    if image.mode=='L':return ImageChops.multiply(image,mask)
    result=image.copy();result.putalpha(ImageChops.multiply(image.getchannel('A'),mask));return result


def translate(image,obj,scale):
    value=transform(obj)
    if value['rotation'] or value['scale']!=1:
        from PIL import Image
        a,b,c,d,e,f=matrix(obj);e*=scale;f*=scale;det=a*d-b*c
        return image.transform(image.size,Image.Transform.AFFINE,
            (d/det,-c/det,(c*f-d*e)/det,-b/det,a/det,(b*e-a*f)/det),Image.Resampling.BICUBIC)
    x,y=offset(obj)
    if not x and not y:return image
    from PIL import Image
    shifted=Image.new(image.mode,image.size)
    shifted.paste(image,(round(x*scale),round(y*scale)))
    return shifted
