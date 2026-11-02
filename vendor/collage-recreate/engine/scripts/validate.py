"""JSON Schema and cross-object validation; errors identify fields, not prose repairs."""
from collage.object_roles import is_customer_photo, is_photo_background
from collage.frame_links import photo_ids, frame_issues
from common import ROOT, read, sha
from jsonschema import Draft202012Validator
from referencing import Registry, Resource


def schema_check(name, value):
    schemas = [read(p) for p in (ROOT / 'schemas').glob('*.json')]
    registry = Registry().with_resources((s['$id'], Resource.from_contents(s)) for s in schemas)
    schema = next(s for s in schemas if s['$id'] == 'urn:collage:v5:' + name)
    errors = sorted(Draft202012Validator(schema, registry=registry).iter_errors(value), key=lambda e: str(list(e.path)))
    if errors:
        raise ValueError('; '.join(f'{".".join(map(str,e.path)) or "$"}: {e.message}' for e in errors[:15]))


def window_check(obj, style):
    window=style.get('window')
    if window is None:return
    if obj['kind']!='overlay' or obj.get('method','local')!='local' or obj.get('recovered') or obj.get('generated'):
        raise ValueError('window requires a local overlay')
    left,top,right,bottom=window['bbox']
    if not (0<=left<right<=1 and 0<=top<bottom<=1):
        raise ValueError('window.bbox must be a positive normalized rectangle')


def analysis_check(value, inputs=None):
    schema_check('analysis', value)
    if inputs and value['reference_size'] != inputs['reference_size']:
        raise ValueError('reference_size must equal prepared image dimensions')
    w, h = value['reference_size']
    objects = value['objects']
    for obj in objects:
        if obj.get('background_type') == 'photo':
            if (obj['kind'] != 'background' or obj.get('method', 'local') != 'local'
                    or obj.get('mode', 'cover') != 'cover' or obj.get('linked_to')
                    or obj.get('parent_id') or obj.get('rotation', 0)
                    or obj['bbox'] != [0, 0, w, h]
                    or value['layer_order'][:1] != [obj['id']]
                    or sum(o['kind'] == 'background' for o in objects) != 1):
                raise ValueError('Customer photo background must be the unique local full-canvas background')
    ids = [o['id'] for o in objects]
    if len(ids) != len(set(ids)):
        raise ValueError('Object IDs must be unique')
    if len(value['layer_order']) != len(ids) or set(value['layer_order']) != set(ids):
        order = value['layer_order']
        missing = sorted(set(ids)-set(order))
        unknown = sorted(set(order)-set(ids))
        repeated = sorted({oid for oid in order if order.count(oid)>1})
        raise ValueError('layer_order must contain each object ID exactly once; '
                         f'missing={missing}; unknown={unknown}; repeated={repeated}')
    for o in objects:
        l,t,r,b = o['bbox']
        if not (0 <= l < r <= w and 0 <= t < b <= h):
            raise ValueError(f'{o["id"]}.bbox: expected positive in-bounds original-pixel box')
        if o.get('parent_id') and (o['parent_id'] not in ids or o['parent_id'] == o['id']):
            raise ValueError(f'{o["id"]}.parent_id: invalid reference')
        style=o.get('style',{})
        window_check(o,style)
        if o.get('appearance') and not is_customer_photo(o):raise ValueError('appearance is for photos only')
        if any(k in style for k in ['card','corner_radius','outline_width','outline_color']) and not is_customer_photo(o):
            raise ValueError(f'{o["id"]}: photo effects require kind=photo')
        if is_customer_photo(o):
            if 'shape' in style:raise ValueError('Photo shape is not supported; use appearance or corner_radius')
            if o.get('appearance')=='polaroid' or 'card' in style:
                if o.get('mode','cover')!='cover':raise ValueError('Photo card requires cover mode')
                from effects import photo_layout,resolve_style
                photo_layout((r-l,b-t),resolve_style(o))
        if style.get('shape')=='curve' and len(style.get('points',[]))<2:raise ValueError('curve requires at least two anchors')
        if 'edge' in style and style.get('shape') not in ['paper','tape']:raise ValueError('edge applies to paper/tape only')
        if o['kind'] == 'text' and not {'text','text_status'} <= set(o):
            raise ValueError(f'{o["id"]}: text requires text and text_status')
        if 'text_status' in o:
            if (o['text_status'] == 'known') != (isinstance(o.get('text'),str) and bool(o['text'].strip())):
                raise ValueError(f'{o["id"]}: known text must be nonempty; unreadable text must be null')
        if not is_customer_photo(o) and 'mode' in o:
            raise ValueError(f'{o["id"]}: mode is only for photos')
        if o.get('style',{}).get('shape')=='polygon' and len(o['style'].get('points',[]))<3:
            raise ValueError(f'{o["id"]}: polygon requires at least three points')
    parents = {o['id']: o.get('parent_id') for o in objects}
    by_id={o['id']:o for o in objects}
    from photo_links import inventory_issues
    link_issues = inventory_issues(objects, value['layer_order'])
    if link_issues:
        raise ValueError('; '.join(issue['id'] + ': ' + issue['message'] for issue in link_issues))
    relation_issues = frame_issues(objects)
    if relation_issues:
        raise ValueError('; '.join(i['id'] + ': ' + i['message'] for i in relation_issues))
    for o in objects:
        if 'photo_id' in o:
            if o['kind'] not in ['overlay','frame'] or any(by_id.get(pid,{}).get('kind')!='photo' for pid in photo_ids(o)):
                raise ValueError('photo_id must link an overlay/frame to a photo')
        if o['kind']=='frame':
            if o.get('embedded_in'):
                raise ValueError(o['id']+': frame must be an independent extraction')
        if 'window_bbox' in o:
            box=o['window_bbox'];outer=o['bbox']
            if not is_customer_photo(o) or not (outer[0]<=box[0]<box[2]<=outer[2] and outer[1]<=box[1]<box[3]<=outer[3]):
                raise ValueError('window_bbox must be a positive rectangle inside its photo bbox')
        if 'embedded_in' in o:
            owner=by_id.get(o['embedded_in'],{})
            if is_customer_photo(o) or owner.get('kind') not in ['overlay','frame','photo'] or owner.get('id')==o['id'] or owner.get('embedded_in'):
                raise ValueError('embedded_in requires a distinct unembedded decoration/card owner')
    for key in ids:
        seen = set()
        while key:
            if key in seen: raise ValueError('parent_id cycle')
            seen.add(key); key = parents.get(key)
    return value


def bindings_check(value, analysis, catalog, verify=True):
    schema_check('bindings', value)
    slots = {o['id'] for o in analysis['objects'] if is_customer_photo(o)}
    selected = [b['slot_id'] for b in value['photos']]
    if len(selected) != len(set(selected)) or set(selected) != slots:
        raise ValueError('bindings.photos must cover every photo slot exactly once')
    assets = {a['id']:a for a in catalog['assets']}
    for b in value['photos']:
        if b['asset_id'] not in assets: raise ValueError(f'Unknown asset: {b["asset_id"]}')
        if verify and sha(assets[b['asset_id']]['file']) != assets[b['asset_id']]['sha256']:
            raise ValueError('Customer source changed; prepare a new task')
        crop = b.get('source_crop',[0,0,1,1])
        if not (crop[0] < crop[2] and crop[1] < crop[3]): raise ValueError('source_crop must have positive area')
    by_id = {o['id']:o for o in analysis['objects']}
    seen = set()
    for t in value['texts']:
        if t['object_id'] in seen: raise ValueError('Duplicate text binding')
        seen.add(t['object_id'])
        o = by_id.get(t['object_id'])
        if not o or o.get('text_status') != 'unreadable':
            raise ValueError('Text bindings only replace explicitly unreadable text')
    return value


def review_check(value, scene):
    schema_check('review', value)
    ids = {o['id'] for o in scene['objects']}
    removed={i['id'] for i in value['items'] if i['action']=='remove'}
    if 'layer_order' in value:
        order=value['layer_order']
        if len(order)!=len(ids-removed) or set(order)!=ids-removed:
            raise ValueError('layer_order must contain each remaining scene object ID exactly once')
        if value['verdict']=='pass':
            raise ValueError('Apply layer_order before writing a new pass review')
    seen = set()
    for item in value['items']:
        if item['id'] not in ids or item['id'] in seen: raise ValueError('Review IDs must be unique known objects')
        seen.add(item['id'])
        window_check(next(o for o in scene['objects'] if o['id']==item['id']),item.get('changes',{}).get('style',{}))
        if item['action']=='adjust' and not item.get('changes'): raise ValueError('adjust requires changes')
        if item['action']!='adjust' and item.get('changes'): raise ValueError('changes only apply to adjust')
        crop=item.get('changes',{}).get('source_crop')
        if crop and not (crop[0]<crop[2] and crop[1]<crop[3]):raise ValueError('source_crop must have positive area')
        if item['action']=='generate' and is_customer_photo(next(o for o in scene['objects'] if o['id']==item['id'])):
            raise ValueError('Customer photos cannot be generated')
        if item['action']=='generate' and next(o for o in scene['objects'] if o['id']==item['id'])['kind']=='frame':
            raise ValueError('Frames require extraction with their linked photo, not generation')
    if value['verdict']=='pass' and ((seen != ids and not value.get('checked_entire_composition')) or any(i['action']!='keep' for i in value['items'])):
        raise ValueError('pass requires keep decisions for all objects')
    return value
