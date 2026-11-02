"""Normalize unsupported optional metadata without dropping objects."""
from collage.object_roles import is_customer_photo


def normalize_boxes(value, current):
    notes=normalize(value)
    old={o['id']:o for o in current['analysis']['objects']}
    for obj in value['analysis']['objects']:
        before=old.get(obj['id'],{})
        aperture=before.get('window_bbox')
        if (obj.get('kind')!='photo' or not aperture or obj.get('window_bbox')!=aperture
                or obj['bbox']==before['bbox']):continue
        l,t,r,b=before['bbox'];nl,nt,nr,nb=obj['bbox']
        obj['window_bbox']=[round(nl+(v-l)*(nr-nl)/(r-l)) if i%2==0
                            else round(nt+(v-t)*(nb-nt)/(b-t)) for i,v in enumerate(aperture)]
        notes.append({'object':obj['id'],'code':'aperture_follows_geometry',
                      'message':obj['id']+'：照片框已修正，同步仍沿用旧坐标的照片内口。'})
    return notes


def normalize(value):
    notes = []
    objects = {item['id']: item for item in value['analysis']['objects']}
    memberships = []
    for item in value['analysis']['objects']:
        owner = objects.get(item.get('embedded_in'))
        if owner and owner['id'] != item['id'] and (
                (is_customer_photo(item) or item['kind']=='frame') or owner['kind'] == 'background'
                or owner.get('embedded_in')):
            memberships.append(item)
    for item in memberships:
        owner = item.pop('embedded_in')
        item.setdefault('parent_id', owner)
        notes.append({'object': item['id'], 'code': 'layout_membership',
                      'before': {'embedded_in': owner}, 'parent_id': item['parent_id'],
                      'message': f"{item['id']}：将布局归属整理为父级关联，保留独立照片或装饰图层。"})
    for item in value['analysis']['objects']:
        oid = item['id']
        style = item.get('style', {})
        if not is_customer_photo(item):
            removed = {key: style.pop(key) for key in
                       ('card', 'corner_radius', 'outline_width', 'outline_color') if key in style}
            for key in ('appearance', 'mode'):
                if key in item:
                    removed[key] = item.pop(key)
            if removed:
                notes.append({'object': oid, 'code': 'inapplicable_photo_options',
                              'removed': removed,
                              'message': f'{oid}：忽略非照片对象不支持的照片参数，保留对象及原有布局。'})
        if 'edge' in style and style.get('shape') not in {'paper', 'tape'}:
            edge = style.pop('edge')
            notes.append({'object': oid, 'code': 'inapplicable_paper_edge',
                          'removed': {'edge': edge},
                          'message': f'{oid}：当前形状不支持纸边参数，按已声明形状继续绘制。'})
        if 'text_status' in item or 'text' in item or item['kind'] == 'text':
            text = item.get('text')
            known = isinstance(text, str) and bool(text.strip())
            if not known and item['kind'] != 'text':
                before = {key: item.pop(key) for key in ('text', 'text_status') if key in item}
                notes.append({'object': oid, 'code': 'empty_artwork_text_metadata',
                              'removed': before,
                              'message': f'{oid}：移除空文字识别元数据，完整保留装饰素材。'})
            elif item.get('text_status') != ('known' if known else 'unreadable') or (not known and text is not None):
                before = {'text': text, 'text_status': item.get('text_status')}
                item['text_status'] = 'known' if known else 'unreadable'
                if not known:
                    item['text'] = None
                notes.append({'object': oid, 'code': 'consistent_text_metadata', 'before': before,
                              'message': f'{oid}：按实际文字内容整理识别状态，未补写文字。'})
    return notes
