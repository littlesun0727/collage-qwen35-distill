"""Review existing boxes in annotated crops; merge only geometry into the analysis."""

from copy import deepcopy
import math
from PIL import Image, ImageDraw
from collage.contracts import obj
from collage.storage import file_sha, save


EDGES = ('left', 'top', 'right', 'bottom')


def rectangle_corners(bbox, rotation=0):
    l, t, r, b = bbox
    cx, cy = (l+r)/2, (t+b)/2
    angle = math.radians(rotation)
    c, s = math.cos(angle), math.sin(angle)
    return [[cx+(x-cx)*c-(y-cy)*s, cy+(x-cx)*s+(y-cy)*c]
            for x, y in [(l, t), (r, t), (r, b), (l, b)]]


def prepare(run, folder, current):
    """Keep native pixels and generous context, including rotated coarse bounds."""
    folder.mkdir(parents=True, exist_ok=True)
    crops = folder/'crops'
    crops.mkdir(exist_ok=True)
    marked_crops = folder/'marked-crops'
    marked_crops.mkdir(exist_ok=True)
    source_path = run/'prepared/reference.png'
    rows, targets = [], []
    images = []
    with Image.open(source_path) as source:
        width, height = source.size
        if [width, height] != current['analysis']['reference_size']:
            raise ValueError('Box refinement reference size differs from analysis')
        for item in current['analysis']['objects']:
            if item['kind'] == 'background':
                continue
            points = rectangle_corners(item['bbox'], item.get('rotation', 0))
            l, t = (min(p[i] for p in points) for i in (0, 1))
            r, b = (max(p[i] for p in points) for i in (0, 1))
            px, py = max(128, (r-l)*.75), max(128, (b-t)*.75)
            region = [max(0, math.floor(l-px)), max(0, math.floor(t-py)),
                      min(width, math.ceil(r+px)), min(height, math.ceil(b+py))]
            path = crops/f'{len(rows)+1:03d}.png'
            crop = source.crop(region)
            crop.save(path)
            size = [region[2]-region[0], region[3]-region[1]]
            local_points = [(x-region[0], y-region[1]) for x, y in points]
            marked = crop.convert('RGB')
            ImageDraw.Draw(marked).line(local_points+[local_points[0]], fill='#ff00ff', width=2)
            model_path = marked_crops/path.name
            marked.save(model_path)
            target = {k: item[k] for k in ('id', 'kind', 'label', 'description', 'method', 'text', 'mode')
                      if k in item}
            # Numeric box coordinates stay internal; percentages use the
            # unmodified intrinsic dimensions of the visible candidate.
            target.update(image_size=size, original_rotation=item.get('rotation', 0))
            # Crop provenance stays internal: a source edge says nothing about
            # whether this target's visible contour is cut off.
            source_edges = [
                edge for edge, hit in zip(('left', 'top', 'right', 'bottom'),
                                         (region[0] == 0, region[1] == 0,
                                          region[2] == width, region[3] == height)) if hit]
            rows.append({'id': item['id'], 'file': str(path), 'model_file': str(model_path),
                         'crop_region': region,
                         'size': size, 'source_edges': source_edges, 'before_bbox': item['bbox'],
                         'before_rotation': item.get('rotation', 0)})
            targets.append(target)
            images.append((f"局部目标 {item['id']}：{item.get('description', item.get('label', ''))}"
                           '。洋红细线为待检查的原框，不属于图像内容。'
                           '左右扩展以原框宽度为百分比基准，上下扩展以原框高度为百分比基准。', model_path))
    manifest = {'version': 5, 'reference_size': [width, height],
                'reference_sha256': file_sha(source_path), 'crops': rows}
    save(folder/'manifest.json', manifest)
    save(folder/'before.json', current)
    return {'movement_unit': 'original_bbox_percent', 'targets': targets}, images, manifest


def schema(ids, version=5):
    if version >= 4:
        percentage = version >= 5
        movement = obj({
            'action': {'enum': ['expand', 'keep', 'uncertain'] if percentage else
                       ['expand', 'shrink', 'keep', 'uncertain']},
            'percent' if percentage else 'amount': {
                'type': 'number', 'minimum': 0, 'maximum': 100 if percentage else 1000},
        })
        return obj({
            'movement_unit': {'const': 'original_bbox_percent' if percentage else 'crop_normalized_1000'},
            'objects': {'type': 'array', 'minItems': len(ids), 'maxItems': len(ids), 'items': obj({
                'id': {'type': 'string', 'enum': ids},
                'status': {'enum': ['keep', 'adjust', 'clipped', 'uncertain']},
                'edges': obj({edge: deepcopy(movement) for edge in EDGES}),
                'reason': {'type': 'string', 'maxLength': 240},
            })},
        })
    # Retain the historical absolute-coordinate contract for saved v1-v3 runs.
    number = {'type': 'number', 'minimum': 0, 'maximum': 1000}
    point = {'type': 'array', 'items': number, 'minItems': 2, 'maxItems': 2}
    bbox = {'type': 'array', 'items': number, 'minItems': 4, 'maxItems': 4}
    corners = {'type': 'array', 'items': point, 'minItems': 4, 'maxItems': 4}
    return obj({
        'coordinate_space': {'const': 'crop_normalized_1000'},
        'objects': {'type': 'array', 'minItems': len(ids), 'maxItems': len(ids), 'items': obj({
            'id': {'type': 'string', 'enum': ids},
            'status': {'enum': ['keep', 'adjust', 'clipped', 'uncertain']},
            'edge_review': obj({edge: {'enum': ['aligned', 'too_tight', 'too_loose', 'uncertain']}
                                for edge in ('left', 'top', 'right', 'bottom')}),
            'bbox': {'anyOf': [bbox, {'type': 'null'}]},
            'corners': {'anyOf': [corners, {'type': 'null'}]},
            'reason': {'type': 'string', 'maxLength': 240},
        })},
    })


def validate(answer, manifest):
    expected = {row['id'] for row in manifest['crops']}
    ids = [item['id'] for item in answer['objects']]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Return every crop target ID exactly once')
    if manifest.get('version', 2) >= 4:
        _validate_moves(answer, percentage=manifest['version'] >= 5)
        return
    for item in answer['objects']:
        status = item['status']
        allowed = {'keep', 'adjust', 'clipped', 'uncertain'}
        if manifest.get('version', 2) < 3:
            allowed.add('located')  # Historical local answers remain readable.
        if status not in allowed:
            raise ValueError(item['id'] + ': invalid refinement status')
        if status in {'adjust', 'located'} and item['bbox'] is None:
            raise ValueError(item['id'] + ': adjust requires a local bbox')
        if status in {'keep', 'clipped', 'uncertain'}:
            if item['bbox'] is not None or item['corners'] is not None:
                raise ValueError(item['id'] + ': unmodified geometry must be null')
        if status == 'keep' and any(value != 'aligned' for value in item['edge_review'].values()):
            raise ValueError(item['id'] + ': keep requires all edges aligned')
        if item['bbox'] is not None:
            l, t, r, b = item['bbox']
            if not (0 <= l < r <= 1000 and 0 <= t < b <= 1000):
                raise ValueError(item['id'] + ': invalid crop-local bbox')


def _validate_moves(answer, percentage=False):
    unit = 'original_bbox_percent' if percentage else 'crop_normalized_1000'
    field, maximum = ('percent', 100) if percentage else ('amount', 1000)
    if answer.get('movement_unit') != unit:
        raise ValueError('Edge movements must use ' + unit)
    for item in answer['objects']:
        status = item['status']
        if status not in {'keep', 'adjust', 'clipped', 'uncertain'}:
            raise ValueError(item['id'] + ': invalid refinement status')
        if 'bbox' in item or 'corners' in item:
            raise ValueError(item['id'] + ': return edge movements, not final coordinates')
        edges = item.get('edges', {})
        if set(edges) != set(EDGES):
            raise ValueError(item['id'] + ': return all four edge movements')
        moving = False
        for edge, move in edges.items():
            action, amount = move.get('action'), move.get(field)
            if percentage and (set(move) != {'action', 'percent'} or action == 'shrink'):
                raise ValueError(item['id'] + ': percentage refinement only expands or keeps edges')
            if (isinstance(amount, bool) or not isinstance(amount, (int, float)) or
                    not math.isfinite(amount) or not 0 <= amount <= maximum):
                raise ValueError(item['id'] + f': edge {field} must be finite and within 0..{maximum}')
            if action in {'expand', 'shrink'}:
                if amount <= 0:
                    raise ValueError(item['id'] + ': moving edges require positive amounts')
                moving = True
            elif action in {'keep', 'uncertain'}:
                if amount != 0:
                    raise ValueError(item['id'] + ': unchanged edges require zero amounts')
            else:
                raise ValueError(item['id'] + ': invalid edge action')
        if (status == 'adjust') != moving:
            raise ValueError(item['id'] + ': only adjust may contain a positive edge movement')
        if status == 'keep' and any(m['action'] != 'keep' for m in edges.values()):
            raise ValueError(item['id'] + ': keep requires all edges keep')


def _move_geometry(item, row, original, reference_size, percentage=False):
    """Restore independent edge movements and retain the original angle.

    Side labels follow the original candidate's local axes. V5 percentages
    use the unmodified intrinsic candidate width/height; V4
    amounts keep their historical crop-normalized interpretation. Moving one
    side never changes the basis for another side.
    """
    l, t, r, b = original['bbox']
    angle = original.get('rotation', 0)
    cx, cy, width, height = (l+r)/2, (t+b)/2, r-l, b-t
    basis = [width, height] if percentage else row['size']
    field, divisor = ('percent', 100) if percentage else ('amount', 1000)
    outward = {edge: move[field] * basis[0 if edge in {'left', 'right'} else 1]/divisor *
                    (1 if move['action'] == 'expand' else -1 if move['action'] == 'shrink' else 0)
               for edge, move in item['edges'].items()}
    left, right = -width/2-outward['left'], width/2+outward['right']
    top, bottom = -height/2-outward['top'], height/2+outward['bottom']
    if left >= right or top >= bottom:
        raise ValueError('边缘移动后框宽高无效，保留原框')
    dx, dy = (left+right)/2, (top+bottom)/2
    theta = math.radians(angle)
    cx += dx*math.cos(theta)-dy*math.sin(theta)
    cy += dx*math.sin(theta)+dy*math.cos(theta)
    width, height = right-left, bottom-top
    box = [round(cx-width/2), round(cy-height/2), round(cx+width/2), round(cy+height/2)]
    # Axis-aligned visible bounds cannot extend beyond the reference itself.
    # Internal crop edges are never treated as a target boundary.
    if not angle:
        box = [max(0, box[0]), max(0, box[1]),
               min(reference_size[0], box[2]), min(reference_size[1], box[3])]
    l, t, r, b = box
    if not (0 <= l < r <= reference_size[0] and 0 <= t < b <= reference_size[1]):
        raise ValueError('边缘移动后制作框超出原图范围，保留原框')
    points = rectangle_corners(box, angle)
    bounds = [min(p[0] for p in points), min(p[1] for p in points),
              max(p[0] for p in points), max(p[1] for p in points)]
    source = [0, 0, *reference_size]
    crop = row['crop_region']
    for i, value in enumerate(bounds):
        beyond = value < source[i]-.001 if i < 2 else value > source[i]+.001
        if beyond:
            raise ValueError('旋转框移动后轮廓超出原图范围，保留原框')
        if crop[i] != source[i]:
            reaches = value <= crop[i]+2 if i < 2 else value >= crop[i]-2
            if reaches:
                raise ValueError('目标贴到局部裁图边缘，无法确认完整边界')
    return box, angle


def _oriented_rectangle(points):
    """Fit the declared TL, TR, BR, BL corners in pixel space, never 0..1000 space."""
    a, b, c, d = points
    ux, uy = b[0]-a[0]+c[0]-d[0], b[1]-a[1]+c[1]-d[1]
    norm = math.hypot(ux, uy)
    if norm < 2:
        raise ValueError('矩形四角无法确定方向')
    u = ux/norm, uy/norm
    v = -u[1], u[0]
    along = [p[0]*u[0]+p[1]*u[1] for p in points]
    across = [p[0]*v[0]+p[1]*v[1] for p in points]
    left, right, top, bottom = min(along), max(along), min(across), max(across)
    w, h = right-left, bottom-top
    if min(w, h) < 2:
        raise ValueError('矩形四角退化')
    # Perspective/trapezoids cannot be faithfully represented by bbox + rotation.
    expected = [(left, top), (right, top), (right, bottom), (left, bottom)]
    error = max(math.hypot(x-ex, y-ey) for x, y, (ex, ey) in zip(along, across, expected))
    if error > max(3, min(w, h)*.15):
        raise ValueError('四角不是稳定的旋转矩形，保留原框')
    cu, cv = (left+right)/2, (top+bottom)/2
    cx, cy = cu*u[0]+cv*v[0], cu*u[1]+cv*v[1]
    return [round(cx-w/2), round(cy-h/2), round(cx+w/2), round(cy+h/2)], round(math.degrees(math.atan2(u[1], u[0])), 3)


def _geometry(item, row, original, reference_size):
    x, y, _, _ = row['crop_region']
    w, h = row['size']
    l, t, r, b = item['bbox']
    visible = [x+l*w/1000, y+t*h/1000, x+r*w/1000, y+b*h/1000]
    # A crop edge inside the source is not an object edge. Do not accept a cut-off target.
    for index, boundary in enumerate((0, 0, reference_size[0], reference_size[1])):
        near = abs(visible[index]-row['crop_region'][index]) <= 2
        if near and row['crop_region'][index] != boundary:
            raise ValueError('目标贴到局部裁图边缘，无法确认完整边界')
    if item['corners'] is not None:
        points = [[x+p[0]*w/1000, y+p[1]*h/1000] for p in item['corners']]
        extrema = [min(p[0] for p in points), min(p[1] for p in points),
                   max(p[0] for p in points), max(p[1] for p in points)]
        if any(abs(a-b) > max(4, (visible[2+i%2]-visible[i%2])*.12)
               for i, (a, b) in enumerate(zip(extrema, visible))):
            raise ValueError('四角与可见外接框不一致')
        box, angle = _oriented_rectangle(points)
    else:
        if original.get('method', 'local') != 'extract' and original.get('rotation', 0):
            raise ValueError('已有旋转的本地对象缺少四角，保留原框及方向')
        box, angle = [round(v) for v in visible], 0
    l, t, r, b = box
    if not (0 <= l < r <= reference_size[0] and 0 <= t < b <= reference_size[1]):
        raise ValueError('旋转前制作框超出原图范围，保留原框')
    return box, angle


def _sync_text_units(answer, current, changed):
    before = {o['id']: o for o in current['analysis']['objects']}
    after = {o['id']: o for o in answer['analysis']['objects']}
    for unit in answer['text_units']:
        owner = unit.get('carrier_id') or unit.get('text_id')
        if unit['treatment'] == 'editable_label':
            members = (unit.get('carrier_id'), unit.get('text_id'))
            if any(k in changed for k in members):
                records = [after[k]['bbox'] for k in members if k in after]
                unit['bbox'] = [min(b[0] for b in records), min(b[1] for b in records),
                                max(b[2] for b in records), max(b[3] for b in records)]
            continue
        if owner not in changed:
            continue
        old, new = before[owner]['bbox'], after[owner]['bbox']
        unit['bbox'] = [round(new[i%2]+(value-old[i%2]) *
                              (new[2+i%2]-new[i%2])/(old[2+i%2]-old[i%2]))
                        for i, value in enumerate(unit['bbox'])]


def merge(answer, current, manifest):
    """Keep unlocated/unsafe objects unchanged, and retain all non-geometric metadata."""
    validate(answer, manifest)
    result = {'decision': 'replace', 'summary': '', **deepcopy(current)}
    rows = {row['id']: row for row in manifest['crops']}
    objects = {item['id']: item for item in result['analysis']['objects']}
    changed, confirmed, unresolved, mapping = [], [], [], []
    for item in answer['objects']:
        oid = item['id']
        original = objects[oid]
        before = {'bbox': original['bbox'], 'rotation': original.get('rotation', 0)}
        review = {k: item[k] for k in ('status', 'edge_review', 'edges', 'reason') if k in item}
        if manifest.get('version', 2) >= 4:
            actions = {'expand': 'too_tight', 'shrink': 'too_loose',
                       'keep': 'aligned', 'uncertain': 'uncertain'}
            review['edge_review'] = {edge: actions[move['action']] for edge, move in item['edges'].items()}
        if item['status'] == 'keep':
            confirmed.append(oid)
            mapping.append({'id': oid, 'before': before, 'after': deepcopy(before), **review})
            continue
        adjusting = item['status'] in {'adjust', 'located'}
        reason = '' if adjusting else item['reason']
        if adjusting:
            try:
                if manifest.get('version', 2) >= 4:
                    bbox, rotation = _move_geometry(item, rows[oid], original, manifest['reference_size'],
                                                    percentage=manifest['version'] >= 5)
                else:
                    bbox, rotation = _geometry(item, rows[oid], original, manifest['reference_size'])
            except ValueError as error:
                reason = str(error)
            else:
                after = {'bbox': bbox, 'rotation': rotation}
                mapping.append({'id': oid, 'before': before, 'after': after, **review})
                if before != after:
                    original['bbox'] = bbox
                    if rotation or 'rotation' in original:
                        original['rotation'] = rotation
                    changed.append(oid)
        if not adjusting or reason:
            unresolved.append({'object': oid, 'code': 'boxes_refinement_unresolved',
                               'message': f"{oid}：局部精修未确认，保留原框；{reason or item['status']}"})
    _sync_text_units(result, current, changed)
    result['summary'] = (f'局部精修：更新 {len(changed)} 个对象，确认保留 {len(confirmed)} 个，'
                         f'{len(unresolved)} 个未确认并保留原框。')
    if changed:
        result['summary'] += ' 更新：' + '、'.join(changed)
    else:
        result = {'decision': 'keep', 'summary': result['summary']}
    return result, unresolved, mapping
