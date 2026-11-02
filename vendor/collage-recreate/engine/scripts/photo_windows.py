"""Reference window plans and deterministic masks shared by still/Live rendering."""
from collage.object_roles import is_customer_photo, is_photo_background
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image, ImageColor, ImageDraw, ImageOps

from window_contract import SCHEMA, VERSION, empty_stroke, execution_plan, execution_spec, validate_drawing


EDGES = ('top', 'bottom', 'left', 'right')


def enforce_canvas_base(analysis, plan=None):
    from collage.object_roles import fill_background_canvas
    return fill_background_canvas(analysis)



def subject_stroke(spec):
    if not spec:
        return False
    spec = execution_spec(spec)
    return spec['subject_stroke']['enabled']


def needs_subject_mask(photo):
    """Mask inference is independent of whether the photo keeps its background."""
    spec = photo.get('photo_window_spec')
    if spec:
        spec = execution_spec(spec)
        return spec['subject']['mode'] == 'cutout' or subject_stroke(spec)
    return photo.get('mode') == 'cutout'


def lower_opaque_rectangles(analysis, photo_id, specs):
    """Guaranteed rectangular backing, excluding unknown silhouettes/soft edges.

    A bbox intersection alone says nothing about coverage at a fading edge.
    Nonrectangular or rotated backing needs a separate opaque background.
    """
    objects={o['id']:o for o in analysis['objects']}
    order=analysis.get('layer_order',list(objects))
    rectangles = []
    for oid in order[:order.index(photo_id)]:
        obj=objects[oid]
        style = obj.get('style', {})
        if (obj['kind'] not in {'background','photo'} or style.get('opacity',1) != 1
                or obj.get('rotation', 0) % 360):
            continue
        l,t,r,b = obj['bbox']; w,h = r-l,b-t
        if is_customer_photo(obj):
            spec = specs.get(oid)
            if (not spec or spec['subject']['mode'] == 'cutout'
                    or spec['shape']['type'] != 'rectangle'):
                continue
            fade = spec['fade'] if spec.get('fade_direction', 'inside') == 'inside' else dict.fromkeys(EDGES, 0)
            l,r = l+w*fade['left'],r-w*fade['right']
            t,b = t+h*fade['top'],b-h*fade['bottom']
        elif obj.get('background_type') == 'reference_clean' and obj.get('method') == 'extract':
            # The cleanup route admits an opaque residual canvas; failed cleanup is reported as missing.
            pass
        elif (obj.get('method','local') != 'local' or style.get('shape','rectangle') != 'rectangle'
              or style.get('window') or style.get('stroke')
              or ImageColor.getcolor(style.get('fill','#FFFFFF'),'RGBA')[3] != 255):
            continue
        rectangles.append((l,t,r,b))
    return rectangles


def _covered(box, rectangles):
    """Exact union coverage of an axis-aligned strip, including split backing."""
    l,t,r,b = box
    if r <= l or b <= t:
        return True
    xs = sorted({l,r} | {x for rect in rectangles for x in (rect[0],rect[2]) if l < x < r})
    for x0,x1 in zip(xs,xs[1:]):
        intervals = sorted((a[1],a[3]) for a in rectangles if a[0] <= x0 and a[2] >= x1)
        end = t
        for start,stop in intervals:
            if start > end + 1e-7:
                break
            end = max(end,stop)
        if end < b - 1e-7:
            return False
    return True


def fade_limits(analysis, photo_id, specs):
    """Conservative per-edge spans with opaque backing over the entire strip."""
    photo = next(o for o in analysis['objects'] if o['id'] == photo_id)
    l,t,r,b = photo['bbox']; w,h = r-l,b-t
    rects = lower_opaque_rectangles(analysis,photo_id,specs)
    angle = math.radians(photo.get('rotation',0)); c,s = math.cos(angle),math.sin(angle)
    cx,cy = (l+r)/2,(t+b)/2
    rw,rh = analysis['reference_size']
    def supported(edge, amount):
        spec = specs[photo_id]
        if specs[photo_id].get('fade_direction', 'inside') == 'outside':
            # Include corner extensions so a diagonal expansion cannot expose a gap.
            f = specs[photo_id]['fade']
            box = {'top':(l-w*f['left'],t-h*amount,r+w*f['right'],t),
                   'bottom':(l-w*f['left'],b,r+w*f['right'],b+h*amount),
                   'left':(l-w*amount,t-h*f['top'],l,b+h*f['bottom']),
                   'right':(r,t-h*f['top'],r+w*amount,b+h*f['bottom'])}[edge]
        else:
            box = {'top':(l,t,r,t+h*amount),'bottom':(l,b-h*amount,r,b),
                   'left':(l,t,l+w*amount,b),'right':(r-w*amount,t,r,b)}[edge]
        if spec['shape']['type'] != 'rectangle':
            # Curved boundary transitions can occupy any part of the nominal
            # box. Do not infer opaque backing from just its outer four strips.
            f = {**spec['fade'], edge: amount} if spec.get('fade_direction') == 'outside' else dict.fromkeys(EDGES, 0)
            box = (l-w*f['left'], t-h*f['top'], r+w*f['right'], b+h*f['bottom'])
        # A rotated strip's enclosing box is a conservative coverage requirement.
        points = [(cx+c*(x-cx)-s*(y-cy),cy+s*(x-cx)+c*(y-cy))
                  for x in (box[0],box[2]) for y in (box[1],box[3])]
        bounds = (max(0,min(x for x,y in points)),max(0,min(y for x,y in points)),
                  min(rw,max(x for x,y in points)),min(rh,max(y for x,y in points)))
        return _covered(bounds,rects)
    limits = {}
    for edge in EDGES:
        low,high = 0.,.5
        if supported(edge,high):
            limits[edge] = high
            continue
        for _ in range(24):
            mid = (low+high)/2
            if supported(edge,mid): low = mid
            else: high = mid
        limits[edge] = math.floor(low*1_000_000)/1_000_000
    return limits


def validate_plan(plan):
    if plan.get('schema_version') != VERSION:
        raise ValueError('Unsupported photo window contract; rerun photo_windows')
    Draft202012Validator(SCHEMA).validate(plan)
    for spec in plan['photos']:
        validate_drawing(spec)


def check(plan, analysis):
    validate_plan(plan)
    original = plan
    plan = execution_plan(plan)
    photos = {p['id']: p for p in analysis['objects'] if is_customer_photo(p)}
    ids = [p['photo_id'] for p in plan['photos']]
    if len(ids) != len(set(ids)) or set(ids) != set(photos):
        raise ValueError('photo_windows must cover every photo ID exactly once')
    specs={p['photo_id']:p for p in plan['photos']}
    for spec in plan['photos']:
        oid = spec['photo_id']; shape = spec['shape']; stroke = spec['stroke']
        if is_photo_background(photos[oid]) and (
                spec['subject']['mode'] != 'preserve' or spec['window_enabled']
                or any(spec['fade'].values()) or spec.get('linked_to')):
            raise ValueError(oid + ': photo background must preserve the full canvas without a mask, fade or linked_to')
        linked=spec.get('linked_to')
        initial_link = photos[oid].get('linked_to')
        if initial_link and linked != initial_link:
            raise ValueError(oid+': window plan must retain the explicit analysis linked_to')
        if linked:
            if linked not in specs or linked==oid or specs[linked].get('linked_to'):
                raise ValueError(oid+': linked_to must name an independent base photo')
            if spec['subject']['mode']!='cutout' or specs[linked]['subject']['mode']!='preserve':
                raise ValueError(oid+': linked pop-out needs cutout foreground and preserved base')
        allowed = {'type'} | {'rounded_rect': {'corner_radius'},
                             'star': {'star_points', 'inner_ratio'},
                             'polygon': {'points'}}.get(shape['type'], set())
        if set(shape) - allowed:
            raise ValueError(oid + ': parameters do not belong to this shape')
        if shape['type'] == 'polygon':
            points = shape['points']
            if len(set(map(tuple, points))) != len(points):
                raise ValueError(oid + ': polygon vertices must be distinct')
            area = abs(sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(points, points[1:]+points[:1])))/2
            if area < .01:
                raise ValueError(oid + ': polygon has negligible area')
            def cross(a, b, c):
                return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            for i, a in enumerate(points):
                b = points[(i+1) % len(points)]
                for j in range(i+1, len(points)):
                    if j == i+1 or (i == 0 and j == len(points)-1):
                        continue
                    c, d = points[j], points[(j+1) % len(points)]
                    if cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0:
                        if max(min(a[0],b[0]),min(c[0],d[0])) <= min(max(a[0],b[0]),max(c[0],d[0])) and max(min(a[1],b[1]),min(c[1],d[1])) <= min(max(a[1],b[1]),max(c[1],d[1])):
                            raise ValueError(oid + ': polygon must not self-intersect')
        if stroke['enabled'] != (stroke['width'] > 0):
            raise ValueError(oid + ': disabled stroke requires width=0; enabled requires width>0')
        if photos[oid].get('background_type') == 'photographic' and spec['subject']['mode'] != 'preserve':
            raise ValueError(oid + ': photographic background must preserve the scene')
        fade = spec['fade']
        if (spec.get('fade_direction', 'inside') == 'inside' and
                (fade['left']+fade['right'] > .9 or fade['top']+fade['bottom'] > .9)):
            raise ValueError(oid + ': opposite fades leave no opaque central region')
        if any(fade.values()):
            limits = fade_limits(analysis,oid,specs)
            for edge in EDGES:
                if fade[edge] > limits[edge] + 1e-6:
                    raise ValueError(f'{oid}: {edge} fade has no overlapping lower opaque coverage '
                                     f'for the full strip; maximum={limits[edge]}. '
                                     'Use zero fade on unsupported edges; keep the base opaque.')
    return original


def defaults(analysis):
    """Preserve explicit source links; do not guess visual effects after failure."""
    specs = []
    for photo in analysis['objects']:
        if not is_customer_photo(photo):
            continue
        specs.append({'photo_id': photo['id'], 'mask': {'enabled': False},
                      'subject': {'preserve_background': not bool(photo.get('linked_to')),
                                  'stroke': empty_stroke()},
                      'edge_gradient': {'enabled': False, 'direction': 'inside', **dict.fromkeys(EDGES, 0)},
                      'reason': '保守兜底：未完成窗口视觉判断；仅保留摆放范围与明确关联，不猜测固定窗口、描边或渐变'})
        if photo.get('linked_to'):
            specs[-1]['linked_to'] = photo['linked_to']
    return check({'schema_version': VERSION, 'photos': specs}, analysis)


def recover_fades(plan, analysis):
    """Keep complete visual judgments; only reduce unsupported fade parameters."""
    validate_plan(plan)
    result = deepcopy(plan)
    # Reject malformed/semantically invalid visual data before considering repairs.
    plain = deepcopy(plan)
    for spec in plain.get('photos', []):
        spec['edge_gradient'].update(enabled=False, **dict.fromkeys(EDGES, 0))
    check(plain,analysis)
    public = {p['photo_id']: p for p in result['photos']}
    specs = {p['photo_id']: p for p in execution_plan(result)['photos']}; notes = []
    for oid in analysis['layer_order']:
        if oid not in specs:
            continue
        spec = specs[oid]; limits = fade_limits(analysis,oid,specs)
        changes = []
        for edge in EDGES:
            before = spec['fade'][edge]
            if before > limits[edge]:
                spec['fade'][edge] = limits[edge]
                public[oid]['edge_gradient'][edge] = limits[edge]
                changes.append(f'{edge} {before:g}→{limits[edge]:g}')
        if changes:
            note = oid+'：下层不透明覆盖约束，'+', '.join(changes)
            public[oid]['edge_gradient']['enabled'] = any(spec['fade'].values())
            public[oid]['reason'] += '；程序修正：'+note+'；保留原形状与描边判断，待视觉复核'
            notes.append(note)
    return check(result,analysis), notes


def configure(photo, spec):
    """Compile the separate window plan into the renderer's photo state."""
    spec = execution_spec(spec)
    photo['photo_window_spec'] = deepcopy(spec)
    photo['mode'] = 'cutout' if spec['subject']['mode'] == 'cutout' else 'cover'
    for key in ('corner_radius', 'feather', 'outline_width', 'outline_color', 'card', 'stroke', 'stroke_width'):
        photo['style'].pop(key, None)
    # appearance is retained as reference information, but may not create another card.


def resolve_seams(plan, analysis):
    """Clamp unsupported window gradients and retain the current public contract."""
    return recover_fades(plan, analysis)


def shape_mask(size, shape):
    w, h = size
    aa = max(1, min(4, int(math.sqrt(16_000_000/max(1,w*h)))))
    mask = Image.new('L', (w*aa, h*aa)); draw = ImageDraw.Draw(mask)
    W, H = mask.size; box = (0, 0, W-1, H-1); kind = shape['type']
    if kind == 'rectangle':
        draw.rectangle(box, fill=255)
    elif kind == 'rounded_rect':
        draw.rounded_rectangle(box, radius=shape['corner_radius']*min(W,H), fill=255)
    elif kind in {'ellipse', 'circle'}:
        if kind == 'circle':
            diameter = min(W,H); x=(W-diameter)/2; y=(H-diameter)/2
            box = (x,y,x+diameter-1,y+diameter-1)
        draw.ellipse(box, fill=255)
    else:
        if kind == 'star':
            count = shape['star_points']; ratio = shape['inner_ratio']
            points = [(.5+.5*(1 if i%2 == 0 else ratio)*math.sin(i*math.pi/count),
                       .5-.5*(1 if i%2 == 0 else ratio)*math.cos(i*math.pi/count)) for i in range(count*2)]
        elif kind == 'heart':
            t = np.linspace(0, 2*math.pi, 240, endpoint=False)
            xs = 16*np.sin(t)**3; ys = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
            points = list(zip((xs-xs.min())/(xs.max()-xs.min()), (ys.max()-ys)/(ys.max()-ys.min())))
        else:
            points = shape['points']
        draw.polygon([(x*(W-1), y*(H-1)) for x,y in points], fill=255)
    return mask.resize(size, Image.Resampling.LANCZOS) if aa > 1 else mask


def apply_window(im, spec, subject_alpha=None, *, source_margin=0, placement_size=None):
    """Draw the independent subject and fixed-window contours."""
    from window_drawing import apply
    return apply(im, execution_spec(spec), subject_alpha, source_margin=source_margin,
                 placement_size=placement_size)


def canvas_mask(size, box, rotation, spec):
    """Geometric aperture only: never apply fade a second time in a carrier hole."""
    l,t,r,b = box; tile = shape_mask((max(1,r-l), max(1,b-t)), spec['shape'])
    if rotation:
        tile = tile.rotate(-rotation, Image.Resampling.BICUBIC, expand=True)
    result = Image.new('L', size)
    result.paste(tile, (round((l+r-tile.width)/2), round((t+b-tile.height)/2)))
    return result


def reference_crops(run, analysis):
    run = Path(run); folder = run/'prepared/photo-window-crops'; folder.mkdir(parents=True,exist_ok=True)
    images = []; regions = []
    with Image.open(run/'prepared/reference.png') as raw:
        for p in analysis['objects']:
            if not is_customer_photo(p):
                continue
            l,t,r,b = p['bbox']; margin = max(12, round(.12*max(r-l,b-t)))
            box = [max(0,l-margin),max(0,t-margin),min(raw.width,r+margin),min(raw.height,b+margin)]
            patch = raw.crop(box); patch.thumbnail((768,768))
            path = folder/(p['id']+'.png'); patch.save(path)
            images.append((f"参考 photo {p['id']} 周边裁片；位置见 crop_regions", path))
            regions.append({'photo_id':p['id'], 'crop_bbox':box})
    return images, regions


def previews(run, analysis, plan):
    """Render real alpha parameters; subject silhouette is unknown until binding."""
    check(plan, analysis)
    plan = execution_plan(plan)
    run = Path(run); folder = run/'previews'; folder.mkdir(parents=True,exist_ok=True)
    specs = {p['photo_id']:p for p in plan['photos']}; cells = []
    with Image.open(run/'prepared/reference.png') as raw:
        full = raw.convert('RGBA')
    reference = full.convert('RGB')
    for p in analysis['objects']:
        if not is_customer_photo(p):
            continue
        spec = specs[p['id']]; l,t,r,b = p['bbox']
        preview_spec = deepcopy(spec)
        if subject_stroke(spec):
            # No customer mask exists at reference analysis time.
            preview_spec['subject_stroke'].update(enabled=False, width=0)
        from window_drawing import sampling_margin
        margin = sampling_margin((r-l, b-t), preview_spec)
        tile, _ = apply_window(Image.new('RGBA',(r-l+2*margin,b-t+2*margin),'#30BAF0'),
                               preview_spec, source_margin=margin)
        alpha = tile.getchannel('A')
        # Unknown customer silhouette: geometry and fades only, visibly labelled below.
        tint = Image.new('RGBA',tile.size,'#00A8FF'); tint.putalpha(alpha.point(lambda v:round(v*.4)))
        if p.get('rotation'):
            tint = tint.rotate(-p['rotation'],Image.Resampling.BICUBIC,expand=True)
        full.alpha_composite(tint,(round((l+r-tint.width)/2),round((t+b-tint.height)/2)))
        ImageDraw.Draw(full).text((l+3,t+3),p['id'],fill='#FF0055',stroke_width=1,stroke_fill='white')
        tile.thumbnail((240,180))
        yy,xx = np.mgrid[:180,:240]; colors = np.where(((xx//12+yy//12)%2)[...,None],220,250)
        checker = Image.fromarray(np.repeat(colors,3,axis=2).astype('uint8')).convert('RGBA')
        checker.alpha_composite(tile,((240-tile.width)//2,(180-tile.height)//2))
        cell = Image.new('RGB',(480,240),'white'); cell.paste(checker,(240,36))
        patch = ImageOps.contain(reference.crop((l,t,r,b)),(236,180))
        cell.paste(patch,((236-patch.width)//2,36+(180-patch.height)//2))
        draw = ImageDraw.Draw(cell)
        window = spec['shape']['type'] if spec['window_enabled'] else 'window OFF'
        draw.text((4,4),p['id']+' / '+window,fill='black')
        draw.text((4,20),'reference | placement/window; subject='+spec['subject']['mode'],fill='black')
        draw.text((4,218),'fade T/B/L/R: '+ '/'.join(str(spec['fade'][e]) for e in EDGES),fill='black')
        if spec['subject']['mode']=='cutout' or subject_stroke(spec):
            draw.text((245,198),'subject silhouette: UNKNOWN',fill='#B00000')
        cells.append(cell)
    path = folder/'photo-windows.png'; full.convert('RGB').save(path)
    paths = [path]
    for offset in range(0,len(cells),8):
        page = cells[offset:offset+8]; sheet = Image.new('RGB',(960,240*math.ceil(len(page)/2)),'#EFEFEF')
        for i,cell in enumerate(page):
            sheet.paste(cell,((i%2)*480,(i//2)*240))
        path = folder/f'photo-window-masks-{offset//8+1}.png'; sheet.save(path); paths.append(path)
    return paths
