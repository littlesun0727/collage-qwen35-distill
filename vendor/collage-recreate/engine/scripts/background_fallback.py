"""Local background approximation from declared style or unobstructed color samples."""
from copy import deepcopy
import math

import numpy as np
from PIL import Image, ImageDraw

from overlays import draw_overlay


def sample_fill(obj, reference, scene):
    if reference is None:
        return '#EEEEEE', {'color_source': 'neutral_default', 'sample_pixels': 0}
    # Sample colors only, never reuse reference pixels as a rendered background.
    image = reference.convert('RGB')
    image.thumbnail((512, 512), Image.Resampling.NEAREST)
    rw, rh = scene['reference_size']
    sx, sy = image.width/rw, image.height/rh
    mask = Image.new('L', image.size)
    draw = ImageDraw.Draw(mask)

    def box(values, padding=0):
        l, t, r, b = values
        return (math.floor((l-padding)*sx), math.floor((t-padding)*sy),
                math.ceil((r+padding)*sx)-1, math.ceil((b+padding)*sy)-1)

    from reveal_chain import visible_bbox
    # Always use reference-space boxes; render may already have scaled obj.style.
    original = next((o for o in scene['objects'] if o['id'] == obj['id']), obj)
    draw.rectangle(box(original['bbox']), fill=255)
    for other in scene['objects']:
        if other['id'] == obj['id']:
            continue
        if other['kind'] == 'background':
            order = scene.get('layer_order', [])
            if obj['id'] in order and other['id'] in order and order.index(other['id']) < order.index(obj['id']):
                continue
        draw.rectangle(box(visible_bbox(other, (rw, rh)), max(rw, rh)*.005), fill=0)
    samples = np.asarray(image)[np.asarray(mask) > 0]
    if len(samples) < 16:
        return '#EEEEEE', {'color_source': 'neutral_default', 'sample_pixels': int(len(samples))}
    rgb = np.median(samples, axis=0).round().astype(int)
    return '#'+''.join(f'{v:02X}' for v in rgb), {
        'color_source': 'uncovered_reference_median', 'sample_pixels': int(len(samples))}


def draw_background(obj, size, reference, scene, reason=None):
    style = deepcopy(obj.get('style', {}))
    sampling = {'color_source': 'analysis_style'}
    if not style.get('fill'):
        style['fill'], sampling = sample_fill(obj, reference, scene)
    # Keep supported drawing parameters; sampling a reference tile is not local drawing.
    patterns = []
    for pattern in style.get('patterns', []):
        pattern = deepcopy(pattern)
        if pattern['kind'] == 'reference_tile':
            pattern['kind'] = pattern.get('fallback_motif', 'none')
            if pattern['kind'] == 'none':
                continue
        patterns.append(pattern)
    style['patterns'] = patterns
    style.pop('window', None)
    local = {'id': obj['id'], 'kind': 'background', 'method': 'local',
             'bbox': obj['bbox'], 'style': style}
    drawing_error = None
    try:
        tile, _ = draw_overlay(local, size)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        drawing_error = str(exc)
        try:
            tile = Image.new('RGBA', size, style['fill'])
        except (ValueError, TypeError):
            style['fill'] = '#EEEEEE'
            tile = Image.new('RGBA', size, style['fill'])
    reason = reason or obj.get('reveal_warning') or 'No completed reference background cleanup'
    metadata = {'quality': 'approximate', 'method': 'local_background', 'fallback': True,
                'background_clean': False, 'semantic_verified': False, 'fallback_reason': reason,
                'fill': style['fill'], **sampling,
                'note': '背景清底未完成，已使用本地近似绘制；原材质和复杂图案可能不同。原因：'+reason}
    if drawing_error:
        metadata['drawing_error'] = drawing_error
    return tile, metadata
