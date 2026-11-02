"""Independent mask/subject strokes and directional geometric alpha gradients."""
import math

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageOps

from overlays import stroke_path


def sampling_margin(size, spec):
    if not spec or not spec['window_enabled'] or spec['fade_direction'] != 'outside':
        return 0
    fade = spec['fade']; w, h = size
    return math.ceil(max(w*fade['left'], w*fade['right'], h*fade['top'], h*fade['bottom']))


def gradient_mask(shape, size, fade, direction):
    """Nearest boundary normals select the local top/bottom/left/right widths."""
    if not any(fade.values()):
        return shape.copy()
    opaque = np.asarray(shape) > 127
    inside = direction == 'inside'
    region = opaque if inside else ~opaque
    _, labels = cv2.distanceTransformWithLabels(region.astype('uint8'), cv2.DIST_L2, 5,
                                               labelType=cv2.DIST_LABEL_PIXEL)
    y, x = np.indices(opaque.shape, dtype=np.int32)
    boundary = ~region
    near_x = np.zeros(int(labels.max())+1, dtype=np.int32)
    near_y = np.zeros_like(near_x)
    near_x[labels[boundary]] = x[boundary]
    near_y[labels[boundary]] = y[boundary]
    dx = (near_x[labels]-x if inside else x-near_x[labels]).astype(np.float32)
    dy = (near_y[labels]-y if inside else y-near_y[labels]).astype(np.float32)
    w, h = size
    sx = np.where(dx < 0, w*fade['left'], w*fade['right'])
    sy = np.where(dy < 0, h*fade['top'], h*fade['bottom'])
    # A zero width leaves that side hard. Half-pixel correction follows the
    # geometric boundary between the nearest opaque and transparent pixels.
    rx = np.maximum(0, np.abs(dx)-.5)/np.maximum(sx, 1e-6)
    ry = np.maximum(0, np.abs(dy)-.5)/np.maximum(sy, 1e-6)
    # Pick the boundary's principal local direction. A disabled left edge must
    # not cancel the top gradient along an upper, sloping part of an ellipse.
    amount = np.clip(np.where(np.abs(dx) > np.abs(dy), rx, ry), 0, 1)
    amount = amount*amount*(3-2*amount)
    alpha = amount if inside else 1-amount
    original = np.asarray(shape, dtype=np.float32)/255
    alpha = np.minimum(original, alpha) if inside else np.maximum(original, alpha)
    if not inside:
        px, py = (shape.width-w)//2, (shape.height-h)//2
        extent = ((x >= px-w*fade['left']) & (x < px+w+w*fade['right']) &
                  (y >= py-h*fade['top']) & (y < py+h+h*fade['bottom']))
        alpha *= extent
    return Image.fromarray(np.rint(alpha*255).astype('uint8'))


def rectangle_gradient(size, fade, direction, margin):
    """Four independent strips, including corners; no nearest-edge partition."""
    w, h = size
    y, x = np.mgrid[:h+2*margin, :w+2*margin].astype(np.float32)
    distances = {'left': x-margin+.5, 'right': margin+w-x-.5,
                 'top': y-margin+.5, 'bottom': margin+h-y-.5}
    alpha = np.ones_like(x)
    for edge, distance in distances.items():
        span = fade[edge]*(w if edge in {'left', 'right'} else h)
        if span:
            amount = np.clip(distance/span + (1 if direction == 'outside' else 0), 0, 1)
            alpha *= amount*amount*(3-2*amount)
        else:
            alpha *= distance > 0
    return Image.fromarray(np.rint(alpha*255).astype('uint8'))


def stroke_mask(boundary, stroke, short):
    ink = Image.new('L', boundary.size)
    if not stroke['enabled']:
        return ink
    width = max(1, stroke['width']*short)
    offset = stroke['offset']*short
    distance = {'inside': -offset-width/2, 'center': 0, 'outside': offset+width/2}[stroke['placement']]
    mask = (np.asarray(boundary) > 127).astype('uint8')
    if not mask.any():
        return ink
    inner = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    outer = cv2.distanceTransform(1-mask, cv2.DIST_L2, 5)
    region = (outer-inner <= distance).astype('uint8')
    contours, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    dash = None if stroke['pattern'] == 'solid' else [v*short for v in stroke['dash']]
    if stroke['pattern'] == 'dot':
        dash = [max(.1, width*.05), max(width*1.7, dash[1])]
    draw = ImageDraw.Draw(ink)
    for contour in contours:
        points = [tuple(map(float, p[0])) for p in contour]
        if len(points) > 1:
            stroke_path(draw, points+[points[0]], 255, max(1, round(width)), dash, stroke['line_cap'])
    return ink


def apply(im, spec, subject_alpha=None, *, source_margin=0, placement_size=None):
    from photo_windows import shape_mask
    size = (im.width-2*source_margin, im.height-2*source_margin)
    if min(size) <= 0:
        raise ValueError('Source margin leaves no photo window')
    # Free linked subjects can use a canvas-sized buffer without enlarging their
    # normalized outline width, dash spacing or offset.
    short = min(placement_size or size)
    strokes = [spec['subject_stroke'], spec['stroke']]
    margin = max(source_margin, sampling_margin(size, spec),
                 *(math.ceil((s['width']+s['offset'])*short)+3 for s in strokes if s['enabled']), 1)
    extra = margin-source_margin
    pixels = ImageOps.expand(im, extra, fill=0)
    free_subject = not spec['window_enabled'] and spec['subject']['mode'] == 'cutout'
    if not spec['window_enabled']:
        if spec['stroke']['enabled'] or any(spec['fade'].values()):
            raise ValueError('Disabled fixed window cannot draw a window stroke or gradient')
        shape = None
        # A free cutout's placement box is not a clipping window. Keep the
        # transparent padding available for its outside silhouette stroke.
        gradient = (Image.new('L', pixels.size, 255) if free_subject else
                    ImageOps.expand(Image.new('L', size, 255), margin, fill=0))
    else:
        shape = ImageOps.expand(shape_mask(size, spec['shape']), margin, fill=0)
        if spec['shape']['type'] == 'rectangle':
            gradient = rectangle_gradient(size, spec['fade'], spec['fade_direction'], margin)
        else:
            gradient = gradient_mask(shape, size, spec['fade'], spec['fade_direction'])
    content = ImageChops.multiply(pixels.getchannel('A'), gradient)
    tile = pixels.copy(); tile.putalpha(content)
    if spec['subject_stroke']['enabled']:
        if spec['subject']['mode'] == 'preserve':
            if subject_alpha is None or subject_alpha.mode != 'L' or subject_alpha.size != im.size:
                raise ValueError('Preserved subject stroke requires an aligned L subject mask')
            if not subject_alpha.getbbox():
                raise ValueError('Preserved subject stroke has no visible subject')
            subject_boundary = ImageChops.multiply(subject_alpha, im.getchannel('A'))
        else:
            subject_boundary = im.getchannel('A')
        # Free cutouts may be tightly trimmed to their silhouette: transparent
        # padding lets the outline close around a head/hand touching that box.
        # Fixed crops continue edge values to avoid outlining a cut-off limb.
        subject_boundary = Image.fromarray(np.pad(np.asarray(subject_boundary), extra,
                                                  mode='constant' if free_subject else 'edge'))
    else:
        subject_boundary = None
    for stroke, boundary in zip(strokes, (subject_boundary, shape)):
        if not stroke['enabled']:
            continue
        ink = stroke_mask(boundary, stroke, short)
        if boundary is subject_boundary or any(spec['fade'].values()):
            ink = ImageChops.multiply(ink, gradient)
        color = Image.new('RGBA', tile.size, stroke['color']); color.putalpha(ink)
        if boundary is subject_boundary and stroke['placement'] == 'outside' and spec['subject']['mode'] == 'cutout':
            tile = Image.alpha_composite(color, tile)
        else:
            tile = Image.alpha_composite(tile, color)
            content = ImageChops.multiply(content, ImageChops.invert(ink))
    return tile, content
