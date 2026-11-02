"""A rejected extraction never removes the structural photo frame."""
from copy import deepcopy
import math
from PIL import ImageChops
from overlays import draw_overlay
from collage.frame_links import carrier_photos


def align_frames(scene):
    """Build fallback paper in photo-local coordinates, including its pivot offset.

    Extracted frames keep their original canvas geometry. The snapshot permits
    screen keep/backing to restore it after a pending extraction used a fallback.
    """
    photos = {p['id']: p for p in scene['objects'] if p['kind'] == 'photo'}
    for obj in scene['objects']:
        if obj['kind'] != 'frame':
            continue
        if obj.get('recovered') or obj.get('generated'):
            previous = obj.pop('fallback_original_geometry', None)
            if previous:
                obj['bbox'] = previous['bbox']
                if previous['rotation'] is None:
                    obj.pop('rotation', None)
                else:
                    obj['rotation'] = previous['rotation']
            continue
        related = carrier_photos(obj, photos.values())
        # Multiple apertures share one rigid carrier. Never shrink that carrier
        # around the first photo or inherit just that photo's rotation.
        if len(related) != 1:
            continue
        photo = related[0]
        l, t, r, b = photo.get('window_bbox', photo['bbox'])
        short = min(r-l, b-t)
        if short <= 0:
            raise ValueError(obj['id'] + ': photo window has no area')
        border = obj.get('style', {}).get('border', {})
        widths = {}
        for edge in ('top', 'right', 'bottom', 'left'):
            value = border.get(edge, .04)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                value = .04
            widths[edge] = short * min(.3, max(0, value))
        top, right, bottom, left = (widths[e] for e in ('top', 'right', 'bottom', 'left'))
        width, height = r-l+left+right, b-t+top+bottom
        # The outer paper center is offset in photo-local axes. Rotate that
        # offset about the PHOTO center before rotating the outer rectangle.
        dx, dy = (right-left)/2, (bottom-top)/2
        angle = photo.get('rotation', 0)
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        cx, cy = (l+r)/2+c*dx-s*dy, (t+b)/2+s*dx+c*dy
        obj.setdefault('fallback_original_geometry', {
            'bbox': deepcopy(obj['bbox']), 'rotation': obj.get('rotation')})
        obj.update(bbox=[cx-width/2, cy-height/2, cx+width/2, cy+height/2], rotation=angle)


def draw_frame(obj, size, reference, photos):
    style=deepcopy(obj.get('style',{}))
    # With no explicit design colour, use a neutral paper placeholder. Sampling
    # bbox edges often measures the surrounding background rather than the frame.
    style.setdefault('fill','#F5F3EF')
    style['shape']='rounded_rectangle' if style.get('radius') else 'rectangle'
    style.pop('window',None)
    tile,_=draw_overlay({'kind':'overlay','method':'local','style':style},size,reference,photos)
    related=carrier_photos(obj, photos)
    photo=related[0] if len(related)==1 else {}
    shape=photo.get('photo_window_spec',{}).get('shape',{})
    if shape.get('type') in {'star','heart','circle','ellipse','polygon'}:
        from photo_windows import shape_mask
        tile.putalpha(ImageChops.multiply(tile.getchannel('A'),shape_mask(size,shape)))
    # The renderer subtracts the actual shaped/rotated photo aperture in canvas
    # coordinates. This preserves asymmetric borders and the original pivot.
    return tile,{'quality':'approximate','method':'local_frame','fallback':True,
                 'photo_id':obj['photo_id'],'fill':style['fill'],
                 'note':'提取相框不可用，已围绕照片内口扩边并共用旋转绘制替代相框；纹样细节为近似'}
