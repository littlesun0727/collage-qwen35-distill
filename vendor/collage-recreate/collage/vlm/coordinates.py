"""Model-facing reference coordinates; persisted workflow documents stay in pixels.

Only declared global geometry is converted. In particular style.window.bbox,
polygon points, source_crop, stroke/fade ratios and rotations are not global boxes.
"""
from collage.object_roles import is_customer_photo

from copy import deepcopy
import math
from collage.prompts.builder import reference_instructions as instructions

SLOT = '{{REFERENCE_COORDINATES}}'
VERSION = 2
# Deliberately explicit: Qwen2.5-VL and unknown future models must not match this.
NORMALIZED_MODELS = frozenset({'qwen/qwen3.8-flash', 'qwen/qwen3.8-max-0902',
                              'qwen3.8-flash', 'qwen3.8-max-0902', 'qwen3.8-max'})


def resolve_mode(model, setting='auto'):
    if setting not in {'auto', 'pixel', 'normalized_1000'}:
        raise ValueError('Unknown coordinate mode: ' + setting)
    return ('normalized_1000' if model in NORMALIZED_MODELS else 'pixel') if setting == 'auto' else setting


def model_context(context, mode):
    """Hide reference pixel dimensions only at the model boundary; never mutate canonical data."""
    value = deepcopy(context)
    if mode != 'normalized_1000':
        return value
    size = value.pop('reference_size', None)
    if isinstance(size, list) and len(size) == 2 and all(type(n) in (int, float) and n > 0 for n in size):
        value['reference_aspect_ratio'] = round(size[0] / size[1], 6)
    if isinstance(value.get('analysis'), dict):
        value['analysis'].pop('reference_size', None)
    for slot in value.get('photo_slots', []):
        width, height = slot.pop('width', None), slot.pop('height', None)
        if width is not None and height:
            slot.setdefault('aspect_ratio', round(width / height, 6))
    for asset in value.get('catalog', []):
        size = asset.pop('size', None)
        if size:
            asset['aspect_ratio'] = round(size[0] / size[1], 6)
    return value


def _records(document):
    """Yield named global rectangles only; never recursively scale arbitrary bbox keys."""
    if not isinstance(document, dict):
        return
    containers = [('', document)]
    if isinstance(document.get('analysis'), dict):
        containers.append(('analysis.', document['analysis']))
    for prefix, container in containers:
        for field in ('objects', 'text_units', 'photo_slots'):
            rows = container.get(field, [])
            if not isinstance(rows, list):
                continue
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                for name in ('bbox', 'window_bbox') if field in {'objects', 'photo_slots'} else ('bbox',):
                    if name in row:
                        yield (prefix + field, row.get('id', row.get('slot_id', index)), name), row, name
    regions = document.get('crop_regions', [])
    if isinstance(regions, list):
        for index, row in enumerate(regions):
            if isinstance(row, dict) and 'crop_bbox' in row:
                yield ('crop_regions', row.get('photo_id', index), 'crop_bbox'), row, 'crop_bbox'
    for index, photo in enumerate(document.get('photos', [])):
        placement = photo.get('placement') if isinstance(photo, dict) else None
        if isinstance(placement, dict) and 'bbox' in placement:
            yield ('photo_placement', photo.get('photo_id', index), 'bbox'), placement, 'bbox'


def _check_box(box, size, label):
    if (not isinstance(box, list) or len(box) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
        raise ValueError(f'{label}: expected four finite coordinates')
    l, t, r, b = box
    if not (0 <= l < r <= size[0] and 0 <= t < b <= size[1]):
        raise ValueError(f'{label}: expected positive box within [0,0,{size[0]},{size[1]}]')


def _bbox_schemas(schema):
    """Only the global geometry fields in reference/boxes outputs."""
    props = schema.get('properties', {})
    analysis = props.get('analysis', {}).get('properties', {})
    fields = analysis.get('objects', {}).get('items', {}).get('properties', {})
    for name in ('bbox', 'window_bbox'):
        if name in fields:
            yield fields[name]
    units = props.get('text_units', {}).get('items', {}).get('properties', {})
    if 'bbox' in units:
        yield units['bbox']
    for node in _placement_schemas(schema):
        if 'bbox' in node.get('properties', {}):
            yield node['properties']['bbox']


def _placement_schemas(schema):
    node = schema.get('properties', {}).get('photos', {}).get('items', {}).get('properties', {}).get('placement', {})
    return node.get('anyOf', [node])


class ReferenceCoordinates:
    """Adapt one request; same instance decodes replies, retries and recovery candidates."""
    def __init__(self, context, mode, input_only=False):
        self.mode = mode
        self.input_only = input_only
        self.size = deepcopy(context.get('reference_size'))
        if (not isinstance(self.size, list) or len(self.size) != 2 or
                any(type(n) is not int or n <= 0 for n in self.size)):
            raise ValueError('Reference coordinate requests require actual reference_size')
        self.context = deepcopy(context)
        self.originals = {}
        self.sent = {}
        for key, row, name in _records(self.context):
            _check_box(row[name], self.size, str(key))
            self.originals[key] = deepcopy(row[name])
            if mode == 'normalized_1000':
                row[name] = [round(v / self.size[i % 2] * 1000, 8) for i, v in enumerate(row[name])]
            self.sent[key] = deepcopy(row[name])
        self.context['coordinate_space'] = mode
        # Give the window stage actual aspect ratios; normalized width/height are not aspect ratios.
        if isinstance(context.get('analysis'), dict) and 'photo_slots' not in self.context:
            self.context['photo_slots'] = []
            for item in context['analysis'].get('objects', []):
                if is_customer_photo(item):
                    l, t, r, b = item['bbox']
                    self.context['photo_slots'].append({'slot_id': item['id'], 'width': r-l, 'height': b-t,
                                                       'aspect_ratio': round((r-l)/(b-t), 6)})
        self.context = model_context(self.context, mode)

    @property
    def metadata(self):
        value = {'mode': self.mode, 'canonical_mode': 'pixel', 'reference_size': self.size,
                 'scale': 1000 if self.mode == 'normalized_1000' else None, 'adapter_version': VERSION}
        # Preserve existing cache identities for stages that also output coordinates.
        if self.input_only:
            value['input_only'] = True
        return value

    def prompt(self, template):
        if template.count(SLOT) != 1:
            raise ValueError('Reference prompt must contain exactly one coordinate slot')
        return template.replace(SLOT, instructions(self.mode, self.input_only))

    def schema(self, canonical):
        schema = deepcopy(canonical)
        if self.input_only:
            return schema
        schema['properties']['coordinate_space'] = {'type': 'string', 'const': self.mode}
        schema['required'] = list(schema.get('required', [])) + ['coordinate_space']
        for box in _bbox_schemas(schema):
            box['items'] = {'type': 'number' if self.mode == 'normalized_1000' else 'integer',
                            'minimum': 0, 'maximum': 1000 if self.mode == 'normalized_1000' else max(self.size)}
            box['description'] = ('全局 [left,top,right,bottom]；' +
                                  ('0～1000 归一化坐标。' if self.mode == 'normalized_1000'
                                   else '原图像素；x<=reference_size[0]，y<=reference_size[1]。'))
        for node in _placement_schemas(schema):
            for point in node.get('properties', {}).get('corners', {}).get('properties', {}).values():
                point['items'] = {'type':'number','minimum':0,
                                  'maximum':1000 if self.mode=='normalized_1000' else max(self.size)}
                point['description'] = '全局参考图 [x,y] 坐标，不是裁片坐标或局部蒙版坐标'
        analysis = schema['properties'].get('analysis', {}).get('properties', {})
        if 'reference_size' in analysis:
            if self.mode == 'normalized_1000':
                del analysis['reference_size']
                node = schema['properties']['analysis']
                node['required'] = [key for key in node.get('required', []) if key != 'reference_size']
            else:
                analysis['reference_size']['const'] = self.size
        return schema

    def decode(self, raw):
        if self.input_only:
            return deepcopy(raw)
        if not isinstance(raw, dict) or raw.get('coordinate_space') != self.mode:
            raise ValueError('coordinate_space must match request: ' + self.mode)
        result = deepcopy(raw)
        result.pop('coordinate_space')
        if isinstance(result.get('analysis'), dict):
            analysis = result['analysis']
            if self.mode == 'normalized_1000' and 'reference_size' not in analysis:
                analysis['reference_size'] = deepcopy(self.size)
            if analysis.get('reference_size') != self.size:
                raise ValueError('reference_size must preserve the actual image dimensions')
        for key, row, name in _records(result):
            _check_box(row[name], [1000, 1000] if self.mode == 'normalized_1000' else self.size, str(key))
            if self.mode == 'normalized_1000':
                # Preserve untouched rectangles exactly, including immutable window_bbox fields.
                if key in self.sent and row[name] == self.sent[key]:
                    row[name] = deepcopy(self.originals[key])
                else:
                    row[name] = [round(v / 1000 * self.size[i % 2]) for i, v in enumerate(row[name])]
                _check_box(row[name], self.size, str(key) + ' converted pixels')
        for photo in result.get('photos', []):
            corners = photo.get('placement', {}).get('corners', {})
            for name, point in corners.items():
                limits = [1000,1000] if self.mode=='normalized_1000' else self.size
                if (not isinstance(point,list) or len(point)!=2 or any(
                        isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                        or not 0<=v<=limits[i] for i,v in enumerate(point))):
                    raise ValueError('Invalid global corner: '+name)
                if self.mode=='normalized_1000':
                    corners[name]=[v/1000*self.size[i] for i,v in enumerate(point)]
        return result
