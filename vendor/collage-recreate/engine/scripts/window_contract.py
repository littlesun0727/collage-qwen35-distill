"""The current drawing contract and its one-way compilation for rendering."""
from copy import deepcopy

from common import ROOT, read


SCHEMA = read(ROOT/'schemas/photo-windows.schema.json')
VERSION = SCHEMA['properties']['schema_version']['const']
EDGES = ('top', 'bottom', 'left', 'right')


def empty_stroke():
    return dict(enabled=False, width=0, color='#FFFFFF', pattern='solid',
                line_cap='butt', placement='inside', offset=0)


def execution_spec(spec):
    """Normalize a copy; never overwrite the public drawing plan or analysis."""
    if 'window_enabled' in spec:
        return deepcopy(spec)
    validate_drawing(spec)
    result = {k: deepcopy(v) for k, v in spec.items()
              if k not in {'mask', 'edge_gradient', 'subject'}}
    gradient = spec['edge_gradient']
    mask = spec['mask']; enabled = mask['enabled']
    result.update(shape=deepcopy(mask['shape']) if enabled else {'type': 'rectangle'},
                  stroke=deepcopy(mask['stroke'] if enabled else empty_stroke()),
                  subject={'mode': 'preserve' if spec['subject']['preserve_background'] else 'cutout'},
                  subject_stroke=deepcopy(spec['subject']['stroke']),
                  fade={edge: gradient[edge] for edge in EDGES},
                  fade_direction=gradient['direction'], window_enabled=enabled)
    return result


def execution_plan(plan):
    result = deepcopy(plan)
    result['photos'] = [execution_spec(p) for p in plan['photos']]
    return result


def validate_drawing(spec):
    """Cross-field rules that provider schema projections cannot enforce."""
    gradient = spec['edge_gradient']
    mask = spec['mask']
    if mask['enabled'] is False:
        if set(mask) != {'enabled'}:
            raise ValueError(spec['photo_id'] + ': disabled window must omit shape and stroke')
        if gradient['enabled'] or any(gradient[e] > 0 for e in EDGES):
            raise ValueError(spec['photo_id'] + ': edge gradient requires an enabled fixed window')
    if gradient['enabled'] != any(gradient[e] > 0 for e in EDGES):
        raise ValueError(spec['photo_id'] + ': gradient enabled must match its nonzero widths')
    strokes = [spec['subject']['stroke']]
    if mask['enabled']:
        strokes.append(mask['stroke'])
    for stroke in strokes:
        if stroke['enabled'] != (stroke['width'] > 0):
            raise ValueError(spec['photo_id'] + ': stroke enabled must match width>0')
        if stroke['pattern'] in {'dash', 'dot'} and 'dash' not in stroke:
            raise ValueError(spec['photo_id'] + ': dashed/dotted stroke requires dash lengths')
        if stroke['pattern'] == 'solid' and 'dash' in stroke:
            raise ValueError(spec['photo_id'] + ': solid stroke must omit dash lengths')
        if stroke['pattern'] == 'dot' and stroke['line_cap'] != 'round':
            raise ValueError(spec['photo_id'] + ': dotted stroke requires round caps')
        if stroke['placement'] == 'center' and stroke['offset']:
            raise ValueError(spec['photo_id'] + ': centered stroke requires offset=0')
