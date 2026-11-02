"""Closed data contracts shared by model requests and local validation."""

from copy import deepcopy
import sys
from collage.config import ROOT, ENGINE
from collage.storage import read

if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

STRING = {'type': 'string'}


def obj(properties, required=None):
    return {'type': 'object', 'properties': properties,
            'required': list(properties) if required is None else required, 'additionalProperties': False}


def array(items):
    return {'type': 'array', 'items': items}


def inline_schema(name):
    documents = {key: read(ROOT / 'engine' / 'schemas' / f'{key}.schema.json')
                 for key in ['analysis', 'bindings', 'review']}
    def expand(value, document):
        if isinstance(value, list):
            return [expand(v, document) for v in value]
        if not isinstance(value, dict):
            return value
        if '$ref' in value:
            uri, _, fragment = value['$ref'].partition('#')
            target = documents[uri.rsplit(':', 1)[-1]] if uri else document
            node = target
            for part in fragment.strip('/').split('/'):
                if part:
                    node = node[part.replace('~1', '/').replace('~0', '~')]
            return expand(node, target)
        return {k: expand(v, document) for k, v in value.items() if k not in {'$schema', '$id', '$defs'}}
    return expand(documents[name], documents[name])


ANALYSIS = inline_schema('analysis')
_object_schema = ANALYSIS['properties']['objects']['items']
_object_schema['properties']['kind']['description'] = (
    '需替换的摄影内容用 photo 绑定客户素材；固定模板背景可本地绘制或从参考清底，使用 background。'
    '普通可编辑文字才用 text；固定艺术字、完整文字标签及装饰用 overlay。'
    '需要独立提取的单窗装饰相框用 frame，method=extract，photo_id 必须指向承载照片；简单几何单色轮廓属于 photo。')
_object_schema['properties']['description']['description'] = (
    '描述完整制作单元及实际素材边界。独立贴纸不因重叠合并；合并须写明共同载体或整体设计依据、成员及内部遮挡。')
_object_schema.setdefault('allOf', []).extend([{'if': {'properties': {'kind': {'const': 'text'}}, 'required': ['kind']},
                          'then': {'required': ['text', 'text_status'],
                                   'properties': {'method': {'const': 'local'}}}}])
_object_schema['allOf'] += [
    # These restrictions already exist in engine validation. Put them in the
    # model contract as well so strict-output providers can prevent the mistakes.
    {'if': {'properties': {'kind': {'enum': ['text', 'overlay', 'frame']}},
            'required': ['kind']},
     'then': {'properties': {'window_bbox': False, 'mode': False, 'appearance': False}}},
    {'if': {'properties': {'kind': {'enum': ['background', 'text', 'photo']}},
            'required': ['kind']},
     'then': {'properties': {'photo_id': False}}},
    {'if': {'properties': {'kind': {'const': 'background'}}, 'required': ['kind']},
     'then': {'required': ['background_type', 'method'], 'properties': {
         'background_type': {'enum': ['photo', 'solid_color', 'reference_clean', 'simple_design', 'complex_design']}}}},
    {'if': {'properties': {'background_type': {'const': 'photo'}}, 'required': ['background_type']},
     'then': {'properties': {'kind': {'const': 'background'}, 'method': {'const': 'local'},
                             'mode': {'const': 'cover'}, 'linked_to': False, 'parent_id': False}}},
    {'if': {'properties': {'background_type': {'const': 'photographic'}}, 'required': ['background_type']},
     'then': {'properties': {'kind': {'const': 'photo'}, 'mode': {'enum': ['cover', 'feather', 'stroke']},
                             'method': {'const': 'local'}}}},
    {'if': {'properties': {'background_type': {'enum': ['solid_color', 'simple_design', 'complex_design']}},
            'required': ['background_type']},
     'then': {'required': ['style'], 'properties': {'kind': {'const': 'background'},
               'method': {'const': 'local'}, 'style': {'required': ['fill']}}}},
    {'if': {'properties': {'background_type': {'const': 'reference_clean'}}, 'required': ['background_type']},
     'then': {'properties': {'kind': {'const': 'background'}, 'method': {'const': 'extract'}}}},
]
BINDINGS = inline_schema('bindings')
# New selection calls must encode the crop they describe. Existing engine files
# retain their optional defaults for backwards compatibility.
BINDING_PLAN = deepcopy(BINDINGS)
BINDING_PLAN['properties']['photos']['items']['required'] += ['crop_center', 'source_crop', 'mirror_x']
STYLE = deepcopy(ANALYSIS['properties']['objects']['items']['properties']['style'])
CHANGES = inline_schema('review')['properties']['items']['items']['properties']['changes']
# Revision nodes only expose the finite style controls they can safely execute.
# Rich shape/path construction belongs to initial analysis or explicit screening.
CHANGES['properties']['style']['properties'] = {
    k: v for k, v in CHANGES['properties']['style']['properties'].items()
    if k in {'fill', 'stroke', 'stroke_width', 'font', 'font_size', 'bold', 'italic',
             'opacity', 'align', 'line_spacing', 'corner_radius'}
}
IDENTIFIER = deepcopy(ANALYSIS['properties']['objects']['items']['properties']['id'])
TEXT_UNIT = obj({
    'id': IDENTIFIER,
    'bbox': deepcopy(ANALYSIS['properties']['objects']['items']['properties']['bbox']),
    'content': {'type': ['string', 'null']},
    'treatment': {'enum': ['plain_text', 'editable_label', 'fixed_artwork']},
    'has_backing': {'type': 'boolean'},
    'decorative_lettering': {'type': 'boolean'},
    'edit_requested': {'type': 'boolean'},
    'text_id': {'anyOf': [IDENTIFIER, {'type': 'null'}]},
    'carrier_id': {'anyOf': [IDENTIFIER, {'type': 'null'}],
                   'description': 'fixed_artwork 必须指向已建立的 kind=overlay/frame, method=extract 对象，不能指向 text；框体上的印字随 frame 保留'},
    'reason': STRING,
})
TEXT_UNITS = array(TEXT_UNIT)
REFERENCE = obj({'analysis': ANALYSIS, 'text_units': TEXT_UNITS})
BUNDLE = obj({'analysis': ANALYSIS, 'bindings': BINDINGS, 'text_units': TEXT_UNITS})
from window_contract import SCHEMA as PHOTO_WINDOWS
BOXES = obj({'decision': {'enum': ['keep', 'replace']}, 'summary': STRING,
             'analysis': ANALYSIS, 'text_units': TEXT_UNITS,
             'photo_windows': PHOTO_WINDOWS}, ['decision', 'summary'])
ISSUE = obj({'id': STRING, 'reason': STRING})
ADJUSTMENT = obj({'id': STRING, 'reason': STRING, 'changes': CHANGES})
REVIEW = obj({'verdict': {'enum': ['pass', 'needs_changes']}, 'summary': STRING,
              'issues': array(ISSUE), 'adjustments': array(ADJUSTMENT), 'layer_order': array(STRING)})
FINAL_REVIEW = obj({'verdict': {'enum': ['pass', 'needs_changes']}, 'summary': STRING, 'issues': array(ISSUE)})
SCREEN = obj({'decisions': array(obj({'id': STRING, 'action': {'enum': ['keep', 'drop', 'backing']},
                                    'reason': STRING, 'photo_content': {'enum': ['present', 'absent', 'uncertain']},
                                    'embedded_ids': array(STRING)},
                                   ['id', 'action', 'reason', 'photo_content']))})
EDIT = obj({'decision': {'enum': ['edit', 'clarify', 'unsupported']}, 'summary': STRING, 'question': STRING,
            'changes': array(ADJUSTMENT), 'remove_ids': array(STRING), 'layer_order': array(STRING)})


def validate_reference(value, run, check_text_units=True, geometry_warnings=None):
    from validate import analysis_check
    analysis_check(value['analysis'], read(run / 'input.json'))
    validate_backgrounds(value['analysis'])
    for item in value['analysis']['objects']:
        if item['kind'] == 'overlay' and item.get('method', 'local') == 'local' and not item.get('style', {}).get('shape'):
            raise ValueError(item['id'] + ': local overlay requires an explicit supported shape')
    if check_text_units:
        validate_text_units(value['text_units'], value['analysis'], geometry_warnings)


def validate_bindings(value, analysis, run):
    from validate import bindings_check
    bindings_check(value, analysis, read(run / 'prepared/catalog.json'))




def validate_bundle(value, run, check_text_units=True):
    validate_reference(value, run, check_text_units)
    validate_bindings(value['bindings'], value['analysis'], run)


def validate_backgrounds(analysis):
    """Validate declared production routes, leaving visual classification to the VLM."""
    background_ids = set()
    for item in analysis['objects']:
        kind = item['kind']; route = item.get('background_type'); style = item.get('style', {})
        if kind == 'background':
            if route not in {'photo', 'solid_color', 'reference_clean', 'simple_design', 'complex_design'}:
                raise ValueError(item['id'] + ': background requires photo, solid_color, reference_clean or a legacy design route')
            if route == 'photo':
                if item.get('method') != 'local' or item.get('mode', 'cover') != 'cover':
                    raise ValueError(item['id'] + ': photo background requires local customer imagery in cover mode')
                if item.get('linked_to') or item.get('parent_id'):
                    raise ValueError(item['id'] + ': canvas background cannot be a child or linked foreground')
                if item['bbox'] != [0, 0, *analysis['reference_size']] or item.get('rotation', 0):
                    raise ValueError(item['id'] + ': photo background must fill canvas without rotation')
                if analysis['layer_order'][0] != item['id']:
                    raise ValueError(item['id'] + ': photo background must be bottommost')
            elif route == 'reference_clean':
                if item.get('method') != 'extract':
                    raise ValueError(item['id'] + ': reference_clean background requires method=extract')
            elif item.get('method') != 'local' or not style.get('fill'):
                raise ValueError(item['id'] + ': design background requires method=local and explicit style.fill; extraction/placeholder/default white is forbidden')
            if style.get('shape', 'rectangle') not in {'rectangle', 'paper'}:
                raise ValueError(item['id'] + ': background uses rectangle/paper; independent shapes belong to overlays')
        elif route == 'photographic':
            if kind != 'photo' or item.get('mode', 'cover') not in {'cover', 'feather', 'stroke'}:
                raise ValueError(item['id'] + ': photographic background must be a bound photo in cover/feather/stroke mode; cutout is a foreground photo')
            if item.get('method', 'local') != 'local':
                raise ValueError(item['id'] + ': photographic background uses customer pixels, not reference extraction')
        elif route is not None:
            raise ValueError(item['id'] + ': background_type only applies to customer photos or fixed backgrounds')
        if kind == 'background' or route == 'photographic':
            background_ids.add(item['id'])
    objects = {item['id']: item for item in analysis['objects']}
    preceding_foreground = []
    for oid in analysis['layer_order']:
        if oid not in background_ids:
            preceding_foreground.append(objects[oid])
            continue
        # Disjoint collage panels may be interleaved with another panel's text.
        # Only a positive-area intersection can obscure that earlier foreground.
        x0, y0, x1, y1 = objects[oid]['bbox']
        for foreground in preceding_foreground:
            fx0, fy0, fx1, fy1 = foreground['bbox']
            if max(x0, fx0) < min(x1, fx1) and max(y0, fy0) < min(y1, fy1):
                raise ValueError(oid + ': background layers must precede overlapping foreground ' + foreground['id'])


def validate_text_units(units, analysis, geometry_warnings=None):
    """Check declared text ownership, without pretending to infer image semantics."""
    from collage.text_units import validate
    validate(units, analysis, geometry_warnings)


def _preserve_box_records(before, after, key, editable, label):
    """Full replacements may correct geometry and append records, never drop content."""
    updated = {item[key]: item for item in after}
    for item in before:
        identifier = item[key]
        if identifier not in updated:
            raise ValueError(f'boxes must preserve existing {label} ID: {identifier}')
        changed = {field for field in item.keys() | updated[identifier].keys()
                   if field not in editable and (field not in item or field not in updated[identifier]
                                                or item[field] != updated[identifier][field])}
        if changed:
            raise ValueError(f'boxes cannot change {label} {identifier} fields: ' + ', '.join(sorted(changed)))


def validate_box_decision(value, current, run, geometry_warnings=None):
    if value['decision'] == 'replace':
        from copy import deepcopy
        from collage.workflow.support.reference_cleanup import normalize
        before, after = deepcopy(current['analysis']), value['analysis']
        # Compare metadata in the same normalized form as an incoming reply.
        # Existing legacy null/empty OCR fields carry no visual content.
        normalize({'analysis': before})
        if {k: v for k, v in before.items() if k not in {'objects', 'layer_order'}} != {
                k: v for k, v in after.items() if k not in {'objects', 'layer_order'}}:
            raise ValueError('boxes must preserve analysis metadata and reference_size')
        # The inner photo aperture is geometry too. Its containment and photo-only
        # ownership are still checked by validate_reference below.
        _preserve_box_records(before['objects'], after['objects'], 'id', {'bbox', 'window_bbox', 'rotation'}, 'object')
        old_ids = {item['id'] for item in before['objects']}
        if [oid for oid in after['layer_order'] if oid in old_ids] != before['layer_order']:
            raise ValueError('boxes must preserve existing relative layer_order; only insert missing objects')
        _preserve_box_records(current['text_units'], value['text_units'], 'id', {'bbox'}, 'text unit')
        validate_reference(value, run, geometry_warnings=geometry_warnings)
        if 'photo_windows' in current or 'photo_windows' in value:
            from photo_windows import resolve_seams
            if 'photo_windows' not in value:
                raise ValueError('replace must include the complete photo_windows plan')
            if 'photo_windows' not in current:
                raise ValueError('boxes cannot create a photo_windows plan when none was supplied')
            # Geometry can reduce the overlap supporting an existing fade. The
            # program owns that adjustment; the model still cannot redesign a
            # window. Build the expected plan from original windows plus new slots.
            original = {p['photo_id']: p for p in current['photo_windows']['photos']}
            baseline = deepcopy(value['photo_windows'])
            baseline['photos'] = [deepcopy(original.get(p['photo_id'], p)) for p in baseline['photos']]
            resolved, notes = resolve_seams(baseline, after)
            expected = {p['photo_id']: p for p in resolved['photos']}
            echoed = deepcopy(value['photo_windows']['photos'])
            for spec in echoed:
                oid = spec['photo_id']
                # Cached accepted replies and apply/replay already contain the
                # deterministic adjustment. Accept that exact result as well.
                if oid in original and spec == expected[oid]:
                    spec.clear()
                    spec.update(deepcopy(original[oid]))
            _preserve_box_records(current['photo_windows']['photos'], echoed,
                                  'photo_id', set(), 'photo window')
            value['photo_windows'] = resolved
            if geometry_warnings is not None:
                geometry_warnings.extend({'code': 'boxes_seam_adjusted', 'message': note} for note in notes)
    elif any(key in value and value[key] != current.get(key) for key in ['analysis', 'bindings', 'text_units', 'photo_windows']):
        raise ValueError('keep cannot change analysis, bindings, text_units or photo_windows; use replace for a correction')
    # Some providers echo unchanged optional fields. Exact echoes authorize no mutation.


def validate_review(value, scene, allow_adjustments=True):
    from validate import review_check
    ids = {o['id'] for o in scene['objects']}
    issues = value.get('issues', [])
    if len({i['id'] for i in issues}) != len(issues) or any(i['id'] not in ids for i in issues):
        raise ValueError('Issues must reference unique existing scene IDs')
    adjustments = value.get('adjustments', [])
    order = value.get('layer_order', [])
    if not allow_adjustments and (adjustments or order):
        raise ValueError('Final review cannot request another adjustment')
    if value['verdict'] == 'pass' and (issues or adjustments or order):
        raise ValueError('pass cannot contain unresolved issues or unapplied changes')
    if value['verdict'] == 'needs_changes' and not (issues or adjustments or order):
        raise ValueError('needs_changes requires specific object issues')
    if adjustments or order:
        check = {'schema_version': 'collage-review-v1', 'render_id': '0'*16, 'verdict': 'needs_changes',
                 'summary': value['summary'], 'items': [{'id': i['id'], 'action': 'adjust',
                     'reason': i['reason'], 'changes': i['changes']} for i in adjustments]}
        if order:
            check['layer_order'] = order
        review_check(check, scene)
