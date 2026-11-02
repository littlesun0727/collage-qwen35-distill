"""阶段 1：参考图 -> 对象清单与显式图层顺序。校验字段、关联并编译。"""

from copy import deepcopy
import math
from jsonschema import Draft202012Validator, ValidationError
from collage import contracts, prompts
from collage.vlm.client import ModelError, ModelTransportError
from collage.vlm.coordinates import resolve_mode
from collage.storage import read, save
from collage.vlm.response_normalization import unwrap_object
from collage.workflow.inventory import schema


VERSION = 'structure-analysis-v9-12-frame-photo-list'
ID = {'type': 'string', 'pattern': '^[a-z][a-z0-9_-]{0,63}$'}
COLOR = {'type': 'string', 'pattern': '^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$'}
TEXT_STYLE_FIELDS = ('font', 'align', 'bold', 'italic')


def execute(ctx):
    ctx.progress(2, 'running', '识别完整对象、范围、归属和层级')
    size = ctx.reference_context()['reference_size']
    mode = resolve_mode(ctx.config.for_stage('analyze').model, ctx.config.for_stage('analyze').coordinate_mode)
    prompt = prompts.render('analyze', mode=mode)
    contract = schema(size, mode)
    check = lambda v: compile_inventory(v, size, mode, ctx.run)
    evidence = {'version': VERSION, 'coordinate_mode': mode, 'first_pass_valid': False, 'repaired': False}
    try:
        value = ctx.client.ask('analyze', prompt, {'reference_size': size}, ctx.analysis_images(),
                                  contract, check, single_request=True, normalize_answer=normalize, reference_mode=mode)
        evidence['first_pass_valid'] = True
    except ModelTransportError as error:
        evidence.update(status='transport_failed', first_error=str(error))
        save(ctx.control/'structure-analysis.json', evidence)
        raise
    except ModelError as error:
        evidence['first_error'] = str(error)
        candidates = ctx.candidates('analyze')
        prior = normalize(candidates[0]) if candidates else None
        plan = repair_plan(prior, size, mode)
        save(ctx.control/'structure-analysis.json', evidence)
        if plan:
            save(ctx.control/'structure-repair-plan.json', plan)
            patch = ctx.client.ask('analyze_repair', prompts.render('analyze_repair', mode=mode),
                                      {'reference_size': size, 'previous': prior, 'issues': plan['issues'],
                                       'allowed_updates': plan['fields']},
                                      ctx.analysis_images(), plan['schema'],
                                      lambda p: check(merge_repair(prior, p, plan)), single_request=True,
                                      normalize_answer=normalize_repair, reference_mode=mode)
            value = merge_repair(prior, patch, plan)
            evidence['repair_kind'] = 'field_patch'
        else:
            value = ctx.client.ask('analyze_repair', prompts.render('analyze_full_retry', mode=mode),
                                      {'reference_size': size, 'validation_error': str(error)}, ctx.analysis_images(),
                                      contract, check, single_request=True, normalize_answer=normalize, reference_mode=mode)
            evidence['repair_kind'] = 'full_retry'
        evidence['repaired'] = True
    bundle = check(value)
    save(ctx.run/'structure.json', {'version': VERSION, 'coordinate_mode': mode, **value})
    evidence.update(status='accepted', objects=len(bundle['analysis']['objects']),
                    generated_text_units=len(bundle['text_units']), aabb_geometry=True)
    evidence['normalizations'] = [read(p) for p in sorted((ctx.control/'vlm/analyze').glob('*/normalization-0.json'))]
    save(ctx.control/'structure-analysis.json', evidence)
    ctx.write_reference(bundle)


def legacy_schema(size, mode):
    number = {'type': 'number' if mode == 'normalized_1000' else 'integer',
              'minimum': 0, 'maximum': 1000 if mode == 'normalized_1000' else max(size)}
    item = contracts.obj({
        'id': ID, 'kind': {'enum': ['background', 'photo', 'overlay', 'frame', 'text']},
        'background_type': {'enum': ['photo', 'reference_clean', 'solid_color']},
        'bbox': {'type': 'array', 'items': number, 'minItems': 4, 'maxItems': 4},
        'description': {'type': 'string', 'minLength': 1, 'maxLength': 400},
        'linked_to': ID,
        'photo_id': {'anyOf': [ID, {'type': 'array', 'items': ID, 'minItems': 1,
                                   'maxItems': 160, 'uniqueItems': True}]}, 'parent_id': ID,
        'text': {'type': ['string', 'null'], 'description': '保留参考图逐行内容；JSON 中用 \\n 表示实际换行，不能合并成单行。'},
        'color': COLOR,
        **{key: deepcopy(contracts.STYLE['properties'][key]) for key in TEXT_STYLE_FIELDS},
    }, ['id', 'kind', 'bbox'])
    item['allOf'] = [
        {'if': {'properties': {'kind': {'const': 'background'}}},
         'then': {'required': ['background_type'],
                  'properties': {'linked_to': False, 'parent_id': False}},
         'else': {'properties': {'background_type': False}}},
        {'if': {'properties': {'background_type': {'const': 'solid_color'}},
                'required': ['background_type']},
         'then': {'required': ['color']}},
        {'if': {'properties': {'background_type': {'enum': ['photo', 'reference_clean']}},
                'required': ['background_type']},
         'then': {'properties': {'color': False}}},
        {'if': {'properties': {'kind': {'const': 'text'}}},
         'then': {'required': ['text']}, 'else': {'required': ['description']}},
        {'if': {'properties': {'kind': {'const': 'frame'}}}, 'then': {'required': ['photo_id']}},
        {'if': {'properties': {'kind': {'const': 'photo'}}},
         'else': {'properties': {'linked_to': False}}}]
    return contracts.obj({
        # Count/ordering are checked by problems(), including providers whose
        # strict schema subset cannot express array-wide contains constraints.
        'objects': {'type': 'array', 'items': item, 'minItems': 1, 'maxItems': 161},
        'layer_order': {'type': 'array', 'items': ID, 'minItems': 1, 'maxItems': 161,
                        'uniqueItems': True},
    })


def normalize(value):
    """Normalize containers, aliases and layer IDs without reordering valid layers."""
    result = unwrap_object(value, {'objects'})
    if not isinstance(result, dict) or not isinstance(result.get('objects'), list):
        return result
    rows = result['objects']
    if len(rows) == 1 and isinstance(rows[0], list) and all(isinstance(o, dict) for o in rows[0]):
        result['objects'] = rows[0]
    for item in result['objects']:
        if not isinstance(item, dict):
            continue
        for key in ('photo_id', 'parent_id', 'linked_to', 'background_type', 'color', *TEXT_STYLE_FIELDS):
            if key in item and item[key] is None:
                del item[key]
        for key, owner in (('text', 'text'),):
            if item.get('kind') != owner and item.get(key, 1) is None:
                del item[key]
        if 'fill' in item and 'color' not in item:
            item['color'] = item.pop('fill')
        if item.get('kind') == 'text' and item.get('description', 1) is None:
            del item['description']
    rows = result['objects']
    if rows and all(isinstance(o, dict) and isinstance(o.get('id'), str) for o in rows):
        ids = list(dict.fromkeys(o['id'] for o in rows))
        result['layer_order'] = normalize_layer_order(ids, result.get('layer_order'))
    return result


def normalize_layer_order(ids, order):
    """Keep the lowest occurrence; insert missing IDs beside their object-list neighbours."""
    known, seen, result = set(ids), set(), []
    for oid in order if isinstance(order, list) else []:
        if isinstance(oid, str) and oid in known and oid not in seen:
            result.append(oid)
            seen.add(oid)
    for i, oid in enumerate(ids):
        if oid in seen:
            continue
        # Earlier object IDs have already been inserted. Anchor after the
        # predecessor, or at the bottom for the first object. Never sort survivors.
        position = result.index(ids[i - 1]) + 1 if i else 0
        result.insert(position, oid)
        seen.add(oid)
    return result


def problems(value, size, mode):
    if isinstance(value, dict) and 'background' in value:
        from collage.workflow.inventory import to_internal_inventory, wire_problems
        issues = wire_problems(value, size, mode)
        if issues:
            return issues
        try:
            return problems(to_internal_inventory(value, size, mode), size, mode)
        except (ValueError, ValidationError) as error:
            return [{'path': '$', 'message': str(error)[:1500]}]
    value = normalize(value)
    issues = []
    for e in Draft202012Validator(legacy_schema(size, mode)).iter_errors(value):
        path = list(e.absolute_path)
        issue = {'path': '.'.join(map(str, path)) or '$', 'message': e.message[:400]}
        if len(path) >= 2 and path[0] == 'objects' and isinstance(path[1], int):
            item = value['objects'][path[1]]
            if isinstance(item, dict) and isinstance(item.get('id'), str):
                issue['id'] = item['id']
                if e.validator == 'required':
                    issue['fields'] = [k for k in e.validator_value if k not in e.instance]
                elif len(path) >= 3 and e.validator != 'additionalProperties':
                    issue['fields'] = [path[2]]
        issues.append(issue)
    if not isinstance(value, dict) or not isinstance(value.get('objects'), list):
        return issues[:24]
    objects = value['objects']
    if not all(isinstance(o, dict) and isinstance(o.get('id'), str) for o in objects):
        return issues[:24]
    by_id = {o['id']: o for o in objects}
    indices = {o['id']: i for i, o in enumerate(objects)}
    if len(by_id) != len(objects):
        issues.append({'path': 'objects', 'message': 'Duplicate object IDs'})
    order = value.get('layer_order')
    backgrounds = [o for o in objects if o.get('kind') == 'background']
    if len(backgrounds) != 1:
        issues.append({'path': 'objects', 'message': 'Exactly one kind=background is required; reassess canvas source and object ownership'})
    elif isinstance(order, list) and order and order[0] != backgrounds[0]['id']:
        issues.append({'path': 'layer_order', 'message': 'The unique background must be first in layer_order'})
    if isinstance(order, list) and all(isinstance(oid, str) for oid in order):
        if len(order) != len(objects) or set(order) != set(by_id):
            issues.append({'path': 'layer_order',
                           'message': 'layer_order must contain each object ID exactly once; '
                                      f'missing={sorted(set(by_id)-set(order))}, '
                                      f'unknown={sorted(set(order)-set(by_id))}'})
    from photo_links import inventory_issues
    for issue in inventory_issues(objects, order if isinstance(order, list) else []):
        issues.append({'path': 'objects.' + str(indices[issue['id']]), **issue})
    from collage.frame_links import frame_issues
    for issue in frame_issues(objects):
        issues.append({'path': 'objects.' + str(indices[issue['id']]), **issue})
    for i, item in enumerate(objects):
        def fail(message, fields=()):
            issues.append({'path': f'objects.{i}', 'id': item['id'], 'message': message,
                           'fields': list(fields)})
        box = item.get('bbox')
        if (isinstance(box, list) and len(box) == 4
                and all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in box)):
            l, t, r, b = box; w, h = [1000, 1000] if mode == 'normalized_1000' else size
            if not (0 <= l < r <= w and 0 <= t < b <= h):
                fail('bbox must be nonempty and inside the reference', ['bbox'])
        kind = item.get('kind')
        if kind != 'frame' and 'photo_id' in item:
            fail('photo_id belongs to frame')
        if 'parent_id' in item:
            owner = item['parent_id']
            if not isinstance(owner, str) or owner not in by_id or owner == item['id']:
                fail('parent_id must reference another existing object', ['parent_id'])
            else:
                visited = {item['id']}
                while isinstance(owner, str) and owner in by_id:
                    if owner in visited:
                        fail('parent_id must not form a cycle', ['parent_id']); break
                    visited.add(owner); owner = by_id[owner].get('parent_id')
        if kind == 'text':
            if 'text' not in item or (isinstance(item['text'], str) and not item['text'].strip()):
                fail('independent text requires readable text or null', ['text'])
        elif 'text' in item or any(key in item for key in TEXT_STYLE_FIELDS):
            fail('text/typography are only for independent text; describe fixed inscriptions on their carrier')
        if 'color' in item and kind not in {'text', 'background'}:
            fail('extracted objects do not need colour or drawing parameters')
    return issues[:24]


def compile_inventory(value, size, mode, run=None):
    if isinstance(value, dict) and 'background' in value:
        from collage.workflow.inventory import to_internal_inventory
        try:
            value = to_internal_inventory(value, size, mode)
        except ValidationError as error:
            raise ValueError(error.message) from error
    value = normalize(value)
    issues = problems(value, size, mode)
    if issues:
        raise ValueError('; '.join(i['path'] + ': ' + i['message'] for i in issues))
    objects, units = [], []
    for source in value['objects']:
        item = {k: deepcopy(source[k]) for k in ('id', 'kind', 'bbox', 'description', 'photo_id', 'parent_id', 'linked_to', 'background_type') if k in source}
        if source['kind'] == 'text' and 'description' not in item:
            item['description'] = (source['text'] or '无法辨认的独立文字')[:400]
        if mode == 'normalized_1000':
            item['bbox'] = [round(v * size[i % 2] / 1000) for i, v in enumerate(item['bbox'])]
        item['label'] = item['description'][:30]
        kind = item['kind']
        item['method'] = 'extract' if kind in {'overlay', 'frame'} else 'local'
        if kind == 'background':
            item['bbox'] = [0, 0, *size]
            item['rotation'] = 0
            item['method'] = 'extract' if source['background_type'] == 'reference_clean' else 'local'
            if source.get('color'):
                item['style'] = {'fill': source['color']}
        if kind == 'text':
            item['text'] = source['text']
            item['text_status'] = 'known' if source['text'] else 'unreadable'
            item['style'] = {k: deepcopy(source[v]) for k, v in
                             [('fill', 'color'), *((key, key) for key in TEXT_STYLE_FIELDS)] if v in source}
            units.append({'id': item['id'], 'bbox': deepcopy(item['bbox']), 'content': item['text'],
                          'treatment': 'plain_text', 'has_backing': False, 'decorative_lettering': False,
                          'edit_requested': False, 'text_id': item['id'], 'carrier_id': None,
                          'reason': 'Derived from the independent text object; no second VLM inventory.'})
        objects.append(item)
    result = {'analysis': {'schema_version': 'collage-analysis-v1', 'reference_size': list(size),
                          'objects': objects, 'layer_order': deepcopy(value['layer_order'])}, 'text_units': units}
    Draft202012Validator(contracts.REFERENCE).validate(result)
    if run is not None:
        contracts.validate_reference(result, run, geometry_warnings=[])
    return result


def repair_plan(original, size, mode):
    """Permit only fields identified by validation, never a whole-object rewrite."""
    current_format = isinstance(original, dict) and 'background' in original
    original = normalize(original)
    if not isinstance(original, dict) or not isinstance(original.get('objects'), list):
        return None
    rows = original['objects']
    if not rows or not all(isinstance(o, dict) and isinstance(o.get('id'), str) for o in rows):
        return None
    if len({o['id'] for o in rows}) != len(rows):
        return None
    issues = problems(original, size, mode)
    fields = {}
    allowed = {'text', 'description', 'bbox', 'photo_id', 'parent_id', 'color', *TEXT_STYLE_FIELDS}
    for issue in issues:
        oid = issue.get('id'); names = issue.get('fields', [])
        if not oid or not names or set(names) - allowed:
            return None
        fields.setdefault(oid, set()).update(names)
    if not fields:
        return None
    definitions = (schema if current_format else legacy_schema)(size, mode)['properties']['objects']['items']['properties']
    updates = {}
    for oid, names in fields.items():
        if any(name not in definitions for name in names):
            return None
        specifications = {k: deepcopy(definitions[k]) for k in sorted(names)}
        if 'parent_id' in specifications:
            specifications['parent_id'] = {'anyOf': [specifications['parent_id'], {'type': 'null'}]}
        updates[oid] = contracts.obj(specifications)
    return {'schema': contracts.obj({'updates': contracts.obj(updates)}),
            'fields': {k: sorted(v) for k, v in fields.items()}, 'issues': issues}


def normalize_repair(value):
    return unwrap_object(value, {'updates'})


def merge_repair(original, patch, plan):
    Draft202012Validator(plan['schema']).validate(patch)
    result = deepcopy(original)
    by_id = {o['id']: o for o in result['objects']}
    for oid, changes in patch.get('updates', {}).items():
        by_id[oid].update(deepcopy(changes))
    return normalize(result)


