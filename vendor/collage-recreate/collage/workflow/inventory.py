"""Current model wire contract; compile once into the stable engine contract."""
from copy import deepcopy
from jsonschema import Draft202012Validator
from collage import contracts


def schema(size, mode, legacy=False):
    from collage.workflow.stages.analyze import legacy_schema, ID
    item = deepcopy(legacy_schema(size, mode)['properties']['objects']['items'])
    item['properties']['kind']['enum'] = ['photo', 'frame', 'overlay', 'text']
    del item['properties']['background_type']
    item['allOf'] = item['allOf'][3:]
    if not legacy:
        del item['properties']['parent_id']
        item['properties']['photo_id'] = {'type': 'array', 'items': ID,
                                          'minItems': 1, 'maxItems': 160, 'uniqueItems': True}
        item['allOf'].append({'if': {'properties': {'kind': {'const': 'frame'}}},
                             'else': {'properties': {'photo_id': False}}})
    background = {'anyOf': [
        contracts.obj({'type': {'enum': ['photographic', 'reference_clean']},
                       'id': ID, 'description': {'type': 'string', 'minLength': 1, 'maxLength': 400}}),
        contracts.obj({'type': {'const': 'solid_color'},
                       'color': {'type': 'string', 'pattern': '^#[0-9a-fA-F]{6}$'}}),
        contracts.obj({'type': {'const': 'transparent'}}),
    ]}
    return contracts.obj({'background': background,
                         'objects': {'type': 'array', 'items': item, 'maxItems': 160},
                         'layer_order': {'type': 'array', 'items': ID, 'maxItems': 160,
                                         'uniqueItems': True}})


def wire_problems(value, size, mode):
    issues = []
    for error in Draft202012Validator(schema(size, mode, legacy=True)).iter_errors(value):
        path = list(error.absolute_path)
        issue = {'path': '.'.join(map(str, path)) or '$', 'message': error.message[:400]}
        if len(path) >= 2 and path[0] == 'objects' and isinstance(path[1], int):
            row = value['objects'][path[1]]
            if isinstance(row, dict) and isinstance(row.get('id'), str):
                issue['id'] = row['id']
                if error.validator == 'required':
                    issue['fields'] = [k for k in error.validator_value if k not in error.instance]
                elif len(path) >= 3 and error.validator != 'additionalProperties':
                    issue['fields'] = [path[2]]
        issues.append(issue)
    return issues


def to_internal_inventory(value, size, mode):
    Draft202012Validator(schema(size, mode, legacy=True)).validate(value)
    rows = deepcopy(value['objects']); order = list(value['layer_order'])
    ids = [o['id'] for o in rows]
    if len(set(ids)) != len(ids) or len(order) != len(ids) or set(order) != set(ids):
        raise ValueError('layer_order must contain every foreground ID exactly once')
    bg = value['background']; ident = bg.get('id', 'bg')
    if 'id' in bg and ident in ids:
        raise ValueError('background.id must be globally unique')
    while ident in ids:
        ident += '_'
    if len(ident) > 64:
        raise ValueError('Cannot allocate canvas background ID')
    for row in rows:
        if row.get('parent_id') == ident:
            raise ValueError('parent_id cannot reference the canvas background')
        if row.get('linked_to') == ident and bg['type'] != 'photographic':
            raise ValueError('linked_to requires a photographic background')
    kind = bg['type']
    background = {'id': ident, 'kind': 'background',
                  'background_type': {'photographic': 'photo', 'transparent': 'solid_color'}.get(kind, kind),
                  'bbox': [0, 0, *([1000, 1000] if mode == 'normalized_1000' else size)],
                  'description': bg.get('description', 'Transparent canvas' if kind == 'transparent' else 'Solid canvas')}
    if kind in {'solid_color', 'transparent'}:
        background['color'] = bg.get('color', '#00000000')
    return {'objects': [background, *rows], 'layer_order': [ident, *order]}
