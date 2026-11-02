"""阶段 4：照片槽 + 客户素材 -> 选材与裁切。一次请求，失败停止制作。"""
from collage.object_roles import is_customer_photo, is_photo_background

from copy import deepcopy
from jsonschema import Draft202012Validator
from collage import contracts, prompts
from collage.storage import read, save
from collage.crop_quality import expand_crop


VERSION = 'photo-binding-v9-2-background'


def execute(ctx):
    ctx.progress(2, 'running', 'VLM 一次完成客户照片与铺底背景的选材、裁切')
    structure = read_structure(ctx.run, ctx.config)
    windows = read(ctx.run/'photo-windows.json') if (ctx.run/'photo-windows.json').exists() else None
    evidence = {'contract': VERSION, 'status': 'running',
                'photo_count': sum(is_customer_photo(o) for o in structure['objects']),
                'texts_policy': 'disabled',
                'automatic_retry': False, 'automatic_repair': 'local_crop_expansion'}
    save(ctx.control/'binding-check.json', evidence)
    before = ctx.client.usage()['requests']
    try:
        base = ctx.input_context()
        base['output_width'] = read(ctx.run/'input.json').get('output_width', base['reference_size'][0])
        base['catalog'] = read(ctx.run/'prepared/catalog.json')['assets']
        context = context_for(base, structure, windows)
        schema = schema_for(context, windows)
        save(ctx.control/'binding-input.json', context)
        if context['photo_slots']:
            answer = ctx.client.ask('bind', prompts.render('bind'), context,
                ctx.analysis_images()+ctx.material_images(), schema,
                lambda value: validate(value, context, structure, ctx.run, windows),
                single_request=True, reference_input_only=True)
        else:
            answer = {'photos': []}
        value = validate(answer, context, structure, ctx.run, windows)
    except Exception as exc:
        evidence.update(status='failed', error=str(exc)[:1500],
                        requests_this_execution=ctx.client.usage()['requests']-before)
        save(ctx.control/'binding-check.json', evidence)
        raise
    save(ctx.run/'bindings.json', value)
    original = {p['slot_id']: p['source_crop'] for p in answer['photos']}
    evidence['crop_expansions'] = [
        {'slot_id': p['slot_id'], 'before': original[p['slot_id']], 'after': p['source_crop']}
        for p in value['photos'] if p['source_crop'] != original[p['slot_id']]]
    # The full renderer still uses this internal file for its progress gate.
    # Standalone binding only needs structure.json, windows and customer materials.
    if (ctx.run/'analysis.json').is_file():
        ctx.command('validate')
    evidence.update(status='accepted', requests_this_execution=ctx.client.usage()['requests']-before)
    save(ctx.control/'binding-check.json', evidence)
    ctx.progress(2, 'complete', '照片绑定已校验并保存，文字绑定为空，进入制作')


def read_structure(run, config):
    """Keep inventory semantics; use finalized pixel placement when available."""
    from collage.vlm.coordinates import resolve_mode
    structure = read(run/'structure.json')
    if not isinstance(structure.get('objects'), list):
        raise ValueError('Photo binding requires structure.json with an objects array')
    stage = config.for_stage('analyze')
    mode = structure.get('coordinate_mode', resolve_mode(stage.model, stage.coordinate_mode))
    if mode not in {'pixel', 'normalized_1000'}:
        raise ValueError('Unknown structure coordinate mode: ' + str(mode))
    if 'background' in structure:
        from collage.workflow.inventory import to_internal_inventory
        structure = {**structure, **to_internal_inventory(
            {key: structure[key] for key in ('background', 'objects', 'layer_order')},
            read(run/'input.json')['reference_size'], mode)}
    if mode == 'normalized_1000':
        size = read(run/'input.json')['reference_size']
        for photo in structure['objects']:
            if is_customer_photo(photo):
                photo['bbox'] = [v * size[i % 2] / 1000 for i, v in enumerate(photo['bbox'])]
    if (run/'analysis.json').is_file():
        current = read(run/'analysis.json')
        placed = {p['id']: p for p in current['objects'] if is_customer_photo(p)}
        expected = {p['id'] for p in structure['objects'] if is_customer_photo(p)}
        if set(placed) != expected:
            raise ValueError('Final photo geometry must match structure photo IDs')
        for photo in structure['objects']:
            if is_customer_photo(photo):
                geometry = placed[photo['id']]
                photo['bbox'] = deepcopy(geometry['bbox'])
                photo['rotation'] = geometry.get('rotation', 0)
    # Standalone binding may only have structure.json. Background placement is
    # still the canvas, regardless of the model's original box.
    for photo in structure['objects']:
        if is_photo_background(photo):
            photo.update(bbox=[0, 0, *read(run/'input.json')['reference_size']], rotation=0)
    return structure


def context_for(base, structure, windows=None):
    """Expose photo descriptions and geometry, with only one derived subject requirement."""
    slots = []
    for photo in structure['objects']:
        if not is_customer_photo(photo):
            continue
        row = {key: deepcopy(photo[key]) for key in
               ('description', 'bbox', 'rotation', 'linked_to', 'kind', 'background_type') if key in photo}
        l, t, r, b = photo['bbox']
        if r <= l or b <= t:
            raise ValueError(photo['id'] + ': photo slot must have positive area')
        row.update(slot_id=photo['id'], aspect_ratio=round((r-l)/(b-t), 6))
        slots.append(row)
    ids = [p['slot_id'] for p in slots]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate photo slot ID')
    result = {'reference_size': deepcopy(base['reference_size']),
              'instructions': base.get('instructions', ''), 'photo_slots': slots,
              'catalog': [{key: deepcopy(asset[key]) for key in ('id', 'size')}
                          for asset in base['catalog']]}
    # Explicit pixel-quality fields survive normalized reference-coordinate mode.
    seen_sources = {}
    for asset, original in zip(result['catalog'], base['catalog']):
        asset['source_size_px'] = deepcopy(asset['size'])
        identity = original.get('sha256', asset['id'])
        if identity in seen_sources:
            asset['duplicate_of'] = seen_sources[identity]
        else:
            seen_sources[identity] = asset['id']
    if not slots:
        return result
    if windows is None:
        raise ValueError('Photo binding requires a completed photo window plan')
    from photo_windows import validate_plan
    validate_plan(windows)
    specs = [spec for spec in windows['photos'] if spec['photo_id'] in ids]
    window_ids = [p['photo_id'] for p in specs]
    if len(window_ids) != len(set(window_ids)) or set(window_ids) != set(ids):
        raise ValueError('Photo window plan must cover each photo slot exactly once')
    by_id = {p['slot_id']: p for p in slots}
    for spec in specs:
        slot = by_id[spec['photo_id']]
        linked = spec.get('linked_to')
        if slot.get('linked_to') and linked != slot['linked_to']:
            raise ValueError('Photo window plan must retain the structure linked_to')
        if linked:
            slot['linked_to'] = linked
        subject = spec['subject']
        slot['requires_segmentable_subject'] = (not subject['preserve_background']
                                                 or subject['stroke']['enabled'])
        scale = base.get('output_width', base['reference_size'][0]) / base['reference_size'][0]
        l,t,r,b = slot['bbox']
        slot['resolution_policy'] = {
            'output_size_px': [max(1,round(r*scale)-round(l*scale)), max(1,round(b*scale)-round(t*scale))],
            'fit': 'cover' if subject['preserve_background'] else 'contain',
        }
    for slot in slots:
        linked = slot.get('linked_to')
        if not linked:
            continue
        if linked not in by_id or linked == slot['slot_id'] or by_id[linked].get('linked_to'):
            raise ValueError('linked_to must name an independent photo slot')
        base_slot = by_id[linked]
        base_slot['requires_segmentable_subject'] |= slot['requires_segmentable_subject']
        # A linked pair is scaled from its segmented foreground; don't impose a
        # separate cover-to-base mapping that the renderer no longer uses.
        base_slot['resolution_policy'] = deepcopy(slot['resolution_policy'])
    return result


def schema_for(context, windows=None):
    item = deepcopy(contracts.BINDING_PLAN['properties']['photos']['items'])
    slots = [p['slot_id'] for p in context['photo_slots']]
    assets = [a['id'] for a in context['catalog']]
    if slots and not assets:
        raise ValueError('Photo binding requires customer assets')
    groups = sum(not p.get('linked_to') for p in context['photo_slots'])
    if groups > sum(not a.get('duplicate_of') for a in context['catalog']):
        raise ValueError('Not enough distinct customer photos for independent slots; reuse is forbidden')
    for field, values in [('slot_id', slots), ('asset_id', assets)]:
        if values:
            item['properties'][field]['enum'] = values
    item['properties']['reason']['minLength'] = 1
    # Keep rendering details local. Only the required crop reaches the output contract.
    specs = {p['photo_id']: p for p in (windows or {}).get('photos', [])}
    full_crop = [p['slot_id'] for p in context['photo_slots']
                 if p['slot_id'] in specs and not p.get('linked_to')
                 and not specs[p['slot_id']]['subject']['preserve_background']]
    if full_crop:
        item['allOf'] = [{'if': {'properties': {'slot_id': {'enum': full_crop}},
                                'required': ['slot_id']},
                          'then': {'properties': {'source_crop': {'const': [0, 0, 1, 1]}}}}]
    return contracts.obj({'photos': {'type': 'array', 'items': item,
                                     'minItems': len(slots), 'maxItems': len(slots)}})


def validate(answer, context, structure, run, windows=None):
    """Validate selection, then expand undersized crops locally without model retries."""
    if windows is None:
        windows = read(run/'photo-windows.json')
    Draft202012Validator(schema_for(context, windows)).validate(answer)
    by_id = {p['slot_id']: p for p in answer['photos']}
    if len(by_id) != len(answer['photos']) or set(by_id) != {p['slot_id'] for p in context['photo_slots']}:
        raise ValueError('photos must cover every photo slot exactly once')
    for slot in context['photo_slots']:
        oid = slot['slot_id']; row = by_id[oid]
        if not row['reason'].strip():
            raise ValueError(oid + ': reason must describe visible evidence and crop choice')
        linked = slot.get('linked_to')
        if linked:
            for field in ('asset_id', 'source_crop', 'crop_center', 'mirror_x'):
                if row[field] != by_id[linked][field]:
                    raise ValueError(f'{oid}: {field} must match linked base {linked}')
    catalog = {a['id']: a for a in read(run/'prepared/catalog.json')['assets']}
    used = {}
    for slot in context['photo_slots']:
        if slot.get('linked_to'):
            continue
        row = by_id[slot['slot_id']]; asset = catalog[row['asset_id']]
        identity = asset.get('sha256', asset['id'])
        if identity in used:
            raise ValueError(f"{slot['slot_id']}: duplicate customer photo already used by {used[identity]}; only linked_to pairs may share")
        used[identity] = slot['slot_id']
    value = {'schema_version': 'collage-bindings-v1', 'photos': deepcopy(answer['photos']), 'texts': []}
    contracts.validate_bindings(value, structure, run)
    repaired = {p['slot_id']: p for p in value['photos']}
    for slot in context['photo_slots']:
        if slot.get('linked_to'):
            continue
        row = repaired[slot['slot_id']]
        policy = slot.get('resolution_policy')
        if policy:
            row['source_crop'] = expand_crop(row['source_crop'], catalog[row['asset_id']]['size'],
                policy['output_size_px'], contain=policy['fit']=='contain')
    for slot in context['photo_slots']:
        if slot.get('linked_to'):
            repaired[slot['slot_id']]['source_crop'] = deepcopy(repaired[slot['linked_to']]['source_crop'])
    contracts.validate_bindings(value, structure, run)
    return value


