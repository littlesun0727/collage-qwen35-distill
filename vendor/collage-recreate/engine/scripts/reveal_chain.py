"""Full-image Reveal batches chained through layer 0; cleanup uses analysis IDs only."""
from copy import deepcopy
import math
from pathlib import Path
import time

from PIL import Image, ImageChops

from common import cache_directory, event, fingerprint, read, save, sha
from reveal_plan import CHAIN_MAX_REQUESTS, HARD_LIMIT, plan, preview


VERSION = 'reveal-chain-v2'
# Extraction requests have not changed; retain reusable service results.
CACHE_VERSION = 'reveal-chain-v1'


def backgrounds(scene):
    return [o for o in scene['objects']
            if o['kind'] == 'background' and o.get('background_type') == 'reference_clean']


def covered_ids(scene, processed):
    """Only explicit pixel ownership extends coverage; parent_id and box overlap do not."""
    covered = set(processed)
    for obj in scene['objects']:
        if obj.get('embedded_in') in processed:
            covered.add(obj['id'])
    return covered


def cleanup_objects(scene, processed, padding=.1, padding_mode='capped'):
    from reveal import request_object
    covered = covered_ids(scene, processed)
    bg = backgrounds(scene)
    size = scene['reference_size']
    result = []
    for obj in scene['objects']:
        if obj['id'] in covered:
            continue
        if obj['kind'] not in ['photo', 'text', 'overlay', 'frame']:
            continue
        box = visible_bbox(obj, size)
        if not any(max(box[0], b['bbox'][0]) < min(box[2], b['bbox'][2]) and
                   max(box[1], b['bbox'][1]) < min(box[3], b['bbox'][3]) for b in bg):
            continue
        request = request_object({**obj, 'bbox': box}, size, padding, padding_mode)
        request.update(role='cleanup', source_id=obj['id'], original_bbox=list(obj['bbox']))
        request['reference_bbox'] = list(request['bbox'])
        result.append(request)
    return result


def visible_bbox(obj, size):
    """Account for declared rotation and rendered edges without detecting image content."""
    l, t, r, b = obj['bbox']
    style = obj.get('style', {})
    spread = style.get('stroke_width', 0)
    spec = obj.get('photo_window_spec', {})
    for stroke in (spec.get('stroke', {}), spec.get('subject_stroke', {})):
        if stroke.get('enabled') and stroke.get('placement', 'inside') != 'inside':
            spread = max(spread, min(r-l, b-t) * (stroke.get('width', 0) + stroke.get('offset', 0)))
    if spec.get('fade_direction') == 'outside':
        from window_drawing import sampling_margin
        spread = max(spread, sampling_margin((r-l, b-t), spec))
    shadow = style.get('shadow') or {}
    spread += 3 * shadow.get('blur', 0) + max(abs(v) for v in shadow.get('offset', [0, 0]))
    angle = math.radians(obj.get('rotation', 0))
    width = (r-l+2*spread)*abs(math.cos(angle)) + (b-t+2*spread)*abs(math.sin(angle))
    height = (r-l+2*spread)*abs(math.sin(angle)) + (b-t+2*spread)*abs(math.cos(angle))
    cx, cy = (l+r)/2, (t+b)/2
    return [max(0, math.floor(cx-width/2)), max(0, math.floor(cy-height/2)),
            min(size[0], math.ceil(cx+width/2)), min(size[1], math.ceil(cy+height/2))]


def input_objects(objects, reference_size, input_size):
    from reveal import request_object
    if list(reference_size) == list(input_size):
        return deepcopy(objects)
    result = []
    for obj in objects:
        box = obj['reference_bbox']
        scaled = [math.floor(box[0]*input_size[0]/reference_size[0]),
                  math.floor(box[1]*input_size[1]/reference_size[1]),
                  math.ceil(box[2]*input_size[0]/reference_size[0]),
                  math.ceil(box[3]*input_size[1]/reference_size[1])]
        bounded = request_object({'id': obj['id'], 'bbox': scaled}, input_size, 0)['bbox']
        result.append({**obj, 'bbox': bounded})
    return result


def layer_zero(task, expected_size, box_mapping=None):
    errors = []
    for name in ['layers_aug_00.png', 'layers_base_00.png']:
        path = task / name
        if not path.is_file():
            continue
        try:
            with Image.open(path) as im:
                im.load()
                if abs(im.width/im.height - expected_size[0]/expected_size[1]) > .02 * expected_size[0]/expected_size[1]:
                    from reveal_grouped import resize_mapping_matches
                    if not resize_mapping_matches(im.size, expected_size, box_mapping):
                        raise ValueError('Layer 0 aspect ratio changed without a consistent box mapping')
                    # Normalize the residual canvas before the next round uses reference coordinates.
                    # Keep the downloaded original and ordinary cache paths unchanged.
                    normalized = path.with_name(path.stem + '-normalized.png')
                    im.resize(tuple(expected_size), Image.Resampling.LANCZOS).save(normalized)
                    return normalized
            return path
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    raise ValueError('No usable layer 0: ' + ('; '.join(errors) or 'download missing'))


def acquire(run, scene, targets, config):
    from reveal import api_key, downloads_ready, fetch_batch, reserve_submission
    from reveal_grouped import restore

    run = Path(run)
    folder = run/'assets/reveal'
    root = folder/'chain'
    root.mkdir(parents=True, exist_ok=True)
    reference = Path(scene['reference']['file'])
    size = scene['reference_size']
    full = [0, 0, *size]
    limit = config.get('max_requests', CHAIN_MAX_REQUESTS)
    if type(limit) is not int or limit < 1:
        raise ValueError('Reveal max_requests must be a positive integer')
    padding, padding_mode = config.get('padding', .1), config.get('padding_mode', 'capped')
    p = plan(scene, targets, padding, 'full', padding_mode, limit)
    preview(reference, p, folder)
    bg = backgrounds(scene)
    # Compute the successful-run bound; missing material IDs may add cleanup boxes.
    planned_ids = {o.get('source_id', o['id']) for b in p['batches'] for o in b['objects']}
    cleanup_count = len(cleanup_objects(scene, planned_ids, padding, padding_mode)) if bg else 0
    planned_rounds = len(p['batches']) + math.ceil(cleanup_count/HARD_LIMIT)
    report = {'version': VERSION, 'reference': scene['reference'], 'background_ids': [o['id'] for o in bg],
              'planned_material_rounds': len(p['batches']), 'planned_cleanup_objects': cleanup_count,
              'planned_rounds': planned_rounds, 'max_requests': limit, 'rounds': [],
              'semantic_verified': False, 'status': 'running'}
    report_path = folder/'chain-report.json'
    found, processed, receipts = {}, set(), []
    current, previous = reference, None

    def persist():
        report['processed_ids'] = sorted(processed)
        report['missing_material_ids'] = sorted(t['id'] for t in targets
                                               if t.get('request', True) and t['id'] not in found)
        save(report_path, report)
        save(folder/'grouped-receipts.json', receipts)

    def execute(objects, phase, require_base):
        nonlocal current, previous
        number = len(report['rounds']) + 1
        started = time.monotonic()
        with Image.open(current) as im:
            # Chain the residual RGB pixels directly; service alpha must not block continuation.
            rgb = im.convert('RGB')
        objects = input_objects(objects, size, rgb.size)
        identity = {'version': CACHE_VERSION, 'reference_sha256': scene['reference']['sha256'],
                    'input_source_sha256': sha(current), 'phase': phase, 'objects': objects,
                    'parent_round': previous}
        task = cache_directory(root, 'round', identity, 'chain-input.json')
        ip = task/'input.png'
        if ip.exists():
            with Image.open(ip) as im:
                if im.size != rgb.size or im.convert('RGB').tobytes() != rgb.tobytes():
                    raise ValueError('Cached chain input changed')
        else:
            rgb.save(ip)
        source_task = task
        # Reuse historical first-round extraction only when its exact input and object list match.
        external = Path(config['cache']) if config.get('cache') else None
        if number == 1 and external and not (task/'request_meta.json').exists():
            for meta_file in sorted(external.glob('grouped/batch_*/request_meta.json')):
                meta = read(meta_file)
                if (meta.get('reference_sha256') == sha(ip) and meta.get('objects') == objects and
                        downloads_ready(meta_file.parent)):
                    source_task = meta_file.parent
                    break
        cache_hit = downloads_ready(source_task)
        receipt = {'phase': phase, 'round': number, 'task': str(source_task),
                   'input_file': str(ip), 'input_sha256': sha(ip), 'input_size': list(rgb.size),
                   'input_source': str(current), 'input_source_sha256': sha(current), 'parent_round': previous,
                   'requested_ids': [o.get('source_id', o['id']) for o in objects],
                   'count': len(objects), 'cache_hit': cache_hit}
        report['rounds'].append(receipt)
        receipts.append(receipt)
        persist()
        try:
            if not cache_hit:
                if not config.get('remote'):
                    raise ValueError('No complete matching chain cache; remote Reveal is disabled')
                reserve_submission(task, folder, limit)
                event(run, 'reveal_request_started', task=str(task), targets=receipt['requested_ids'], phase=phase)
                fetch_batch(task, ip, objects, api_key(config.get('key_file')),
                            config.get('timeout', 360))
            meta = read(source_task/'request_meta.json')
            if (meta['reference_sha256'] != sha(ip) or meta['objects'] != objects or
                    meta.get('model') != 'ecommerce_layer' or meta.get('version') != 'v1.1'):
                raise ValueError('Chain request metadata changed')
            q = read(source_task/'query_response.json')
            if q.get('status') != 'done':
                raise ValueError('Reveal round is not complete')
            out = q['output']
            indices = out.get('boxes_mapping_index')
            if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < len(objects) for i in indices):
                raise ValueError('Invalid Reveal mapping')
            if len(set(indices)) != len(indices):
                raise ValueError('Duplicate Reveal mapping')
            boxes = [o['bbox'] for o in objects]
            if out.get('image_boxes') not in [None, boxes, [boxes[i] for i in indices]]:
                raise ValueError('Service box echo differs')
            available, unavailable = [], []
            for j, i in enumerate(indices, 1):
                obj = objects[i]
                raw = None
                for name in [f'layers_aug_{j:02d}.png', f'layers_base_{j:02d}.png']:
                    candidate = source_task/name
                    if not candidate.is_file():
                        continue
                    try:
                        with Image.open(candidate) as im:
                            im.load()
                            if 'A' not in im.getbands():
                                raise ValueError('Reveal element lacks alpha')
                        raw = candidate
                        break
                    except (OSError, ValueError):
                        continue
                if raw is None:
                    # The service mapped removal from layer 0 even if its foreground PNG is unusable.
                    unavailable.append(obj.get('source_id', obj['id']))
                    processed.add(obj.get('source_id', obj['id']))
                    continue
                if obj['role'] == 'asset':
                    dest = task/(obj['id']+'-restored.png')
                    # The input may already be scaled; map through its canvas to the original once.
                    restore(raw, [0, 0, rgb.width, rgb.height], [rgb.width, rgb.height], dest,
                            (out.get('image_boxes', boxes), out.get('resized_image_boxes')))
                    with Image.open(dest) as im:
                        restored = im.resize(tuple(size), Image.Resampling.LANCZOS)
                    restored.save(dest)
                    found[obj['id']] = {'file': str(dest.resolve()), 'sha256': sha(dest),
                        'bbox': obj['reference_bbox'], 'original_bbox': obj['original_bbox'],
                        'padding': obj['padding'], 'padding_mode': obj.get('padding_mode', padding_mode),
                        'raw_file': str(raw.resolve()), 'raw_sha256': sha(raw), 'task': str(source_task),
                        'crop_box': full, 'input_sha256': sha(ip)}
                    available.append(obj['id'])
                    event(run, 'material_extracted', object_id=obj['id'], raw_file=str(raw),
                          processed_file=str(dest), cache_hit=cache_hit)
                processed.add(obj.get('source_id', obj['id']))
            receipt.update(available_ids=available, unavailable_layer_ids=unavailable,
                           mapped_ids=[objects[i].get('source_id', objects[i]['id']) for i in indices],
                           task_id=q.get('task_id') or read(source_task/'submit_response.json').get('task_id'))
            missing = [objects[i].get('source_id', objects[i]['id']) for i in range(len(objects)) if i not in indices]
            receipt['missing_ids'] = missing
            try:
                base = layer_zero(source_task, rgb.size,
                                  (out.get('image_boxes', boxes), out.get('resized_image_boxes')))
            except ValueError:
                if require_base:
                    raise
                base = None
            if base:
                receipt['layer_0'] = {'file': str(base.resolve()), 'sha256': sha(base)}
                current = base
            previous = fingerprint(identity)
            if missing and phase == 'background_cleanup':
                raise ValueError('Background cleanup has unmapped objects: '+', '.join(missing))
            receipt['status'] = 'partial' if missing or unavailable else 'complete'
        except Exception as exc:
            receipt.update(status='failed', error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            receipt['elapsed_seconds'] = round(time.monotonic()-started, 3)
            persist()
            event(run, 'reveal_group_finished', **receipt)

    try:
        if p['blocked']:
            raise ValueError('; '.join(b['reason'] for b in p['blocked']))
        if planned_rounds > limit:
            raise ValueError(f'Material and cleanup plan needs {planned_rounds} requests; limit is {limit}')
        save(folder/'submission-budget.json', {'limit': limit, 'planned_rounds': planned_rounds, 'version': VERSION})
        for index, batch in enumerate(p['batches']):
            objects = [o for o in batch['objects'] if o.get('source_id', o['id']) not in processed]
            if objects:
                execute(objects, 'materials', bool(bg) or index+1 < len(p['batches']))
        # A service response can omit a requested asset while returning the other layers.
        # Retry those unmapped IDs once on the residual canvas, preserving completed work.
        # Mapped layers with download problems must resume downloads, not incur another submit.
        missing = [o for batch in p['batches'] for o in batch['objects']
                   if o['role'] == 'asset' and o.get('source_id', o['id']) not in processed]
        cleanup_rounds = math.ceil(len(cleanup_objects(scene, processed, padding, padding_mode))/HARD_LIMIT) if bg else 0
        if (missing and current != reference and len(report['rounds']) + 1 + cleanup_rounds <= limit):
            retry_objects = missing[:HARD_LIMIT]
            report['material_retry_ids'] = [o.get('source_id', o['id']) for o in retry_objects]
            execute(retry_objects, 'materials_retry', bool(bg))
        cleanup = cleanup_objects(scene, processed, padding, padding_mode) if bg else []
        report['cleanup_ids'] = [o['source_id'] for o in cleanup]
        report['actual_planned_rounds'] = len(report['rounds']) + math.ceil(len(cleanup)/HARD_LIMIT)
        if report['actual_planned_rounds'] > limit:
            raise ValueError(f'Background cleanup exceeds the {limit}-request limit')
        for start in range(0, len(cleanup), HARD_LIMIT):
            execute(cleanup[start:start+HARD_LIMIT], 'background_cleanup', True)
        if bg:
            with Image.open(current) as im:
                base = im.convert('RGB').convert('RGBA').resize(tuple(size), Image.Resampling.LANCZOS)
            for obj in bg:
                # Backgrounds may cover a panel rather than the whole reference canvas.
                from reveal_assets import make_window
                mask = make_window(tuple(size), obj['bbox'], obj.get('rotation', 0))
                image = base.copy()
                image.putalpha(ImageChops.multiply(image.getchannel('A'), mask))
                dest = folder/(obj['id']+'-background-source.png')
                image.save(dest)
                found[obj['id']] = {'file': str(dest.resolve()), 'sha256': sha(dest),
                    'bbox': obj['bbox'], 'original_bbox': obj['bbox'], 'padding': 0,
                    'raw_file': str(current.resolve()), 'raw_sha256': sha(current),
                    'background_clean': True, 'semantic_verified': False}
            report['background_source'] = {'file': str(current.resolve()), 'sha256': sha(current)}
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc))
        receipts.append({'error': report['error'], 'requested_ids': [o['id'] for o in bg], 'blocked': True})
    persist()
    return found, receipts
