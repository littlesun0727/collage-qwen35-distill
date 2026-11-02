"""Sequential top-down extraction and self-consistent rendered dataset export."""
import copy
import hashlib
import shutil
from decimal import Decimal
from pathlib import Path
from PIL import Image
from . import service as reveal
from .geometry import clean_layer


def plan_batches(analysis, batch_size=20):
    if not 1 <= batch_size <= 20:
        raise ValueError('Batch size must be between 1 and 20')
    # layer_order is bottom-to-top; remove upper layers before underlying ones.
    top_down = [i for i in reversed(analysis['layer_order']) if i != 'layer_0']
    return [top_down[n:n+batch_size] for n in range(0, len(top_down), batch_size)]


def scaled_analysis(source, ids, size):
    sw, sh = source['reference_size']
    w, h = size
    objects = []
    for original in source['objects']:
        if original['id'] not in ids:
            continue
        item = copy.deepcopy(original)
        item['bbox'] = [round(v * (w/sw if n % 2 == 0 else h/sh))
                        for n, v in enumerate(item['bbox'])]
        # Only the request box needs the service minimum; keep prediction boxes
        # unchanged in the dataset and use them to guide geometry cleanup.
        from reveal import request_object
        item['bbox'] = request_object(item, size, padding=0)['bbox']
        l, t, r, b = item['bbox']
        if not 0 <= l < r <= w or not 0 <= t < b <= h:
            raise ValueError('Invalid batch box: ' + item['id'])
        objects.append(item)
    return {'reference_size': list(size), 'objects': objects,
            'layer_order': [i for i in source['layer_order'] if i in ids]}


def extract_chain(run, key, timeout, batch_size=20):
    source = reveal.read(run/'source-analysis.json')
    groups = plan_batches(source, batch_size)
    if not groups:
        raise ValueError('Reveal requires at least one foreground object; use --analyze-only for a background-only image')
    plan = {'version': 2, 'continuation_variant': 'aug', 'batch_size': batch_size,
            'batches_top_down': groups,
            'source_analysis_sha256': hashlib.sha256((run/'source-analysis.json').read_bytes()).hexdigest()}
    plan_path = run/'chain-plan.json'
    if plan_path.exists() and reveal.read(plan_path) != plan:
        raise ValueError('Chain plan changed; use a new run folder')
    reveal.save(plan_path, plan)
    current = run/'input/reference.png'
    manifests = []
    canvas = None
    for number, ids in enumerate(groups, 1):
        batch = run/'batches'/f'{number:03d}'
        (batch/'input').mkdir(parents=True, exist_ok=True)
        target = batch/'input/reference.png'
        if target.exists():
            if target.read_bytes() != current.read_bytes():
                raise ValueError('Batch input changed; use a new run folder')
        else:
            shutil.copyfile(current, target)
        with Image.open(target) as im:
            analysis = scaled_analysis(source, ids, im.size)
        reveal.save(batch/'input/analysis.json', analysis)
        print(f'Extracting batch {number}/{len(groups)}: {len(ids)} foreground objects', flush=True)
        reveal.submit_and_fetch(batch, key, timeout)
        reveal.assemble(batch)
        manifest = reveal.read(batch/'manifest.json')
        if manifest['missing_ids'] or 'aug' not in manifest['variants']:
            raise ValueError(f'Batch {number} incomplete; inspect it before continuing')
        for variant, data in manifest['variants'].items():
            if variant != 'aug':
                continue
            size = tuple(data['size'])
            if canvas is None:
                canvas = size
            if size != canvas:
                raise ValueError('Chain canvas size changed; refusing misaligned composition')
            for ident in ['layer_0'] + ids:
                path = batch/f'layers/{variant}/{ident}.png'
                with Image.open(path) as im:
                    if im.size != canvas or (ident != 'layer_0' and im.getchannel('A').getbbox() is None):
                        raise ValueError('Empty or misaligned layer: ' + str(path))
                dest = run/f'layers/{variant}/{ident}.png'
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        manifests.append(manifest)
        # Only the final residual becomes dataset background. Never stack intermediates.
        current = batch/'layers/aug/layer_0.png'
    order = [i for i in source['layer_order'] if i != 'layer_0']
    merged = {'schema_version': 'reveal-chain-result-v1', 'reference_size': source['reference_size'],
              'layer_order': order, 'stack_order': ['layer_0']+order,
              'requested_count': len(order), 'returned_count': len(order), 'missing_ids': [],
              'total_layer_count': len(order)+1, 'batch_count': len(groups),
              'continuation_variant': 'aug', 'task_ids': [m['task_id'] for m in manifests],
              'charge_amount': str(sum((Decimal(m.get('charge_amount') or '0') for m in manifests), Decimal(0))),
              'generation_seconds': sum(float(m.get('generation_seconds') or 0) for m in manifests),
              'background': {'id': 'layer_0', 'source': 'final_batch_residual'}, 'variants': {}}
    by_id = {o['id']: o for o in source['objects']}
    geometry = {}
    for variant in ('aug',):
        composite = Image.open(run/f'layers/{variant}/layer_0.png').convert('RGBA')
        entries = []
        for ident in order:
            file = f'layers/{variant}/{ident}.png'
            with Image.open(run/file) as im:
                im = im.convert('RGBA')
                original = run/f'layers-original/{variant}/{ident}.png'
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(run/file, original)
                im, review = clean_layer(im, by_id[ident]['bbox'], source['reference_size'])
                review['original_layer'] = original.relative_to(run).as_posix()
                geometry[f'{variant}/{ident}'] = review
                im.save(run/file)
                composite = Image.alpha_composite(composite, im)
                entries.append({'id': ident, 'kind': by_id[ident]['kind'],
                                'description': by_id[ident]['description'],
                                'visible_bbox': list(im.getchannel('A').getbbox()), 'file': file})
        composite.save(run/f'reconstructed-{variant}.png')
        composite.save(run/f'reference-{variant}.png')
        merged['variants'][variant] = {'size': list(canvas), 'layers': entries,
                                      'background_file': f'layers/{variant}/layer_0.png',
                                      'mean_absolute_rgb_error': None}
    reveal.save(run/'manifest.json', merged)
    reveal.save(run/'geometry-review.json', geometry)
    reveal.create_html(run, merged)
    page = run/'index.html'
    text = page.read_text(encoding='utf-8')
    text = text.replace('原图</h3>', '输入原图（仅供追溯）</h3>').replace('图层叠合</h3>', '新 reference · 素材叠合</h3>')
    start = 'RGB 平均绝对误差 ${data.mean_absolute_rgb_error} / 255（仅用于比较复原，不代表分层质量）。'
    text = text.replace(start, '新 reference 由当前素材本地渲染生成。')
    text = text.replace('<div class="controls">', '<p><a href="reference-base.png">新 reference（base）</a> · <a href="reference-aug.png">新 reference（aug）</a> · <a href="analysis.json">最终标注</a> · <a href="comparison/index.html">逐层对比</a></p><div class="controls">', 1)
    page.write_text(text, encoding='utf-8')


def finalize_dataset(run):
    """Write pixel refined geometry without replacing prediction analysis."""
    from .artifacts import pixel_document
    from .export import write_svg
    dataset = reveal.read(run/'dataset.json')
    source = reveal.read(run/'source-analysis.json')
    manifest = reveal.read(run/'manifest.json')
    shutil.copyfile(run/'reference-aug.png', run/'reference.png')
    refined = copy.deepcopy(source)
    refined['reference_size'] = manifest['variants']['aug']['size']
    records = {r['id']: r for r in dataset['records']}
    for item in refined['objects']:
        record = records.get(item['id'])
        if record is None:
            raise ValueError('Missing extracted asset: ' + item['id'])
        item['bbox'] = record['crop_bbox']
        if 'window_bbox' in item:
            sw, sh = source['reference_size']
            w, h = refined['reference_size']
            item['window_bbox'] = [round(v * (w/sw if i % 2 == 0 else h/sh))
                                   for i, v in enumerate(item['window_bbox'])]
    refined = pixel_document(refined)
    reveal.save(run/'analysis-refined.json', refined)
    write_svg(run, refined, records)
    # Keep historical exports for traceability, outside the published layers.
    for name in ('layers/base', 'assets/base', 'analysis-base.json', 'analysis-aug.json',
                 'reference-base.png', 'reconstructed-base.png'):
        old = (run/name).resolve()
        archive = (run/'legacy-export'/name).resolve()
        old.relative_to(run.resolve())
        archive.relative_to(run.resolve())
        if old.exists() and not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(archive))
    reveal.create_html(run, manifest)
    dataset.update(reference='reference.png', source_reference='original.png',
                   analysis='analysis-refined.json', source_analysis='analysis.json',
                   canonical_variant='aug', references={'aug': 'reference.png'})
    for record in dataset['records']:
        record.update(reference='reference.png', analysis='analysis-refined.json')
    reveal.save(run/'dataset.json', dataset)
    import json
    (run/'dataset.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n'
                                            for r in dataset['records']), encoding='utf-8')
