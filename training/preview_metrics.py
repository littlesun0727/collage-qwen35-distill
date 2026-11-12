"""GT snapshots, fixed-cohort scoring and offline six-metric report exports."""
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps
from response_parsing import PARSING_POLICY, unwrap_response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.focused import REFERENCE_URL, evaluate_focused, overlap_pairs, summarize_focused
from eval.schema import parse_structure, strict_loads, validate_occlusions


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def image_path(record, base):
    images = record.get('images', [])
    if len(images) != 1:
        raise ValueError('Expected exactly one image in GT/input record')
    value = images[0]
    if isinstance(value, dict) and value.get('bytes') is None:
        value = value.get('path')
    if not isinstance(value, str) or not value:
        raise ValueError('Expected a local image path')
    return (Path(base) / value).resolve()


def pixel_digest(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def make_gt(record, base):
    path = image_path(record, base)
    raw = record.get('structure')
    if raw is None:
        messages = record.get('messages', [])
        if not messages or messages[-1].get('role') != 'assistant':
            raise ValueError(f'GT has no assistant answer/structure: {path}')
        raw = messages[-1]['content']
    raw = strict_loads(raw) if isinstance(raw, str) else raw
    gt = parse_structure(raw)
    if not gt.valid:
        raise ValueError(f'Invalid GT {path}: ' + '; '.join(gt.errors))
    # Preview and SFT contract both use normalized_1000, never guess units.
    if raw.get('coordinate_mode', 'normalized_1000') != 'normalized_1000':
        raise ValueError(f'Preview GT must use normalized_1000 coordinates: {path}')
    supplied_pairs = 'occlusion_pairs' in record
    pairs = record['occlusion_pairs'] if supplied_pairs else overlap_pairs(gt)
    validate_occlusions(pairs, gt)
    digest = record.get('image_digest')
    if digest is None:
        with Image.open(path) as im:
            digest = pixel_digest(ImageOps.exif_transpose(im).convert('RGB'))
    return {'id': str(record.get('id', path.name)), 'images': [str(path)],
            'image_digest': digest, 'structure': raw, 'occlusion_pairs': pairs,
            'occlusion_policy': record.get('occlusion_policy',
                'supplied_pairs' if supplied_pairs else 'gt_bbox_positive_area_and_layer_order_v1')}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_jsonl(path, records):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records), encoding='utf-8')
    tmp.replace(path)


def save_ground_truth(records, run):
    """Call before removing assistant messages; this file is never model input."""
    targets = [make_gt(record, run) for record in records]
    write_jsonl(Path(run) / 'ground_truth.jsonl', targets)


def load_ground_truth(run, prediction_rows, explicit=None, cache=None):
    run = Path(run)
    saved = run / 'ground_truth.jsonl'
    source = Path(explicit).resolve() if explicit else saved
    strict = bool(explicit) or saved.exists()
    if not source.exists() and not explicit:
        command = run / 'command.json'
        config = json.loads(command.read_text(encoding='utf-8')) if command.exists() else {}
        data = config.get('source', {}).get('local_dir')
        split = config.get('split')
        if not data or split not in ('train', 'validation', 'test'):
            return [], '未找到 GT；旧结果可用 --gt /数据目录/test.jsonl 补建。'
        source = Path(data) / f'{split}.jsonl'
    if not source.is_file():
        if explicit:
            raise ValueError(f'GT JSONL not found: {source}')
        return [], f'GT 文件不可访问：{source}；请用 --gt 指定本地 JSONL。'
    inputs = run / 'input.jsonl'
    signature = (str(source), source.stat().st_mtime_ns, source.stat().st_size,
                 inputs.stat().st_mtime_ns if inputs.exists() else None)
    key = ('ground_truth', str(run), signature)
    if cache is not None and key in cache:
        return cache[key]
    candidates = read_jsonl(source)
    indexed = {}
    for record in candidates:
        path = image_path(record, source.parent)
        if path in indexed:
            raise ValueError(f'Duplicate GT image path: {path}')
        indexed[path] = record
    if inputs.exists():
        roster = [image_path(r, run) for r in read_jsonl(inputs)]
    elif strict:
        roster = list(indexed)
    else:
        roster = [row['source'].resolve() for row in prediction_rows]
    if len(set(roster)) != len(roster):
        raise ValueError('Duplicate images in fixed evaluation input')
    missing = [p for p in roster if p not in indexed]
    if missing:
        message = f'{len(missing)} 张输入未匹配到 GT（按完整路径匹配，不按文件名猜测）：{missing[0]}'
        if strict:
            raise ValueError(message)
        return [], message + '；请用 --gt 指定对应数据。'
    targets = [make_gt(indexed[path], source.parent) for path in roster]
    if len({r['id'] for r in targets}) != len(targets):
        raise ValueError('Duplicate GT sample IDs')
    if targets and not saved.exists():
        write_jsonl(saved, targets)
    answer = (targets, str(source))
    if cache is not None:
        cache[key] = answer
    return answer


def score_rows(targets, predictions, *, cache=None):
    """Use exactly the current run's GT for both columns; missing results stay in."""
    by_path, by_digest = {}, {}
    for row in predictions:
        path = row['source'].resolve()
        if path in by_path:
            raise ValueError(f'Duplicate prediction image: {path}')
        by_path[path] = row
        by_digest.setdefault(row['digest'], []).append(row)
    samples = []
    for gt in targets:
        path = Path(gt['images'][0])
        row = by_path.get(path)
        if row is None:
            matches = by_digest.get(gt['image_digest'], [])
            if len(matches) > 1:
                raise ValueError(f'Ambiguous identical-image predictions: {path}')
            row = matches[0] if matches else None
        if row is not None and row['digest'] != gt['image_digest']:
            raise ValueError(f'Image pixels changed since GT snapshot: {path}')
        raw = row['response'] if row else None
        present = row is not None and not row.get('missing', False)
        key = ('six_metrics', PARSING_POLICY, json.dumps(gt, sort_keys=True, ensure_ascii=False),
               json.dumps(raw, sort_keys=True, ensure_ascii=False), present)
        if cache is not None and key in cache:
            sample = cache[key]
        else:
            text, wrappers = unwrap_response(raw) if isinstance(raw, str) else (raw, [])
            raw_structure = parse_structure(raw, present=present)
            sample = evaluate_focused(gt['id'], parse_structure(gt['structure']),
                                      parse_structure(text, present=present), gt['occlusion_pairs'])
            sample['parsing'] = {'policy': PARSING_POLICY, 'removed_wrappers': wrappers,
                                 'raw_structure_valid': raw_structure.valid,
                                 'raw_errors': raw_structure.errors}
            if cache is not None:
                cache[key] = sample
        samples.append(sample)
    summary = summarize_focused(samples)
    summary['parsing_policy'] = PARSING_POLICY
    summary['diagnostics']['wrapper_images'] = sum(bool(s['parsing']['removed_wrappers']) for s in samples)
    summary['diagnostics']['raw_structure_valid_images'] = sum(s['parsing']['raw_structure_valid'] for s in samples)
    return summary, samples


FIELDS = [
    ('photo', 'recall', 'photo 召回率', 'percent'),
    ('photo', 'matched_mean_iou', 'photo 已找到框 IoU', 'iou'),
    ('fixed', 'recall', '固定元素召回率', 'percent'),
    ('fixed', 'matched_mean_iou', '固定元素已找到框 IoU', 'iou'),
    ('layer_order', 'rate', '重叠元素顺序一致率', 'percent'),
    ('frame_links', 'connected_rate', 'frame 连上率', 'percent'),
    ('frame_links', 'correct_rate', 'frame 全部连对率', 'percent'),
    ('linked_to', 'rate', 'linked_to 正确率', 'percent'),
]


def fmt(value, kind='percent'):
    return 'N/A' if value is None else (f'{value:.3f}' if kind == 'iou' else f'{value * 100:.2f}%')


def metric_table(columns, *, per_sample=False):
    """Render six rows with identical definitions in overview and sample cards."""
    def get(data, group, key, kind='percent'):
        value = data[group][key]
        number = value if per_sample else value['mean']
        display = fmt(number, kind)
        if number is None:
            display = '无适用 GT' if data[group]['gt_count'] == 0 else '无匹配框'
        if per_sample:
            return display
        return f'{display} <small>({value["eligible_images"]}/{value["total_images"]} 张)</small>'

    def values(data):
        valid = float(data['structure_valid']) if per_sample else data['output_usability']['first_pass_core_valid_rate']
        detail = ''
        if per_sample and data.get('errors'):
            detail = '<br><small>' + html.escape('；'.join(data['errors'])) + '</small>'
        return [f'首次 {fmt(valid)}{detail}<br><small>复检：未运行</small>',
                '召回 ' + get(data, 'photo', 'recall') + '<br>已找到框 IoU ' + get(data, 'photo', 'matched_mean_iou', 'iou'),
                '召回 ' + get(data, 'fixed', 'recall') + '<br>已找到框 IoU ' + get(data, 'fixed', 'matched_mean_iou', 'iou'),
                get(data, 'layer_order', 'rate'),
                '连上 ' + get(data, 'frame_links', 'connected_rate') + '<br>全部连对 ' + get(data, 'frame_links', 'correct_rate'),
                get(data, 'linked_to', 'rate')]

    labels = ['1. JSON 结构通过率', '2. photo 找到多少、框多准', '3. overlay/text/frame 找到多少、框多准',
              '4. 重叠元素顺序一致率', '5. frame 是否连上、是否全部连对', '6. linked_to 出框是否识别正确']
    cells = [values(data) for _, data in columns]
    first = columns[0][1]
    total = 1 if per_sample else first['output_usability']['total_images']
    scopes = [f'{total} 张']
    for group, field, unit in [('photo', 'recall', '个元素'), ('fixed', 'recall', '个元素'),
                               ('layer_order', 'rate', '对关系'), ('frame_links', 'connected_rate', '个 frame'),
                               ('linked_to', 'rate', '个主体')]:
        entry = first[group]
        n = (int(entry['gt_count'] > 0) if per_sample else entry[field]['eligible_images'])
        scopes.append(f'{n} 张 · {entry["gt_count"]} {unit}')
    table = '<table class="metrics"><thead><tr><th>关注项</th><th>固定 GT 范围</th>'
    table += ''.join(f'<th>{html.escape(label)}</th>' for label, _ in columns) + '</tr></thead><tbody>'
    for i, label in enumerate(labels):
        table += f'<tr><td>{label}</td><td>{scopes[i]}</td>' + ''.join(f'<td>{c[i]}</td>' for c in cells) + '</tr>'
    return table + '</tbody></table>'


def export_metrics(run, targets, evaluations, source, *, running=False):
    out = Path(run) / 'metrics'
    out.mkdir(exist_ok=True)
    cohort = [{k: t[k] for k in ('id', 'image_digest', 'structure', 'occlusion_pairs')} for t in targets]
    digest = hashlib.sha256(json.dumps(cohort, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report = {'protocol': {'version': 'collage-structure-eval-v3-preview', 'reference_url': REFERENCE_URL,
                          'iou_threshold': .5, 'aggregation': 'per-image mean on fixed GT-positive cohorts',
                          'prediction_stage': 'raw_first', 'repair_performed': False,
                          'response_parsing': PARSING_POLICY,
                          'gt_sha256': digest, 'gt_source': source, 'running': running,
                          'order_policies': sorted({t['occlusion_policy'] for t in targets})},
              'models': {name: summary for name, (summary, _) in evaluations.items()}}
    write_json(out / 'summary.json', report)
    write_jsonl(out / 'gt.jsonl', targets)
    detailed = [{'id': t['id'], 'image': t['images'][0],
                 'models': {name: samples[i] for name, (_, samples) in evaluations.items()}}
                for i, t in enumerate(targets)]
    write_jsonl(out / 'per_sample.jsonl', detailed)
    with (out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'metric', 'value_0_to_1', 'display', 'eligible_images', 'total_images', 'gt_count'])
        for name, (summary, _) in evaluations.items():
            u = summary['output_usability']
            writer.writerow([name, '首次 JSON 结构通过率', u['first_pass_core_valid_rate'], fmt(u['first_pass_core_valid_rate']), len(targets), len(targets), ''])
            writer.writerow([name, '复检后 JSON 结构通过率（未运行）', '', 'N/A', 0, len(targets), ''])
            for group, key, label, kind in FIELDS:
                m = summary[group][key]
                display = fmt(m['mean'], kind) if m['mean'] is not None else (
                    '无适用 GT' if summary[group]['gt_count'] == 0 else '无匹配框')
                writer.writerow([name, label, m['mean'], display, m['eligible_images'], m['total_images'], summary[group]['gt_count']])
    with (out / 'per_sample.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'image', 'model', 'prediction_present', 'structure_valid'] + [f[2] for f in FIELDS])
        for name, (_, samples) in evaluations.items():
            for target, sample in zip(targets, samples):
                writer.writerow([sample['id'], target['images'][0], name, sample['present'], sample['structure_valid']] +
                                [sample[g][k] for g, k, _, _ in FIELDS])
    return report
