"""GT snapshots, fixed-cohort scoring and offline six-metric report exports."""
import csv
import hashlib
import html
import json
import re
from pathlib import Path
import sys

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.focused import REFERENCE_URL, evaluate_focused, overlap_pairs, summarize_focused
from eval.photo_effects import evaluate_effects, has_effects, summarize_effects
from eval.schema import parse_structure, strict_loads, validate_occlusions


def unwrap_response(response):
    """Remove only empty thinking and a complete outer JSON code fence."""
    text = response.strip()
    wrappers = []
    empty_think = re.match(r'^<think>\s*</think>\s*', text)
    if empty_think:
        text = text[empty_think.end():]
        wrappers.append('空 think 标签')
    fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
        wrappers.append('JSON 代码围栏')
    return text, wrappers


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
    size = record.get('image_size')
    if not size and path.is_file():
        with Image.open(path) as im:
            size = im.size[::-1] if im.getexif().get(274) in (5, 6, 7, 8) else im.size
    v3 = record.get('v3_prompt') or any('popout' in str(m.get('content','')) for m in record.get('messages',[])[:-1])
    return {'v3_prompt': bool(v3), 'image_size': size, 'id': str(record.get('id', path.name)), 'images': [str(path)],
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
    use_v3 = any(t.get('v3_prompt') or has_effects(t['structure']) for t in targets)
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
        finish = row.get('finish_reason') if row else None
        key = ('preview_metrics', use_v3, finish, json.dumps(gt, sort_keys=True, ensure_ascii=False),
               json.dumps(raw, sort_keys=True, ensure_ascii=False), present)
        if cache is not None and key in cache:
            sample = cache[key]
        else:
            payload = unwrap_response(raw)[0] if isinstance(raw, str) else raw
            target, predicted = parse_structure(gt['structure']), parse_structure(payload, present=present)
            sample = evaluate_focused(gt['id'], target, predicted, gt['occlusion_pairs'])
            if use_v3:
                sample['photo_effects'] = evaluate_effects(target, predicted, gt.get('image_size'), finish)
            if cache is not None:
                cache[key] = sample
        samples.append(sample)
    summary = summarize_focused(samples)
    if use_v3:
        summary['photo_effects'] = summarize_effects((s['photo_effects'] for s in samples), 'raw_first')
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
    """Show applicable measurements on sample cards and full coverage in the overview."""
    first = columns[0][1]
    total = 1 if per_sample else first['output_usability']['total_images']
    rows = []

    def add_row(label, scope, cells):
        if per_sample and not any(cells):
            return
        rows.append(f'<tr><td>{label}</td><td>{scope}</td>' +
                    ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')

    def score(value, kind='percent'):
        return '' if per_sample and value is None else fmt(value, kind)

    def group_cell(entry, fields):
        parts = []
        for key, prefix, kind in fields:
            value = entry[key]
            if per_sample:
                if value is not None:
                    parts.append(prefix + fmt(value, kind))
            else:
                parts.append(f'{prefix}{fmt(value["mean"], kind)} '
                             f'<small>({value["eligible_images"]}/{value["total_images"]} 张)</small>')
        return '<br>'.join(parts)

    validity = []
    for _, data in columns:
        valid = float(data['structure_valid']) if per_sample else data['output_usability']['first_pass_core_valid_rate']
        cell = f'首次 {fmt(valid)}'
        if not per_sample:
            cell += '<br><small>复检后：N/A（未运行复检）</small>'
        validity.append(cell)
    add_row('1. JSON 结构通过率', f'{total} 张', validity)

    detection = [('recall', '召回 ', 'percent'), ('matched_mean_iou', '已找到框 IoU ', 'iou')]
    groups = [
        ('2. photo 找到多少、框多准', 'photo', '个元素', detection),
        ('3. overlay/text/frame 找到多少、框多准', 'fixed', '个元素', detection),
        ('4. 重叠元素顺序一致率', 'layer_order', '对关系', [('rate', '', 'percent')]),
        ('5. frame 是否连上、是否全部连对', 'frame_links', '个 frame',
         [('connected_rate', '连上 ', 'percent'), ('correct_rate', '全部连对 ', 'percent')]),
    ]
    if 'photo_effects' not in first:
        groups.append(('6. linked_to 出框是否识别正确', 'linked_to', '个主体', [('rate', '', 'percent')]))
    for label, group, unit, fields in groups:
        entry = first[group]
        if per_sample and not entry['gt_count']:
            continue
        n = 1 if per_sample else entry[fields[0][0]]['eligible_images']
        add_row(label, f'{n} 张 · {entry["gt_count"]} {unit}',
                [group_cell(data[group], fields) for _, data in columns])

    if 'photo_effects' in first:
        effects = [summarize_effects([d['photo_effects']], 'raw_first') if per_sample else d['photo_effects']
                   for _, d in columns]
        add_row('V3 首轮完整契约通过率', f'{total} 张', [score(d['valid_rate']) for d in effects])
        for metric, label in [('background_accuracy', '背景类型准确率'), ('invalid_json_rate', '已返回预测的 JSON 解析失败率'),
                              ('truncated_rate', '截断率（仅有结束原因的预测）')]:
            add_row(label, '固定 GT', [score(d[metric]) for d in effects])
        for field, stats in effects[0]['detection'].items():
            if per_sample and not stats['gt_images']:
                continue
            cells = []
            for data in effects:
                scores = data['detection'][field]['end_to_end']
                keys = ('precision', 'recall', 'f1')
                if per_sample and any(scores[k] is None for k in keys):
                    cells.append(' / '.join(f'{name} {fmt(scores[k])}' for name, k in zip(('P', 'R', 'F1'), keys)
                                            if scores[k] is not None))
                else:
                    cells.append(' / '.join(fmt(scores[k]) for k in keys))
            add_row(field + ' · P / R / F1', f"{stats['gt_images']} 张 · {stats['gt']} 个", cells)
        for field in sorted(set().union(*(d['quality'] for d in effects))):
            cells = []
            for data in effects:
                measurement = data['quality'].get(field, {})
                value = score(measurement.get('mean'), 'iou')
                cells.append(f"{value} ({measurement.get('count', 0)})" if value else '')
            add_row(field, '匹配对象', cells)

    headers = ''.join(f'<th>{html.escape(label)}</th>' for label, _ in columns)
    return ('<table class="metrics"><thead><tr><th>关注项</th><th>固定 GT 范围</th>' +
            headers + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def export_metrics(run, targets, evaluations, source, *, running=False):
    out = Path(run) / 'metrics'
    out.mkdir(exist_ok=True)
    cohort = [{k: t[k] for k in ('id', 'image_digest', 'structure', 'occlusion_pairs')} for t in targets]
    digest = hashlib.sha256(json.dumps(cohort, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report = {'protocol': {'version': 'collage-structure-eval-v3-preview', 'reference_url': REFERENCE_URL,
                          'iou_threshold': .5, 'aggregation': 'per-image mean on fixed GT-positive cohorts',
                          'prediction_stage': 'raw_first', 'repair_performed': False,
                          'response_preprocessing': 'strip_empty_think_and_outer_json_fence_v1',
                          'gt_sha256': digest, 'gt_source': source, 'running': running,
                          'order_policies': sorted({t['occlusion_policy'] for t in targets})},
              'models': {name: summary for name, (summary, _) in evaluations.items()}}
    if any('photo_effects' in s for s,_ in evaluations.values()):
        report['protocol']['photo_effects'] = next(s['photo_effects']['version'] for s,_ in evaluations.values() if 'photo_effects' in s)
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
            if 'photo_effects' in summary:
                for field, d in summary['photo_effects']['detection'].items():
                    for scope in ('end_to_end','matched'):
                        for metric in ('precision','recall','f1'):
                            value = d[scope][metric]
                            writer.writerow([name, f'{field}.{scope}.{metric}', value, fmt(value), d['gt_images'], len(targets), d['gt']])
                for field, m in summary['photo_effects']['quality'].items():
                    writer.writerow([name, field, m['mean'], fmt(m['mean'], 'iou'), '', len(targets), m['count']])
            for group, key, label, kind in FIELDS:
                if group == 'linked_to' and 'photo_effects' in summary:
                    continue
                m = summary[group][key]
                writer.writerow([name, label, m['mean'], fmt(m['mean'], kind), m['eligible_images'], m['total_images'], summary[group]['gt_count']])
    with (out / 'per_sample.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        v3 = 'photo_effects' in report['protocol']
        fields = [f for f in FIELDS if not v3 or f[0] != 'linked_to']
        effect_names = list(next(iter(evaluations.values()))[0]['photo_effects']['detection']) if v3 else []
        extra = ['v3_valid', 'finish_reason'] + [k + '.f1' for k in effect_names] if v3 else []
        writer.writerow(['id', 'image', 'model', 'prediction_present', 'structure_valid'] + [f[2] for f in fields] + extra)
        for name, (_, samples) in evaluations.items():
            for target, sample in zip(targets, samples):
                extra = []
                if v3:
                    effects = sample['photo_effects']
                    extra = [not effects['errors'], effects['finish_reason']]
                    for k in effect_names:
                        g, p, correct, _, _ = effects['counts'][k]
                        extra.append(2 * correct / (g + p) if g + p else None)
                writer.writerow([sample['id'], target['images'][0], name, sample['present'], sample['structure_valid']] +
                                [sample[g][k] for g, k, _, _ in fields] + extra)
    return report
