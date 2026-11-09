"""Render saved ms-swift inference results; no GPU or model needed."""
import argparse
import hashlib
import html
import json
import math
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from dataset_io import validate_target
from preview_metrics import (REFERENCE_URL, export_metrics, image_path, load_ground_truth,
                             metric_table, pixel_digest, score_rows)


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


def drawable_target(text):
    """Best-effort visualization only; never replace the strict training validator."""
    notes = []
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return {'objects': []}, ['回复不是完整 JSON，无法可靠提取框']
    if not isinstance(value, dict) or not isinstance(value.get('objects'), list):
        return {'objects': []}, ['没有可读取的 objects 列表']
    mode = value.get('coordinate_mode', 'normalized_1000')
    if mode != 'normalized_1000':
        return {'objects': []}, [f'未知或非 normalized_1000 坐标模式：{mode}，未猜测坐标尺度']
    objects = []
    for index, obj in enumerate(value['objects'], 1):
        box = obj.get('bbox') if isinstance(obj, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            notes.append(f'对象 {index}：bbox 不是四个坐标，已跳过')
            continue
        try:
            if any(isinstance(n, bool) for n in box):
                raise ValueError('boolean coordinate')
            box = [float(n) for n in box]
            if not all(math.isfinite(n) for n in box):
                raise ValueError('non-finite coordinate')
        except (ValueError, TypeError, OverflowError):
            notes.append(f'对象 {index}：坐标不是有限数字，已跳过')
            continue
        x1, y1, x2, y2 = box
        fixed = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        fixed = [min(1000, max(0, n)) for n in fixed]
        if fixed[0] >= fixed[2] or fixed[1] >= fixed[3]:
            notes.append(f'对象 {index}：框没有可见面积，已跳过')
            continue
        if fixed != box:
            notes.append(f'对象 {index}：仅为显示整理反向坐标或裁剪越界部分，原始输出未改')
        objects.append({'bbox': fixed, 'kind': str(obj.get('kind') or 'unknown'),
                        'id': str(obj.get('id') or f'object_{index}')})
    return {'objects': objects}, notes


def load_results(run, cache=None, partial=False, progress=False):
    rows = []
    path = run / 'predictions.jsonl'
    if not path.exists():
        return rows
    # A writer can be midway through a UTF-8 character or JSON line.
    if progress:
        print(f'[PREVIEW] 读取结果文件：{path}', flush=True)
    data = path.read_bytes()
    if partial:
        data = data[:data.rfind(b'\n') + 1]
    lines = [line for line in data.decode('utf-8').splitlines() if line.strip()]
    for number, line in enumerate(lines, 1):
        cache_key = ('prediction', str(run), line)
        if cache is not None and cache_key in cache:
            rows.append(cache[cache_key])
            continue
        row = json.loads(line)
        paths = row.get('images', [])
        reference = paths[0] if len(paths) == 1 else None
        # datasets.Image / SWIFT may serialize local images as {path, bytes}.
        if isinstance(reference, dict) and reference.get('bytes') is None:
            reference = reference.get('path')
        if not isinstance(reference, str) or not reference:
            raise ValueError('Expected one image path in each ms-swift result')
        source = (run / reference).resolve()
        if progress:
            print(f'[PREVIEW] 读取图片 {number}/{len(lines)}：{source}', flush=True)
        with Image.open(source) as raw:
            im = ImageOps.exif_transpose(raw).convert('RGB')
            digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            # Hash full pixels for baseline identity, then retain only display pixels.
            # This does not alter model input, raw predictions or exported coordinates.
            im.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        response = row.get('response')
        error, target, wrappers = '', None, []
        text = ''
        try:
            if not isinstance(response, str):
                raise ValueError('Missing textual response')
            text, wrappers = unwrap_response(response)
            target = validate_target(text)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            if error == 'Duplicate IDs or inconsistent layer_order':
                error = '对象 ID 重复或 layer_order 不一致；所有可用 bbox 仍分别绘制'
            target = None
        drawing, drawing_notes = drawable_target(text)
        parsed = {'source': source, 'image': im, 'digest': digest, 'response': response,
                  'target': target, 'error': error, 'wrappers': wrappers,
                  'drawing': drawing, 'drawing_notes': drawing_notes}
        rows.append(parsed)
        if cache is not None:
            cache[cache_key] = parsed
    return rows


def draw_boxes(image, target):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    colors = {'photo': '#008cff', 'frame': '#00a040', 'overlay': '#ff304f', 'text': '#a030df'}
    w, h = result.size
    for obj in target['objects']:
        box = [n * (w if i % 2 == 0 else h) / 1000 for i, n in enumerate(obj['bbox'])]
        color = colors.get(obj['kind'], '#ff9900')
        draw.rectangle(box, outline=color, width=max(2, w // 350))
        # IDs follow the project ASCII contract; tolerate unexpected IDs in diagnostics.
        label = f"{obj['kind']}:{obj['id']}".encode('ascii', 'replace').decode()
        draw.text((box[0], max(0, box[1] - 12)), label, fill=color, stroke_width=1, stroke_fill='white')
    return result


def build_report(run, baseline=None, *, partial=False, expected=None, cache=None, quiet=False, running=None, gt=None):
    started = time.monotonic()
    run = Path(run).resolve()
    if not quiet:
        print('[PREVIEW] 开始生成报告（仅 CPU 画图，不加载模型）', flush=True)
    rows = load_results(run, cache=cache, partial=partial, progress=not quiet)
    completed = len(rows)
    targets, gt_source = load_ground_truth(run, rows, gt, cache)
    if not rows and not partial and not targets:
        raise ValueError('No inference results')
    old = {}
    baseline_rows = []
    if baseline:
        baseline = Path(baseline).resolve()
        # Prompt and visual budget must match for a meaningful comparison.
        current_input = json.loads((run / 'input.jsonl').read_text(encoding='utf-8').splitlines()[0])
        prior_input = json.loads((baseline / 'input.jsonl').read_text(encoding='utf-8').splitlines()[0])
        if current_input['messages'] != prior_input['messages']:
            raise ValueError('Baseline prompt differs from current prompt')
        current_cmd = json.loads((run / 'command.json').read_text(encoding='utf-8'))
        prior_cmd = json.loads((baseline / 'command.json').read_text(encoding='utf-8'))
        if current_cmd['image_tokens'] != prior_cmd['image_tokens']:
            raise ValueError('Baseline image-token budget differs')
        baseline_rows = load_results(baseline, cache=cache, progress=not quiet)
        for item in baseline_rows:
            if item['digest'] in old:
                raise ValueError('Baseline has ambiguous duplicate image pixels')
            old[item['digest']] = item
    running = partial if running is None else running
    evaluations = {}
    overview = ''
    if targets:
        if not quiet:
            print(f'[PREVIEW] 六项评测：固定 GT {len(targets)} 张；缺失预测保留在分母', flush=True)
        current_paths = {Path(t['images'][0]) for t in targets}
        if any(r['source'] not in current_paths for r in rows):
            raise ValueError('Predictions contain images outside the fixed input/GT cohort')
        if baseline:
            evaluations[f'基线 · {baseline.name}'] = score_rows(targets, baseline_rows, cache=cache)
        evaluations[f'当前模型 · {run.name}'] = score_rows(targets, rows, cache=cache)
        export_metrics(run, targets, evaluations, gt_source, running=running)
        overview = '<div class="overview"><h2>六项评测 · 固定 GT ' + str(len(targets)) + ' 张</h2>'
        overview += metric_table([(name, summary) for name, (summary, _) in evaluations.items()])
        overview += ('<p class="note">主表计分来源：当前首次原始输出（未运行复检）。'
                     'IoU ≥ 0.50 一对一匹配；先逐图计算再平均。缺失/坏 JSON 不从 GT 分母剔除；'
                     '已找到框 IoU 为条件分数，须同时看召回率及适用图片数。'
                     '固定元素组内不区分 overlay/text/frame。顺序读取 GT 中固定的关系；'
                     '自动关系由 GT 框的正面积重叠和 layer_order 生成，是遮挡的近似。'
                     '教师 GT 得分表示与标注的一致性。</p>'
                     '<p><a href="metrics/summary.csv" download>下载汇总 CSV</a> · '
                     '<a href="metrics/per_sample.csv" download>下载逐图 CSV</a> · '
                     '<a href="metrics/summary.json">完整指标 JSON</a> · '
                     '<a href="metrics/per_sample.jsonl">逐图诊断 JSONL</a> · '
                     f'<a href="{REFERENCE_URL}">指标口径参考</a></p>')
        diagnostics = {name: summary['diagnostics'] for name, (summary, _) in evaluations.items()}
        overview += '<details><summary>多报、反序与失败诊断</summary><pre>' + html.escape(
            json.dumps(diagnostics, ensure_ascii=False, indent=2)) + '</pre></details></div>'
        by_path = {r['source']: r for r in rows}
        display_rows = []
        for target in targets:
            source = Path(target['images'][0])
            row = by_path.get(source)
            if row is None:
                key = ('missing_image', str(source))
                row = cache.get(key) if cache is not None else None
                if row is None:
                    with Image.open(source) as raw:
                        im = ImageOps.exif_transpose(raw).convert('RGB')
                        digest = pixel_digest(im)
                        im.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                    if digest != target['image_digest']:
                        raise ValueError(f'Image pixels changed since GT snapshot: {source}')
                    row = {'source': source, 'image': im, 'digest': digest, 'response': None,
                           'target': None, 'drawing': {'objects': []}, 'drawing_notes': [],
                           'wrappers': [], 'error': '暂无完整预测，已保留在固定评测分母', 'missing': True}
                    if cache is not None:
                        cache[key] = row
            display_rows.append(row)
        rows = display_rows
        if expected is None:
            expected = len(targets)
    else:
        overview = '<div class="overview"><h2>GT / 指标尚未就绪</h2><p>' + html.escape(gt_source) + '</p></div>'
    assets = run / 'preview-assets'
    assets.mkdir(exist_ok=True)
    cards = []
    valid = 0
    def save_picture(row, name, drawing=None):
        drawing = row['drawing'] if drawing is None else drawing
        stamp = (row['digest'], json.dumps(drawing, sort_keys=True))
        key = ('rendered_asset', str(run), name)
        if cache is None or cache.get(key) != stamp or not (assets / name).exists():
            draw_boxes(row['image'], drawing).save(assets / name, compress_level=1)
            if cache is not None:
                cache[key] = stamp

    def cell(row, name, label):
        if row is None:
            return f'<div><h3>{label}</h3><p>没有对应的基线图片；按缺失预测计分。</p></div>'
        save_picture(row, name)
        status = f'已绘制 {len(row["drawing"]["objects"])} 个框。'
        status += '基础结构检查通过' if row['target'] is not None else '结构提示（不阻止画框）：' + row['error']
        if row['drawing_notes']:
            status += '；' + '；'.join(row['drawing_notes'])
        if row['wrappers']:
            status += '（解析前已移除' + '、'.join(row['wrappers']) + '）'
        return (f'<div><h3>{label}</h3><a href="preview-assets/{name}"><img loading="lazy" width="{row["image"].width}" height="{row["image"].height}" alt="{label}框线图" src="preview-assets/{name}"></a><p>{html.escape(status)}</p>'
                f'<details><summary>查看模型原始 JSON / 回复</summary><pre>{html.escape(str(row["response"]))}</pre></details></div>')
    for i, row in enumerate(rows):
        if not quiet:
            print(f'[PREVIEW] 绘制 {i + 1}/{len(rows)}：{row["source"].name} '
                  f'（{len(row["drawing"]["objects"])} 个可用框）', flush=True)
        save_picture(row, f'{i}-original.png', {'objects': []})
        valid += row['target'] is not None
        if targets:
            target = targets[i]
            save_picture(row, f'{i}-gt.png', target['structure'])
            cells = (f'<div class="gt"><h3>GT 标注框线图</h3><a href="preview-assets/{i}-gt.png">'
                     f'<img loading="lazy" width="{row["image"].width}" height="{row["image"].height}" alt="GT 标注框线图" src="preview-assets/{i}-gt.png"></a>'
                     f'<p>{len(target["structure"]["objects"])} 个 GT 框 · '
                     f'<a href="preview-assets/{i}-original.png">查看原图</a></p>'
                     '<details><summary>查看 GT JSON</summary><pre>' +
                     html.escape(json.dumps(target['structure'], ensure_ascii=False, indent=2)) + '</pre></details></div>')
        else:
            cells = (f'<div class="gt"><h3>GT 标注框线图</h3><p>未提供 GT</p>'
                     f'<h3>原图</h3><img loading="lazy" width="{row["image"].width}" height="{row["image"].height}" alt="原图" src="preview-assets/{i}-original.png"></div>')
        if baseline:
            cells += cell(old.get(row['digest']), f'{i}-baseline.png', '基线')
        cells += cell(row, f'{i}-current.png', '当前模型')
        if row['target']:
            (assets / f'{i}-structure.json').write_text(json.dumps({
                'coordinate_mode': 'normalized_1000', **row['target']}, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            (assets / f'{i}-structure.json').unlink(missing_ok=True)
        scores = metric_table([(name, samples[i]) for name, (_, samples) in evaluations.items()], per_sample=True) if targets else ''
        cards.append(f'<article id="sample-{i + 1}"><h2>{i + 1}. {html.escape(row["source"].name)}</h2><section class="pictures">{cells}</section>{scores}</article>')
    progress = f'已完成 {completed} / {expected}' if expected is not None else f'已完成 {completed} 张'
    refresh = '<meta http-equiv="refresh" content="10">' if running else ''
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + refresh + '<title>GT 对照与六项评测</title>'
            '<style>body{font:15px/1.6 system-ui,sans-serif;margin:24px;background:#f2f5fa;color:#203249}'
            'article,.overview{background:white;padding:20px;margin:24px 0;border:1px solid #dce4ee;border-radius:12px}'
            'h1{font-size:28px}h2{font-size:20px}h3{margin:0 0 12px;font-size:16px}a{color:#1264bd}'
            '.pictures{display:flex;gap:16px;align-items:flex-start}.pictures>div{flex:1;min-width:0}.gt h3{color:#087b64}'
            'img{width:100%;height:auto;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}'
            '.metrics{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}.metrics td,.metrics th{border:1px solid #d8e1ed;padding:10px;text-align:left}'
            '.metrics th{background:#eaf0f8}.metrics tbody tr:nth-child(even){background:#f8fafc}small,.note{color:#617085}'
            '.pictures p{font-size:12px}.legend{padding:8px 0}.legend span{margin-right:18px}'
            '@media(max-width:760px){body{margin:10px}.overview,article{padding:12px;overflow:auto}.pictures{min-width:760px}.metrics{min-width:640px}}'
            '@media print{body{margin:0;background:white}article{break-inside:avoid}details{display:none}a{color:inherit;text-decoration:none}}</style>'
            f'<h1>GT 对照与六项评测</h1><p>{progress}。' + ('推理进行中，每10秒刷新；当前分数为临时结果，未完成预测暂按缺失计分。' if running else '推理已结束；是否全部成功请核对完成数量和日志。') +
            f'</p><p>基础结构检查通过 {valid}/{len(rows)}。'
            '画框独立于结构检查：能读取的框尽量显示，坏框单独跳过；显示调整不修改原始回复。'
            '预览图最长边限制为1600像素，不改变推理输入或 bbox 坐标。'
            '检查通过比例不是准确率；查看照片框、装饰、frame 关联和 layer_order 的 JSON。'
            '若使用训练图片，只能检查拟合，不能证明泛化。</p>'
            '<p class="legend"><span style="color:#008cff">■ photo</span><span style="color:#00a040">■ frame</span>'
            '<span style="color:#ff304f">■ overlay</span><span style="color:#a030df">■ text</span>'
            '点击图片查看大图；GT 固定在左列。</p>' + overview + ''.join(cards) + '</html>')
    tmp = run / 'preview.html.tmp'
    tmp.write_text(page, encoding='utf-8')
    tmp.replace(run / 'preview.html')
    if not quiet:
        print(f'PREVIEW: {run / "preview.html"}（耗时 {time.monotonic() - started:.1f} 秒）', flush=True)
    return completed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--gt', type=Path, help='GT JSONL / dataset test.jsonl; defaults to saved GT or command.json data/split')
    args = p.parse_args()
    build_report(args.run, args.baseline, gt=args.gt)
