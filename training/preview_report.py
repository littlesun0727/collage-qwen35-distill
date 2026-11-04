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
    if not path.exists() and partial:
        return rows
    # A writer can be midway through a UTF-8 character or JSON line.
    if progress:
        print(f'[PREVIEW] 读取结果文件：{path}', flush=True)
    data = path.read_bytes()
    if partial:
        data = data[:data.rfind(b'\n') + 1]
    lines = [line for line in data.decode('utf-8').splitlines() if line.strip()]
    for number, line in enumerate(lines, 1):
        if cache is not None and line in cache:
            rows.append(cache[line])
            continue
        row = json.loads(line)
        paths = row.get('images', [])
        reference = paths[0] if len(paths) == 1 else None
        # datasets.Image / SWIFT may serialize local images as {path, bytes}.
        if isinstance(reference, dict) and reference.get('bytes') is None:
            reference = reference.get('path')
        if not isinstance(reference, str) or not reference:
            raise ValueError('Expected one image path in each ms-swift result')
        source = Path(reference)
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
            cache[line] = parsed
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


def build_report(run, baseline=None, *, partial=False, expected=None, cache=None, quiet=False, running=None):
    started = time.monotonic()
    run = Path(run).resolve()
    if not quiet:
        print('[PREVIEW] 开始生成报告（仅 CPU 画图，不加载模型）', flush=True)
    rows = load_results(run, cache=cache, partial=partial, progress=not quiet)
    if not rows and not partial:
        raise ValueError('No inference results')
    old = {}
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
        old = {r['digest']: r for r in load_results(baseline, cache=cache, progress=not quiet)}
    assets = run / 'preview-assets'
    assets.mkdir(exist_ok=True)
    cards = []
    valid = 0
    def cell(row, name, label):
        if row is None:
            return '<div>没有对应的基线图片</div>'
        if cache is None or not (assets / name).exists():
            picture = draw_boxes(row['image'], row['drawing'])
            picture.save(assets / name, compress_level=1)
        status = f'已绘制 {len(row["drawing"]["objects"])} 个框。'
        status += '基础结构检查通过' if row['target'] is not None else '结构提示（不阻止画框）：' + row['error']
        if row['drawing_notes']:
            status += '；' + '；'.join(row['drawing_notes'])
        if row['wrappers']:
            status += '（解析前已移除' + '、'.join(row['wrappers']) + '）'
        return (f'<div><h3>{label}</h3><img src="preview-assets/{name}"><p>{html.escape(status)}</p>'
                f'<details><summary>查看模型原始 JSON / 回复</summary><pre>{html.escape(str(row["response"]))}</pre></details></div>')
    for i, row in enumerate(rows):
        if not quiet:
            print(f'[PREVIEW] 绘制 {i + 1}/{len(rows)}：{row["source"].name} '
                  f'（{len(row["drawing"]["objects"])} 个可用框）', flush=True)
        if cache is None or not (assets / f'{i}-original.png').exists():
            row['image'].save(assets / f'{i}-original.png', compress_level=1)
        valid += row['target'] is not None
        cells = f'<div><h3>原图</h3><img src="preview-assets/{i}-original.png"></div>'
        if baseline:
            cells += cell(old.get(row['digest']), f'{i}-baseline.png', '基线')
        cells += cell(row, f'{i}-current.png', '当前模型')
        if row['target']:
            (assets / f'{i}-structure.json').write_text(json.dumps({
                'coordinate_mode': 'normalized_1000', **row['target']}, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            (assets / f'{i}-structure.json').unlink(missing_ok=True)
        cards.append(f'<article><h2>{html.escape(row["source"].name)}</h2><section>{cells}</section></article>')
    progress = f'已完成 {len(rows)} / {expected}' if expected is not None else f'已完成 {len(rows)} 张'
    running = partial if running is None else running
    refresh = '<meta http-equiv="refresh" content="10">' if running else ''
    page = ('<!doctype html><meta charset="utf-8">' + refresh + '<title>LoRA 预测预览</title>'
            '<style>body{font-family:sans-serif;margin:24px;background:#f4f5f7}article{background:white;padding:16px;margin:20px 0}'
            'section{display:flex;gap:12px}section>div{flex:1;min-width:0}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            f'<h1>LoRA 预测预览</h1><p>{progress}。' + ('推理进行中，每10秒刷新。' if running else '推理已结束；是否全部成功请核对完成数量和日志。') +
            f'</p><p>基础结构检查通过 {valid}/{len(rows)}。'
            '画框独立于结构检查：能读取的框尽量显示，坏框单独跳过；显示调整不修改原始回复。'
            '预览图最长边限制为1600像素，不改变推理输入或 bbox 坐标。'
            '检查通过比例不是准确率；查看照片框、装饰、frame 关联和 layer_order 的 JSON。'
            '若使用训练图片，只能检查拟合，不能证明泛化。</p>' + ''.join(cards))
    tmp = run / 'preview.html.tmp'
    tmp.write_text(page, encoding='utf-8')
    tmp.replace(run / 'preview.html')
    if not quiet:
        print(f'PREVIEW: {run / "preview.html"}（耗时 {time.monotonic() - started:.1f} 秒）', flush=True)
    return len(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    args = p.parse_args()
    build_report(args.run, args.baseline)
