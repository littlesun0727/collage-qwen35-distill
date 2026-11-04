"""Render saved ms-swift inference results; no GPU or model needed."""
import argparse
import hashlib
import html
import json
import re
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


def load_results(run, cache=None, partial=False):
    rows = []
    path = run / 'predictions.jsonl'
    if not path.exists() and partial:
        return rows
    # A writer can be midway through a UTF-8 character or JSON line.
    data = path.read_bytes()
    if partial:
        data = data[:data.rfind(b'\n') + 1]
    for line in data.decode('utf-8').splitlines():
        if not line.strip():
            continue
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
        with Image.open(source) as raw:
            im = ImageOps.exif_transpose(raw).convert('RGB')
            digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        response = row.get('response')
        error, target, wrappers = '', None, []
        try:
            if not isinstance(response, str):
                raise ValueError('Missing textual response')
            text, wrappers = unwrap_response(response)
            target = validate_target(text)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            target = None
        parsed = {'source': source, 'image': im, 'digest': digest, 'response': response,
                  'target': target, 'error': error, 'wrappers': wrappers}
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
    run = Path(run).resolve()
    rows = load_results(run, cache=cache, partial=partial)
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
        old = {r['digest']: r for r in load_results(baseline, cache=cache)}
    assets = run / 'preview-assets'
    assets.mkdir(exist_ok=True)
    cards = []
    valid = 0
    def cell(row, name, label):
        if row is None:
            return '<div>没有对应的基线图片</div>'
        if cache is None or not (assets / name).exists():
            picture = draw_boxes(row['image'], row['target']) if row['target'] else row['image']
            picture.save(assets / name)
        status = '基础结构检查通过' if row['target'] else '检查失败：' + row['error']
        if row['wrappers']:
            status += '（解析前已移除' + '、'.join(row['wrappers']) + '）'
        return (f'<div><h3>{label}</h3><img src="preview-assets/{name}"><p>{html.escape(status)}</p>'
                f'<details><summary>查看模型原始 JSON / 回复</summary><pre>{html.escape(str(row["response"]))}</pre></details></div>')
    for i, row in enumerate(rows):
        if cache is None or not (assets / f'{i}-original.png').exists():
            row['image'].save(assets / f'{i}-original.png')
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
            '这不是完整业务校验或准确率；查看照片框、装饰、frame 关联和 layer_order 的 JSON。'
            '若使用训练图片，只能检查拟合，不能证明泛化。</p>' + ''.join(cards))
    tmp = run / 'preview.html.tmp'
    tmp.write_text(page, encoding='utf-8')
    tmp.replace(run / 'preview.html')
    if not quiet:
        print('PREVIEW:', run / 'preview.html')
    return len(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    args = p.parse_args()
    build_report(args.run, args.baseline)
