"""Render saved ms-swift inference results; no GPU or model needed."""
import argparse
import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from prepare_smoke import normalize_structure


def load_results(run):
    rows = []
    for line in (run / 'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        paths = row.get('images', [])
        if len(paths) != 1 or not isinstance(paths[0], str):
            raise ValueError('Expected one image path in each ms-swift result')
        source = Path(paths[0])
        with Image.open(source) as raw:
            im = ImageOps.exif_transpose(raw).convert('RGB')
            digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        response = row.get('response')
        error, target = '', None
        try:
            if not isinstance(response, str):
                raise ValueError('Missing textual response')
            target = json.loads(response)
            if not isinstance(target, dict) or set(target) != {'background', 'objects', 'layer_order'}:
                raise ValueError('Output must contain exactly background, objects, layer_order')
            target = normalize_structure({'coordinate_mode': 'normalized_1000', **target}, im.size)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            target = None
        rows.append({'source': source, 'image': im, 'digest': digest, 'response': response,
                     'target': target, 'error': error})
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


def build_report(run, baseline=None):
    run = Path(run).resolve()
    rows = load_results(run)
    if not rows:
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
        old = {r['digest']: r for r in load_results(baseline)}
    assets = run / 'preview-assets'
    assets.mkdir(exist_ok=True)
    cards = []
    valid = 0
    def cell(row, name, label):
        if row is None:
            return '<div>没有对应的基线图片</div>'
        picture = draw_boxes(row['image'], row['target']) if row['target'] else row['image']
        picture.save(assets / name)
        status = '基础结构检查通过' if row['target'] else '检查失败：' + row['error']
        return (f'<div><h3>{label}</h3><img src="preview-assets/{name}"><p>{html.escape(status)}</p>'
                f'<details><summary>查看模型原始 JSON / 回复</summary><pre>{html.escape(str(row["response"]))}</pre></details></div>')
    for i, row in enumerate(rows):
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
    page = ('<!doctype html><meta charset="utf-8"><title>LoRA 预测预览</title>'
            '<style>body{font-family:sans-serif;margin:24px;background:#f4f5f7}article{background:white;padding:16px;margin:20px 0}'
            'section{display:flex;gap:12px}section>div{flex:1;min-width:0}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            f'<h1>LoRA 预测预览</h1><p>基础结构检查通过 {valid}/{len(rows)}。'
            '这不是完整业务校验或准确率；查看照片框、装饰、frame 关联和 layer_order 的 JSON。'
            '若使用训练图片，只能检查拟合，不能证明泛化。</p>' + ''.join(cards))
    (run / 'preview.html').write_text(page, encoding='utf-8')
    print('PREVIEW:', run / 'preview.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    args = p.parse_args()
    build_report(args.run, args.baseline)
