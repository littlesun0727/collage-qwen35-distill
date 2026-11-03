"""Export a small, portable wiring-test dataset; not a train/validation split."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


def normalize_structure(source, size):
    value = copy.deepcopy({k: source[k] for k in ('background', 'objects', 'layer_order')})
    mode = source.get('coordinate_mode')
    if mode not in ('pixel', 'normalized_1000'):
        raise ValueError(f'Unknown coordinates: {mode}')
    ids = [o['id'] for o in value['objects']]
    if len(ids) != len(set(ids)) or sorted(ids) != sorted(value['layer_order']):
        raise ValueError('Duplicate IDs or inconsistent layer_order')
    photos = {o['id'] for o in value['objects'] if o['kind'] == 'photo'}
    bg = value['background']
    if bg.get('id') in ids:
        raise ValueError('Background ID collides with object')
    for obj in value['objects']:
        box = obj['bbox']
        bounds = size if mode == 'pixel' else (1000, 1000)
        if len(box) != 4 or not all(type(n) in (int, float) and math.isfinite(n) for n in box):
            raise ValueError('Invalid bbox values')
        if not (0 <= box[0] < box[2] <= bounds[0] and 0 <= box[1] < box[3] <= bounds[1]):
            raise ValueError('Invalid bbox extent')
        # Preserve small boxes with decimal precision rather than collapsing them to zero.
        converted = [round(n * 1000 / bounds[i % 2], 3) for i, n in enumerate(box)]
        if converted[0] >= converted[2] or converted[1] >= converted[3]:
            raise ValueError('Box collapsed during normalization')
        obj['bbox'] = converted
        if obj['kind'] == 'frame':
            refs = obj['photo_id']
            if not isinstance(refs, list) or not refs or len(refs) != len(set(refs)) or not set(refs) <= photos:
                raise ValueError('Invalid frame photo_id')
        if 'linked_to' in obj:
            targets = photos | ({bg['id']} if bg['type'] == 'photographic' else set())
            if obj['kind'] != 'photo' or obj['linked_to'] not in targets or obj['linked_to'] == obj['id']:
                raise ValueError('Invalid linked_to')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--prompt-template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    args = parser.parse_args()
    if args.limit < 1 or args.output.exists():
        parser.error('Use a positive limit and a new output directory')
    text = args.prompt_template.read_text(encoding='utf-8').strip()
    text = text.replace('{{BBOX_COORDINATES}}', 'bbox 格式为 [xmin, ymin, xmax, ymax]。所有坐标使用整张参考图的 0～1000 归一化坐标；x 相对于宽度，y 相对于高度。允许最多三位小数。')
    text = text.replace('{{EXAMPLE_COORDINATES}}', '示例坐标均为 0～1000 归一化坐标。')
    if '{{' in text:
        raise ValueError('Unresolved prompt placeholder')
    # Freeze the readable JSONL snapshot; no changes to teacher files.
    rows = [json.loads(line) for line in (args.report / 'annotations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    seen = set()
    selected = []
    for row in rows:
        sha = row['source_sha256']
        if sha in seen:
            continue
        image = args.report / 'inputs' / row['input_name']
        with Image.open(image) as raw:
            im = ImageOps.exif_transpose(raw)
            size = im.size
        if list(size) != row['analysis']['reference_size']:
            raise ValueError(f'Reference size mismatch: {image}')
        target = normalize_structure(row['structure'], size)
        selected.append((row, image, target))
        seen.add(sha)
        if len(selected) == args.limit:
            break
    if not selected:
        raise ValueError('No usable rows')
    args.output.mkdir(parents=True)
    (args.output / 'images').mkdir()
    (args.output / 'prompt.txt').write_text(text, encoding='utf-8')
    records, metadata = [], []
    for row, image, target in selected:
        name = f"images/{row['image_id']}.png"
        with Image.open(image) as raw:
            ImageOps.exif_transpose(raw).convert('RGB').save(args.output / name)
        records.append({'messages': [
            {'role': 'system', 'content': text},
            {'role': 'user', 'content': '<image>请解析参考图，只返回完整 JSON 对象，不输出代码围栏或解释。'},
            {'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False, separators=(',', ':'))},
        ], 'images': [name]})
        metadata.append({'image_id': row['image_id'], 'source_sha256': row['source_sha256'],
                         'label_status': row.get('label_status'), 'source_coordinate_mode': row['structure']['coordinate_mode']})
    (args.output / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    (args.output / 'manifest.json').write_text(json.dumps({
        'purpose': 'smoke_only_not_for_model_selection', 'count': len(records),
        'family_leakage_checked': False, 'coordinate_mode': 'normalized_1000',
        'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'samples': metadata,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Exported {len(records)} smoke samples to {args.output}')


if __name__ == '__main__':
    main()
