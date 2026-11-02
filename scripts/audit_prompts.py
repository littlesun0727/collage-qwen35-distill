"""Audit saved model-facing messages, separately from local provenance metadata."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
report = {'requests_checked': 0, 'stages': {}, 'reference_size_in_messages': [],
          'pixel_caption_in_messages': [], 'parameters_exposing_dimensions': [],
          'generic_pixel_mode_clause': [], 'sample_user_text': None}
for path in sorted((args.root / 'cases').glob('*/workflow/vlm/*/*/request-*.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    texts = []
    user_texts = []
    for message in payload['messages']:
        content = message['content']
        parts = [content] if isinstance(content, str) else [part['text'] for part in content if part.get('type') == 'text']
        texts.extend(parts)
        if message['role'] == 'user':
            user_texts.extend(parts)
    combined = '\n'.join(texts)
    relative = path.relative_to(args.root).as_posix()
    report['requests_checked'] += 1
    stage = path.parents[1].name
    report['stages'][stage] = report['stages'].get(stage, 0) + 1
    if 'reference_size' in combined:
        report['reference_size_in_messages'].append(relative)
    if '实际像素尺寸' in combined or 'bbox 使用原图整数像素' in combined:
        report['pixel_caption_in_messages'].append(relative)
    if '像素模式应使用实际画布尺寸' in combined:
        report['generic_pixel_mode_clause'].append(relative)
    parameters = json.loads((path.parent / 'parameters.json').read_text(encoding='utf-8'))
    if parameters.get('image_dimensions_in_prompt') is not False:
        report['parameters_exposing_dimensions'].append(relative)
    if report['sample_user_text'] is None and stage == 'analyze':
        report['sample_user_text'] = {'request': relative, 'texts': user_texts}
report['generic_pixel_mode_clause_count'] = len(report.pop('generic_pixel_mode_clause'))
(args.root / 'prompt-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=True, indent=2))
