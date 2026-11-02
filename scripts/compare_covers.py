"""Compare completed covers runs; coordinate rounding is not localization accuracy."""
import argparse
import json
from pathlib import Path


def summarize(root):
    summary = json.loads((root / 'batch-summary.json').read_text(encoding='utf-8'))
    counts = {'coordinates': 0, 'multiples_of_10': 0, 'interior_coordinates': 0,
              'interior_multiples_of_10': 0, 'objects': 0, 'all_four_multiple_of_10': 0}
    conversions, errors = 0, []
    for row in summary['cases']:
        if row['status'] != 'complete':
            continue
        folder = root / 'cases' / row['id']
        structure = json.loads((folder / 'structure.json').read_text(encoding='utf-8'))
        analysis = json.loads((folder / 'analysis.json').read_text(encoding='utf-8'))
        size = analysis['reference_size']
        pixels = {item['id']: item for item in analysis['objects']}
        for item in structure['objects']:
            box = item['bbox']
            counts['objects'] += 1
            counts['all_four_multiple_of_10'] += all(v % 10 == 0 for v in box)
            for v in box:
                counts['coordinates'] += 1
                counts['multiples_of_10'] += v % 10 == 0
                if 0 < v < 1000:
                    counts['interior_coordinates'] += 1
                    counts['interior_multiples_of_10'] += v % 10 == 0
            expected = [round(v * size[i % 2] / 1000) for i, v in enumerate(box)]
            conversions += 1
            if expected != pixels[item['id']]['bbox']:
                errors.append({'case': row['id'], 'object': item['id']})
    for name, numerator, denominator in [
        ('coordinate_multiple_of_10_rate', 'multiples_of_10', 'coordinates'),
        ('interior_multiple_of_10_rate', 'interior_multiples_of_10', 'interior_coordinates'),
        ('whole_bbox_multiple_of_10_rate', 'all_four_multiple_of_10', 'objects')]:
        counts[name] = round(counts[numerator] / counts[denominator], 4) if counts[denominator] else None
    return {**{k: v for k, v in summary.items() if k != 'cases'},
            'rounding': counts, 'pixel_conversions_checked': conversions,
            'pixel_conversion_errors': errors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    args = parser.parse_args()
    result = {'old': summarize(args.old), 'new': summarize(args.new),
              'scope': 'Accepted outputs only; rounding rates and schema validity do not measure localization accuracy. Consult each run provenance for model, prompt and configuration differences.'}
    (args.new / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))
