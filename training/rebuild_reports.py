"""Rebuild saved evaluation reports recursively, without model loading or inference."""
import argparse
import json
from pathlib import Path

from preview_report import build_report, load_results
from preview_metrics import load_ground_truth, write_json


def discover(root, include_history=False):
    return sorted({p.parent.resolve() for p in root.rglob('predictions.jsonl')
                   if include_history or '_history' not in p.relative_to(root).parts})


def queue_baseline(run):
    """Recover explicit queue comparisons; do not guess from directory names."""
    plan_path = run.parent / 'plan.json'
    if not plan_path.is_file():
        return None
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    job = next((j for j in plan if j.get('name') == run.name), None)
    if not job or not job.get('compare_to'):
        return None
    name = job['compare_to']
    baseline = run.parent / name
    if baseline.parent.resolve() != run.parent or not (baseline / 'predictions.jsonl').is_file():
        raise ValueError(f'Missing or invalid queue baseline: {name}')
    return baseline


def rebuild(root, gt=None, baseline=None, include_history=False, dry_run=False):
    root = root.resolve()
    runs = discover(root, include_history)
    if not runs:
        raise ValueError(f'No predictions.jsonl found under {root}')
    results = []
    for index, run in enumerate(runs, 1):
        print(f'[{index}/{len(runs)}] {run}', flush=True)
        if dry_run:
            continue
        try:
            comparison = baseline.resolve() if baseline else queue_baseline(run)
            if comparison == run:
                comparison = None
            cache = {}
            rows = load_results(run, cache=cache)
            targets, source = load_ground_truth(run, rows, gt, cache)
            if not targets:
                raise ValueError(f'Cannot score without GT: {source}; supply --gt')
            count = build_report(run, comparison, gt=gt, cache=cache, quiet=True)
            results.append({'run': str(run), 'status': 'success', 'predictions': count,
                            'gt_count': len(targets), 'preview': str(run / 'preview.html')})
            print(f'  OK: {count}/{len(targets)} predictions; preview.html + metrics/', flush=True)
        except Exception as exc:
            results.append({'run': str(run), 'status': 'failed', 'error': str(exc)})
            print(f'  FAILED: {exc}', flush=True)
    if dry_run:
        return 0
    write_json(root / 'report-rebuild-status.json', results)
    failed = sum(r['status'] == 'failed' for r in results)
    print(f'Done: {len(results) - failed} succeeded, {failed} failed. '
          f'Status: {root / "report-rebuild-status.json"}', flush=True)
    return int(failed > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Top-level results directory')
    parser.add_argument('--gt', type=Path, help='Optional shared test.jsonl; otherwise recover each run\'s GT')
    parser.add_argument('--baseline', type=Path, help='Optional shared baseline; defaults to queue plan comparisons')
    parser.add_argument('--include-history', action='store_true', help='Also rebuild archived _history runs')
    parser.add_argument('--dry-run', action='store_true', help='List discovered runs without writing')
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f'Results directory not found: {args.root}')
    try:
        code = rebuild(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
