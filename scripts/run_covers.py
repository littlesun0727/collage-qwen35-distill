"""Run the bundled collage-recreate analyze pipeline against local on-demand vLLM."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time

from local_transport import infer

ROOT = Path(__file__).resolve().parents[1]

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=ROOT/'vendor/collage-recreate')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model', choices=['Qwen3.5-9B', 'Qwen3.5-4B'], default='Qwen3.5-9B')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-failures-from', type=Path,
                        help='Completed batch directory whose failed cases should be rerun in a new output directory')
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--workers and --limit must be positive')
    manifest_path = ROOT/'models'/args.model/'download-manifest.json'
    if not manifest_path.exists():
        parser.error('Download the selected model first: scripts/download_curl.py')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['model'] != 'Qwen/' + args.model:
        parser.error('Model manifest mismatch')
    sys.path.insert(0, str(args.repo.resolve()))
    from collage.analysis import run_analysis
    from collage.analysis_batch import discover, write_analysis_bbox
    from collage.config import Config
    from collage.storage import read, save, now, exclusive
    from collage.vlm.client import Client, ModelTransportError
    from collage import prompts
    from collage.workflow.stages.analyze import VERSION
    from PIL import Image

    config = Config(provider='vllm', model=args.model, coordinate_mode='normalized_1000',
                    reasoning_enabled=False, response_format='json_object', max_tokens=16000,
                    max_requests=4, timeout=600)
    source, output, cases = discover(args.input, args.output)
    if args.retry_failures_from:
        prior = read(args.retry_failures_from / 'batch-summary.json')
        if not prior or 'elapsed_seconds_this_invocation' not in prior:
            raise ValueError('Retry source must be a completed batch.')
        if output.resolve() == args.retry_failures_from.resolve():
            raise ValueError('Retry output must be a separate directory.')
        failed = {row['id'] for row in prior['cases'] if row['status'] == 'failed'}
        cases = [case for case in cases if case['id'] in failed]
        if {case['id'] for case in cases} != failed:
            raise ValueError('Some failed inputs are missing.')
        if not cases:
            raise ValueError('No failed cases to retry.')
    if args.limit:
        cases = cases[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    provenance = {'source': str(source), 'upstream': str(args.repo.resolve()), 'version': VERSION,
                  'policy': config.model_policy(), 'transport': 'local loopback HTTP vLLM',
                  'model_revision': manifest['revision'], 'files': {}}
    if args.retry_failures_from:
        provenance['retry_failures_from'] = str(args.retry_failures_from.resolve())
    provenance['inputs'] = {c['relative_path']: hashlib.sha256(Path(c['reference']).read_bytes()).hexdigest() for c in cases}
    for folder in ('collage', 'engine/schemas', 'engine/scripts'):
        for path in sorted((args.repo / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.md', '.json'):
                provenance['files'][path.relative_to(args.repo).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    old = read(output / 'provenance.json')
    if old and old != provenance:
        raise ValueError('Upstream or configuration changed; use a new output directory.')
    save(output / 'provenance.json', provenance)
    (output / 'analyze-prompt.txt').write_text(prompts.render('analyze', mode='normalized_1000'), encoding='utf-8')

    def process(case):
        run = output / 'cases' / case['id']
        existing = read(run / 'case-summary.json')
        if existing:
            return existing
        started = time.monotonic()
        row = {**case, 'status': 'running', 'started_at': now()}
        run.mkdir(parents=True, exist_ok=True)
        transport_count = 0
        def transport(payload):
            nonlocal transport_count
            transport_count += 1
            try:
                return infer(payload)
            except Exception as error:
                raise ModelTransportError(str(error)) from error
        client = Client(run / 'workflow/vlm', config, transport=transport)
        try:
            analysis = run_analysis(case['reference'], run, config, client)
            write_analysis_bbox(run, analysis)
            structure = read(run / 'structure.json')
            row.update(status='complete', objects=len(structure['objects']),
                       kinds=dict(Counter(o['kind'] for o in structure['objects'])),
                       background_type=structure['background']['type'])
        except Exception as error:
            row.update(status='failed', error=str(error)[:3000])
        evidence = read(run / 'workflow/structure-analysis.json') or {}
        row.update(seconds=round(time.monotonic() - started, 2), evidence=evidence, usage=client.usage(), calls=transport_count)
        row['raw_schema_valid'] = False
        normalizations = list((run / 'workflow/vlm/analyze').glob('*/normalization-0.json'))
        if normalizations:
            row['raw_schema_valid'] = read(normalizations[0]).get('raw_schema_valid', False)
        for label, path in [('original', run / 'prepared/reference.png'), ('bbox', run / 'analysis.bbox.png')]:
            if path.exists():
                with Image.open(path) as image:
                    image.thumbnail((800, 800))
                    image.convert('RGB').save(run / (label + '.jpg'), quality=85)
        save(run / 'case-summary.json', row)
        return row

    with exclusive(output / '.batch.lock'):
        rows = []
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, case): case for case in cases}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                save(output / 'batch-summary.json', {'total': len(cases), 'processed': len(rows), 'cases': sorted(rows, key=lambda r:r['id'])})
                print('%d/%d %s %s %.1fs' % (len(rows), len(cases), row['status'], row['relative_path'], row['seconds']), flush=True)
        rows.sort(key=lambda row: row['id'])
        totals = {'total': len(rows), 'complete': sum(r['status'] == 'complete' for r in rows),
                  'failed': sum(r['status'] == 'failed' for r in rows),
                  'first_pass_valid': sum(bool(r['evidence'].get('first_pass_valid')) for r in rows),
                  'raw_schema_valid': sum(r['raw_schema_valid'] for r in rows),
                  'repaired_accepted': sum(bool(r['evidence'].get('repaired')) for r in rows),
                  'elapsed_seconds_this_invocation': round(time.monotonic() - started, 2),
                  'completion_tokens': sum(r['usage']['completion_tokens'] for r in rows),
                  'prompt_tokens': sum(r['usage']['prompt_tokens'] for r in rows)}
        kinds = Counter()
        for row in rows:
            kinds.update(row.get('kinds', {}))
        totals['object_kinds'] = dict(kinds)
        totals['background_types'] = dict(Counter(r['background_type'] for r in rows if 'background_type' in r))
        totals['interpretation'] = 'Schema/workflow validity only; this input directory has no reviewed ground truth.'
        save(output / 'batch-summary.json', {**totals, 'cases': rows})
        cards = []
        for row in rows:
            path = 'cases/' + row['id']
            pictures = ''.join('<a href="%s/%s"><img loading="lazy" src="%s/%s"></a>' % (path, img, path, img)
                               for img in ('original.jpg', 'bbox.jpg') if (output / path / img).exists())
            links = ' '.join('<a href="%s/%s">%s</a>' % (path, name, name) for name in ('structure.json', 'analysis.json', 'case-summary.json') if (output/path/name).exists())
            cards.append('<article><h3>%s — %s</h3>%s<p>%s</p><pre>%s</pre></article>' %
                         (html.escape(row['relative_path']), row['status'], pictures, links, html.escape(row.get('error', ''))))
        page = '<!doctype html><meta charset="utf-8"><title>Qwen3.5-9B covers analysis</title><style>body{font-family:system-ui;margin:24px;background:#f4f5f7}article{background:white;padding:16px;margin:20px 0}img{max-width:46%;max-height:600px;margin:1%}pre{white-space:pre-wrap}</style><h1>Qwen3.5-9B · covers</h1><pre>' + html.escape(json.dumps(totals, ensure_ascii=False, indent=2)) + '</pre>' + ''.join(cards)
        page = page.replace('Qwen3.5-9B', args.model)
        (output / 'index.html').write_text(page, encoding='utf-8')
        print(json.dumps(totals, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
