"""Analyze independent images, stopping before decomposition or collage creation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont

from collage.analysis import run_analysis
from collage.config import Config
from collage.storage import exclusive, now, save
from collage.vlm.client import Client
from collage.vlm.codex.shared_transport import SharedTransport

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def write_analysis_bbox(run, analysis):
    """Local visualization only; preserve the compiled analysis and original dimensions."""
    with Image.open(run/'prepared/reference.png') as original:
        canvas = original.convert('RGB')
    pen = ImageDraw.Draw(canvas)
    width = max(2, canvas.width // 400)
    font = ImageFont.load_default(size=max(12, canvas.width // 70))
    colors = {'photo': '#008cff', 'overlay': '#ff3050', 'frame': '#00b870', 'text': '#b000ff'}
    for item in analysis['objects']:
        if item['kind'] == 'background':
            continue
        box = item['bbox']
        color = colors.get(item['kind'], '#ff3050')
        pen.rectangle(box, outline=color, width=width)
        label = item['id']
        position = (box[0] + width, box[1] + width)
        pen.rectangle(pen.textbbox(position, label, font=font), fill='white')
        pen.text(position, label, font=font, fill=color)
    canvas.save(run/'analysis.bbox.png')


def discover(input_dir, output_dir, recursive=False):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError('Input directory does not exist')
    if output == source or output.is_relative_to(source):
        raise ValueError('Output directory must be outside the input directory')
    images = sorted((p for p in (source.rglob('*') if recursive else source.iterdir())
                     if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                    key=lambda p: p.relative_to(source).as_posix())
    if not images:
        raise ValueError('Input directory contains no supported images')
    cases = []
    for image in images:
        relative = image.relative_to(source).as_posix()
        # Full relative name (including extension) prevents same-stem collisions.
        digest = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:16]
        label = re.sub(r'[^a-zA-Z0-9_-]+', '_', image.stem)[:48] or 'image'
        cases.append({'id': f'case-{label}-{digest}', 'reference': str(image.resolve()),
                      'relative_path': relative})
    if len({case['id'].casefold() for case in cases}) != len(cases):
        raise ValueError('Case directory collision')
    return source, output, cases


def run_analysis_batch(input_dir, output_dir, config=None, concurrency=10, recursive=False,
                       *, transport_factory=SharedTransport):
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError('concurrency must be a positive integer')
    config = config or Config.load()
    stage = config.for_stage('analyze')
    if stage.provider != 'codex':
        raise ValueError('analyze-batch requires a Codex analyze model policy')
    source, output, cases = discover(input_dir, output_dir, recursive)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output/'.analysis-batch.lock'):
        summary = {'version': 1, 'status': 'running', 'started_at': now(),
                   'input_dir': str(source), 'concurrency': concurrency,
                   'model_policy': config.model_policy(), 'cases': []}
        for case in cases:
            run = output/'cases'/case['id']
            summary['cases'].append({**case, 'run': str(run),
                                     'analysis': str(run/'analysis.json'),
                                     'bbox': str(run/'analysis.bbox.png'), 'status': 'pending'})
        save(output/'batch-summary.json', summary)

        def process(row, transport):
            run = Path(row['run'])
            reused = (run/'analysis.json').is_file()
            try:
                # The mutable Client, budgets, caches and run lock are per case.
                client = Client(run/'workflow/vlm', config, transport=transport)
                analysis = run_analysis(row['reference'], run, config, client)
                write_analysis_bbox(run, analysis)
                return {'status': 'reused' if reused else 'complete', 'objects': len(analysis['objects'])}
            except Exception as error:
                return {'status': 'failed', 'error': str(error)[:1800]}

        with transport_factory(timeout=stage.timeout) as transport:
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='analysis-case') as pool:
                futures = {pool.submit(process, row, transport): row for row in summary['cases']}
                for future in as_completed(futures):
                    futures[future].update(future.result(), finished_at=now())
                    # Only the coordinator writes this shared file.
                    save(output/'batch-summary.json', summary)
        summary['failed'] = sum(row['status'] == 'failed' for row in summary['cases'])
        summary['succeeded'] = len(cases) - summary['failed']
        summary['status'] = 'partial_failure' if summary['failed'] else 'complete'
        summary['finished_at'] = now()
        save(output/'batch-summary.json', summary)
        return summary
