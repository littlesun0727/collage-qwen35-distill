"""Standalone entry to the same analysis stage used by the collage workflow."""
from pathlib import Path
from PIL import Image, ImageOps
from collage import contracts, prompts
from collage.config import Config
from collage.storage import exclusive, file_sha, read, save
from collage.vlm.client import Client
from collage.vlm.coordinates import resolve_mode
from collage.workflow.stages import analyze


class AnalysisContext:
    def __init__(self, run, config, client=None):
        self.run = Path(run)
        self.control = self.run/'workflow'
        self.config = config
        self.client = client or Client(self.control/'vlm', config)

    def reference_context(self):
        return {'reference_size': read(self.run/'input.json')['reference_size']}

    def analysis_images(self):
        return [('参考图', self.run/'prepared/reference.png')]

    def candidates(self, stage):
        return self.client.candidates(stage) if hasattr(self.client, 'candidates') else []

    def progress(self, number, status, message):
        save(self.control/'analysis-status.json', {'status': status, 'message': message})

    def write_reference(self, bundle):
        contracts.validate_reference(bundle, self.run, geometry_warnings=[])
        save(self.run/'text-units.json', bundle['text_units'])
        save(self.run/'analysis.json', bundle['analysis'])
        self.progress(2, 'complete', '参考图分析完成')


def run_analysis(reference, run, config=None, client=None):
    run = Path(run).resolve(); reference = Path(reference).resolve()
    config = config or Config.load()
    run.mkdir(parents=True, exist_ok=True)
    with exclusive(run/'.workflow.lock'):
        mode = resolve_mode(config.for_stage('analyze').model, config.for_stage('analyze').coordinate_mode)
        identity = {'source_sha256': file_sha(reference), 'model_policy': config.model_policy(),
                    'prompt': str(prompts.render('analyze', mode=mode)), 'version': analyze.VERSION}
        snapshot = run/'analysis-input.json'
        if snapshot.exists() and read(snapshot) != identity:
            raise ValueError('Reference, prompt or analysis configuration changed; use a new run directory')
        if not snapshot.exists() and any((run/name).exists() for name in
                                        ('input.json', 'analysis.json', 'prepared/reference.png')):
            raise ValueError('Existing collage run: consume its analysis with decompose instead')
        save(snapshot, identity)
        image = run/'prepared/reference.png'; image.parent.mkdir(exist_ok=True)
        if not image.exists():
            with Image.open(reference) as original:
                ImageOps.exif_transpose(original).convert('RGB').save(image)
        with Image.open(image) as im:
            size = list(im.size)
        receipt = {'reference_size': size, 'reference_sha256': file_sha(image)}
        if (run/'input.json').exists() and read(run/'input.json') != receipt:
            raise ValueError('Prepared reference changed; use a new run directory')
        save(run/'input.json', receipt)
        if (run/'analysis.json').exists():
            bundle = {'analysis': read(run/'analysis.json'), 'text_units': read(run/'text-units.json')}
            contracts.validate_reference(bundle, run, geometry_warnings=[])
            return bundle['analysis']
        ctx = AnalysisContext(run, config, client)
        analyze.execute(ctx)
        return read(run/'analysis.json')
