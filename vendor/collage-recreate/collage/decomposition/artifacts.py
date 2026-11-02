"""Decomposition-only coordinate boundary and portable output artifacts."""
from copy import deepcopy
from pathlib import Path
import shutil
from PIL import Image, ImageDraw
from collage.storage import read, save, file_sha
from collage.vlm.coordinates import _records, _check_box


REQUIRED_DELIVERABLES = frozenset({'analysis.json', 'bbox.png', 'original.png', 'layers',
                          'reference.png', 'analysis-refined.json', 'reference.svg'})
DELIVERABLES = REQUIRED_DELIVERABLES | {'structure.json', 'structure-provenance.json'}


def complete_delivery(run):
    return all((run/name).exists() for name in REQUIRED_DELIVERABLES)


def structure_snapshot(analysis_path, analysis, size, mode, image, mapping):
    """Preserve a supplied/sibling model inventory; never reverse-engineer labels."""
    raw = read(analysis_path)
    path = analysis_path if raw.get('schema_version') != 'collage-analysis-v1' else analysis_path.with_name('structure.json')
    if not path.is_file():
        return None
    value = read(path)
    # A sibling structure must declare its own coordinates, not inherit an
    # override intended for the compiled analysis input.
    source_mode = mode if path == analysis_path else None
    compiled = load_analysis(value, size, source_mode)
    if pixel_document(compiled) != pixel_document(analysis):
        # Portable decomposition inputs use remapped extraction IDs, while
        # the preserved structure deliberately retains the original model IDs.
        from .pipeline import extraction_inventory
        remapped, original_mapping = extraction_inventory(compiled)
        receipt = path.with_name('structure-provenance.json')
        if (not receipt.is_file()
                or read(receipt).get('id_mapping_to_analysis') != original_mapping
                or pixel_document(remapped) != pixel_document(analysis)):
            raise ValueError('Structure does not compile to source analysis; use matching original analysis/structure')
        mapping = original_mapping
    snapshot = deepcopy(value)
    snapshot.setdefault('coordinate_mode', value.get('coordinate_space') or source_mode)
    provenance = {'source_structure_sha256': file_sha(path),
                  'reference_sha256': file_sha(image), 'reference': 'original.png',
                  'reference_size': list(size), 'id_mapping_to_analysis': mapping,
                  'annotation_status': 'unreviewed',
                  'note': 'Source inventory, not verified ground truth. Refined extraction geometry is separate.'}
    return snapshot, provenance


def write_structure_snapshot(run, snapshot):
    if snapshot is None:
        return
    value, provenance = snapshot
    for name, document in (('structure.json', value), ('structure-provenance.json', provenance)):
        if (run/name).exists() and read(run/name) != document:
            raise ValueError('Source structure or provenance changed; use a new run for decomposition')
    save(run/'structure.json', value)
    save(run/'structure-provenance.json', provenance)


def clean_delivery(run):
    """Discard intermediates after successful export; do not archive duplicates."""
    run = Path(run).resolve()
    if not complete_delivery(run):
        raise ValueError('Refusing cleanup before all deliverables exist')
    if not (run/'layers/aug').is_dir():
        raise ValueError('Missing aug layers; refusing cleanup')
    targets = [p for p in run.iterdir() if p.name not in DELIVERABLES]
    targets += [p for p in (run/'layers').iterdir() if p.name != 'aug']
    for target in targets:
        target.resolve().relative_to(run)
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


def load_analysis(value, size, mode=None):
    value = deepcopy(value)
    declared = value.pop('coordinate_mode', value.pop('coordinate_space', None))
    if mode and declared and mode != declared:
        raise ValueError('Conflicting analysis coordinate modes')
    mode = mode or declared
    if value.get('reference_size', size) != size:
        raise ValueError('Analysis dimensions differ from reference')
    if mode is None:
        if value.get('schema_version') == 'collage-analysis-v1':
            mode = 'pixel'
        else:
            raise ValueError('Specify --coordinate-mode for an analysis without coordinate metadata')
    if mode not in ('pixel', 'normalized_1000'):
        raise ValueError('Unsupported coordinate mode: ' + str(mode))
    if value.get('schema_version') != 'collage-analysis-v1':
        from collage.workflow.stages.analyze import compile_inventory
        value.pop('version', None)
        value.pop('reference_size', None)
        return compile_inventory(value, size, mode)['analysis']
    value['reference_size'] = size
    for key, row, name in _records(value):
        _check_box(row[name], [1000, 1000] if mode == 'normalized_1000' else size, str(key))
        if mode == 'normalized_1000':
            row[name] = [round(v * size[i % 2] / 1000) for i, v in enumerate(row[name])]
    return value


def pixel_document(analysis):
    """Convert declared normalized geometry once; preserve existing pixel boxes."""
    result = load_analysis(analysis, analysis['reference_size'])
    result['coordinate_mode'] = 'pixel'
    return result


def upgrade_delivery(run):
    """Migrate an existing seven-item delivery locally, without extraction caches."""
    from .export import write_svg
    prediction = read(run/'analysis.json')
    refined = read(run/'analysis-refined.json')
    for value, filename in ((prediction, 'original.png'), (refined, 'reference.png')):
        with Image.open(run/filename) as im:
            if list(im.size) != value['reference_size']:
                raise ValueError('Analysis dimensions differ from ' + filename)
    prediction = pixel_document(prediction)
    refined = pixel_document(refined)
    records = {o['id']: {'full_layer': f"layers/aug/{o['id']}.png"}
               for o in refined['objects']}
    write_svg(run, refined, records)
    save(run/'analysis.json', prediction)
    save(run/'analysis-refined.json', refined)


def write_prediction(run, source, image):
    save(run/'analysis.json', pixel_document(source))
    with Image.open(image) as im:
        im.save(run/'original.png')
        canvas = im.convert('RGB')
    pen = ImageDraw.Draw(canvas)
    for item in source['objects']:
        if item['kind'] == 'background':
            continue
        box = item['bbox']
        pen.rectangle(box, outline='#ff3050', width=max(2, canvas.width // 400))
        pen.text((box[0]+3, box[1]+3), item['id'], fill='#ff3050')
    canvas.save(run/'bbox.png')
