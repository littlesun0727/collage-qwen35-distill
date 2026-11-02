"""Consume shared analysis without modifying it or choosing customer assets."""
from copy import deepcopy
from pathlib import Path
import shutil
from PIL import Image
from jsonschema import Draft202012Validator
from collage import contracts
from collage.config import ROOT
from collage.storage import exclusive, file_sha, read, save
from .chain import plan_batches, extract_chain, finalize_dataset
from .export import export_dataset
from .artifacts import load_analysis, write_prediction, pixel_document, complete_delivery, clean_delivery, upgrade_delivery
from .artifacts import structure_snapshot, write_structure_snapshot


def extraction_inventory(analysis):
    Draft202012Validator(contracts.ANALYSIS).validate(analysis)
    from validate import analysis_check
    analysis_check(analysis)
    backgrounds = [o for o in analysis['objects'] if o['kind'] == 'background']
    if len(backgrounds) != 1:
        raise ValueError('Decomposition requires exactly one compiled canvas background')
    bg = backgrounds[0]
    if bg['bbox'] != [0, 0, *analysis['reference_size']] or analysis['layer_order'][0] != bg['id']:
        raise ValueError('Canvas background must fill the image and be bottommost')
    result = deepcopy(analysis)
    used = {o['id'] for o in result['objects']}
    mapping = {bg['id']: 'layer_0'}
    if 'layer_0' in used and bg['id'] != 'layer_0':
        candidate = 'source_layer_0'
        while candidate in used:
            candidate += '_'
        mapping['layer_0'] = candidate
    for item in result['objects']:
        if item.get('rotation', 0):
            raise ValueError('Use the original analysis before photo-window rotation: '+item['id'])
        item['id'] = mapping.get(item['id'], item['id'])
        for field in ('linked_to', 'parent_id', 'photo_id'):
            if field in item:
                value = item[field]
                item[field] = ([mapping.get(v, v) for v in value] if isinstance(value, list)
                               else mapping.get(value, value))
    result['layer_order'] = [mapping.get(oid, oid) for oid in result['layer_order']]
    return result, mapping


def run_decomposition(run, extract=False, key_file=None, timeout=900, batch_size=20,
                      analysis_path=None, reference=None, coordinate_mode=None, export_only=False,
                      resume_extraction=False):
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with exclusive(run/'.workflow.lock'):
        analysis_path = Path(analysis_path).resolve() if analysis_path else run/'analysis.json'
        image = Path(reference).resolve() if reference else run/'prepared/reference.png'
        if not analysis_path.is_file() or not image.is_file():
            raise ValueError('Run analyze first, or supply an existing v9 run with prepared/reference.png')
        with Image.open(image) as im:
            size = list(im.size)
        analysis = load_analysis(read(analysis_path), size, coordinate_mode)
        source, mapping = extraction_inventory(analysis)
        snapshot = structure_snapshot(analysis_path, analysis, size, coordinate_mode, image, mapping)
        with Image.open(image) as im:
            if list(im.size) != source['reference_size']:
                raise ValueError('Analysis dimensions differ from prepared reference')
        output = run/'decomposition'; output.mkdir(exist_ok=True)
        if complete_delivery(output):
            if pixel_document(read(output/'analysis.json')) != pixel_document(source):
                raise ValueError('Source analysis changed; use a new run for decomposition')
            with Image.open(image) as original, Image.open(output/'original.png') as saved:
                if original.convert('RGBA').tobytes() != saved.convert('RGBA').tobytes():
                    raise ValueError('Source reference changed; use a new run for decomposition')
            write_structure_snapshot(output, snapshot)
            upgrade_delivery(output)
            clean_delivery(output)
            return {'status': 'exported' if export_only else 'existing', 'output': str(output)}
        identity = {'analysis_sha256': file_sha(analysis_path), 'reference_sha256': file_sha(image),
                    'batch_size': batch_size, 'id_mapping': mapping}
        if coordinate_mode:
            identity['coordinate_mode'] = coordinate_mode
        if (output/'input-receipt.json').exists() and read(output/'input-receipt.json') != identity:
            raise ValueError('Source analysis/reference changed; use a new run for decomposition')
        write_structure_snapshot(output, snapshot)
        save(output/'input-receipt.json', identity)
        save(output/'source-analysis.json', source)
        write_prediction(output, source, image)
        groups = plan_batches(source, batch_size)
        plan = {'batches_top_down': groups, 'foreground_count': sum(map(len, groups)),
                'background': {'id': 'layer_0', 'source': 'final_batch_residual'},
                'analysis': 'analysis.json', 'reference': 'original.png'}
        save(output/'plan.json', plan)
        if export_only:
            if not (output/'manifest.json').is_file():
                raise ValueError('--export-only requires existing extracted layers and manifest')
            export_dataset(output)
            finalize_dataset(output)
            clean_delivery(output)
            return {'status': 'exported', 'output': str(output)}
        if not extract:
            return {'status': 'planned', **plan}
        if (ROOT/'collage/decomposition/EXTRACTION_PAUSED.json').exists() and not resume_extraction:
            raise ValueError('Reveal extraction is paused; plan saved. Resume only after explicit authorization.')
        if not groups:
            raise ValueError('Background-only image: analysis and empty plan saved; no Reveal request needed')
        from reveal import api_key
        key = api_key(key_file)
        (output/'input').mkdir(exist_ok=True)
        shutil.copyfile(image, output/'input/reference.png')
        # Shared HTTP transport budgets gateway retries separately, up to three per batch.
        save(output/'submission-budget.json', {'limit': len(groups)*3})
        extract_chain(output, key, timeout, batch_size)
        export_dataset(output)
        finalize_dataset(output)
        dataset = read(output/'dataset.json')
        geometry_review = any(r.get('geometry_review', {}).get('review_reasons') for r in dataset['records'])
        result = {'status': 'needs_review' if dataset['empty_layers'] or geometry_review else 'complete',
                  'visual_review': 'pending', 'output': str(output)}
        clean_delivery(output)
        return result
