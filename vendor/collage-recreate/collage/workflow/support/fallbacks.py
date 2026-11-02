"""Deterministic recovery only; never invent model observations or image content."""

import math
from jsonschema import Draft202012Validator
from collage import contracts
from collage.storage import read


def simple_bundle(run):
    """A visible, editable fallback made only from verified customer pixels."""
    from common import verify_source
    from PIL import Image
    inp = read(run/'input.json')
    assets = read(run/'prepared/catalog.json')['assets']
    available = []
    for asset in assets:
        # Integrity mismatches are not quality failures and must not be ignored.
        path = verify_source(asset)
        try:
            with Image.open(path) as image:
                image.verify()
        except OSError:
            continue
        available.append(asset)
    if not available:
        raise ValueError('没有可用的客户素材，无法生成简化拼图')
    available = available[:6]
    w, h = inp['reference_size']
    cols = 1 if len(available) == 1 else 2
    rows = math.ceil(len(available)/cols)
    gap = max(1, round(min(w, h)*.025))
    objects = [{'id': 'draft_background', 'kind': 'background', 'background_type': 'solid_color',
                'method': 'local', 'bbox': [0, 0, w, h], 'label': '简化草稿底色',
                'description': '分析不可用时的中性设计底图', 'style': {'fill': '#F4F1EB'}}]
    photos = []
    for i, asset in enumerate(available):
        col, row = i % cols, i // cols
        oid = f'draft_photo_{i+1}'
        bbox = [round(col*w/cols)+gap, round(row*h/rows)+gap,
                round((col+1)*w/cols)-gap, round((row+1)*h/rows)-gap]
        objects.append({'id': oid, 'kind': 'photo', 'method': 'local', 'mode': 'cover',
                        'bbox': bbox, 'label': '客户照片', 'description': '简化拼图中的真实客户素材'})
        photos.append({'slot_id': oid, 'asset_id': asset['id'], 'reason': '简化草稿按素材目录顺序绑定'})
    bundle = {'analysis': {'schema_version': 'collage-analysis-v1', 'reference_size': [w, h],
                           'objects': objects, 'layer_order': [o['id'] for o in objects]},
              'bindings': {'schema_version': 'collage-bindings-v1', 'photos': photos, 'texts': []},
              'text_units': []}
    Draft202012Validator(contracts.BUNDLE).validate(bundle)
    contracts.validate_bundle(bundle, run)
    return bundle
