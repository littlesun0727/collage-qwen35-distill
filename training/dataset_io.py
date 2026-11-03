"""Read portable structure SFT JSONL without rewriting prompts or target strings."""
import copy
import hashlib
import json
import os
from pathlib import Path

from prepare_smoke import normalize_structure

DATASET_ID = 'congerjing/collage-structure-sft-v1'
DATASET_REVISION = '634239ab2347fe36b807f6a143df6171ef697f0f'
CORE_KEYS = {'background', 'objects', 'layer_order'}


def validate_target(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f'Duplicate JSON key: {key}')
            value[key] = item
        return value
    target = json.loads(text, object_pairs_hook=unique)
    if not isinstance(target, dict) or set(target) not in (CORE_KEYS, CORE_KEYS | {'version', 'coordinate_mode'}):
        raise ValueError('Expected three core fields, optionally with version and coordinate_mode')
    if target.get('coordinate_mode', 'normalized_1000') != 'normalized_1000':
        raise ValueError('SFT targets must already use normalized_1000 coordinates')
    # Validate geometry/references without changing the original answer string.
    normalize_structure({'coordinate_mode': 'normalized_1000', **target}, (1000, 1000))
    return target


def load_split(root, split):
    root = Path(root).resolve()
    source = root / f'{split}.jsonl'
    content = source.read_bytes()
    checksum_file = root / 'checksums.json'
    if checksum_file.exists():
        checksums = json.loads(checksum_file.read_text(encoding='utf-8'))
        expected = checksums.get(source.name)
        if not expected or hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {source}')
    rows = []
    for n, line in enumerate(content.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row['messages']
        if not messages or messages[-1]['role'] != 'assistant':
            raise ValueError(f'{split}:{n}: expected final assistant answer')
        validate_target(messages[-1]['content'])
        if len(row['images']) != 1:
            raise ValueError(f'{split}:{n}: expected one image')
        paths = []
        for name in row['images']:
            path = Path(name)
            lexical = Path(os.path.abspath(root / path))
            if not path.is_absolute() and root not in lexical.parents:
                raise ValueError(f'{split}:{n}: escaping relative image path')
            # HF cache snapshots legitimately symlink image files into ../blobs.
            resolved = lexical.resolve()
            if not resolved.is_file():
                raise ValueError(f'Missing image: {resolved}; download the full dataset snapshot')
            paths.append(str(resolved))
        rows.append({'messages': messages, 'images': paths})
    if not rows:
        raise ValueError(f'Empty split: {split}')
    return rows


def inputs_only(row):
    result = copy.deepcopy(row)
    result['messages'] = result['messages'][:-1]
    if any(m['role'] == 'assistant' for m in result['messages']):
        raise ValueError('Preview expects single-turn samples without prior assistant answers')
    return result


def assert_split_isolation(train, validation):
    train_images = {im for row in train for im in row['images']}
    validation_images = {im for row in validation for im in row['images']}
    if train_images & validation_images:
        raise ValueError('Training and validation contain overlapping image paths')


def download_snapshot(repo_id=DATASET_ID, revision=None, local_dir=None):
    from huggingface_hub import HfApi, snapshot_download
    revision = revision or (DATASET_REVISION if repo_id == DATASET_ID else 'main')
    # Resolve moving refs once; train and validation must use exactly the same snapshot.
    sha = HfApi().dataset_info(repo_id, revision=revision).sha
    path = snapshot_download(repo_id, repo_type='dataset', revision=sha, local_dir=local_dir)
    return Path(path), {'repo_id': repo_id, 'revision': sha}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Download the complete pinned HF dataset for offline training.')
    p.add_argument('--repo', default=DATASET_ID)
    p.add_argument('--revision')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    path, provenance = download_snapshot(args.repo, args.revision, str(args.output))
    (path / 'download-source.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('Downloaded:', path, provenance)
