"""Check/fix the Transformers 5.5.3 Qwen3.5 visual-key save regression.

Creates a separate model directory. Tensor payload bytes and source files are
unchanged; this does not merge adapters or repair genuinely missing weights.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import tempfile

BAD = 'model.language_model.visual.'
GOOD = 'model.visual.'


def renamed(key):
    return GOOD + key[len(BAD):] if key.startswith(BAD) else key


def remap(mapping):
    result = {}
    for key, value in mapping.items():
        new = renamed(key)
        if new in result:
            raise ValueError(f'Weight name collision: {new}')
        result[new] = value
    return result


def read_header(path):
    # Let the installed safetensors reader validate offsets and file integrity.
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as tensors:
        keys = set(tensors.keys())
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
    return header, 8 + length, keys


def inspect(source):
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'qwen3_5':
        raise ValueError('Only a full Qwen3.5 multimodal model is supported (model_type=qwen3_5).')
    shards = sorted(source.glob('*.safetensors'))
    if not shards or (source / 'adapter_config.json').exists():
        raise ValueError('Use the full checkpoint-xxx-merged directory, not the LoRA adapter directory.')
    plans, all_keys, weights = {}, set(), {}
    for shard in shards:
        header, offset, keys = read_header(shard)
        if all_keys.intersection(keys):
            raise ValueError('Duplicate weights across shards.')
        all_keys.update(keys)
        weights.update({key: shard.name for key in keys})
        plans[shard.name] = (remap(header), offset)
    remap(weights)  # Also reject collisions spanning different shards.
    index_path = source / 'model.safetensors.index.json'
    index = None
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        if index.get('weight_map') != weights:
            raise ValueError('Checkpoint index and actual shard keys differ; refusing automatic repair.')
        index['weight_map'] = remap(weights)
    elif len(shards) != 1:
        raise ValueError('Multi-shard model is missing model.safetensors.index.json.')
    bad = sorted(key for key in all_keys if key.startswith(BAD))
    print(f'Checked {len(shards)} shards, {len(all_keys)} tensors; bad visual keys: {len(bad)}', flush=True)
    for key in bad[:3]:
        print(f'  {key} -> {renamed(key)}', flush=True)
    return plans, index, len(bad)


def repair(source, output=None):
    source = Path(source).resolve()
    plans, index, count = inspect(source)
    if not count:
        print('Known visual-prefix bug not found. No files changed.', flush=True)
        return False
    if output is None:
        print('Check only. Supply --output with a NEW directory to create a repaired copy.', flush=True)
        return True
    output = Path(output).resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError('Output must be a new directory outside the source model directory.')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    print(f'Writing separate copy to {stage} (requires space for a full model).', flush=True)
    try:
        for item in source.iterdir():
            target = stage / item.name
            if item.name in plans:
                header, offset = plans[item.name]
                encoded = json.dumps(header, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                encoded += b' ' * (-len(encoded) % 8)
                with item.open('rb') as src, target.open('wb') as dst:
                    dst.write(struct.pack('<Q', len(encoded)))
                    dst.write(encoded)
                    src.seek(offset)
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                read_header(target)
                print(f'Copied tensor payload unchanged: {item.name}', flush=True)
            elif item.name == 'model.safetensors.index.json' and index is not None:
                target.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            elif item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        _, _, remaining = inspect(stage)
        if remaining:
            raise ValueError('Repaired copy still contains bad keys.')
        stage.rename(output)
    except BaseException:
        print(f'Repair incomplete; original untouched. Partial copy retained at: {stage}', flush=True)
        raise
    print(f'Repaired model: {output}\nEvaluate this as --model with --preview; do not apply the adapter again.', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New directory; omit for read-only inspection')
    args = parser.parse_args()
    repair(args.source, args.output)


if __name__ == '__main__':
    main()
