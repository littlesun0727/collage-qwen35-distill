"""External runtime storage and read-only relocation of historical file paths."""

from functools import lru_cache
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get('COLLAGE_OUTPUT_ROOT', str(ROOT.with_name(ROOT.name + '-output')))).resolve()
RUNTIME_ROOT = OUTPUT_ROOT / 'runtime'


@lru_cache(maxsize=8)
def _load_mappings(path, modified):
    records = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    pairs = [(row['from'].replace('\\', '/').rstrip('/'),
              row['to'].replace('\\', '/').rstrip('/')) for row in records]
    return tuple(sorted(pairs, key=lambda pair: len(pair[0]), reverse=True))


def _mappings():
    registry = OUTPUT_ROOT / 'path-relocations.json'
    try:
        modified = registry.stat().st_mtime_ns
    except FileNotFoundError:
        return ()
    return _load_mappings(str(registry), modified)


def _replace(source, mappings):
    normalized = source.replace('\\', '/')
    for old, new in mappings:
        if normalized.casefold() == old.casefold():
            return new
        if normalized.casefold().startswith(old.casefold() + '/'):
            return new + normalized[len(old):]
    return source


def relocated_path(value):
    """Resolve moved files without rewriting signed/hash-checked task records."""
    return _replace(str(value), _mappings())


def relocated_values(value):
    mappings = _mappings()
    if not mappings:
        return value

    def visit(item):
        if isinstance(item, str):
            return _replace(item, mappings)
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()}
        return item

    return visit(value)
