"""Small deterministic IO helpers. No model calls."""
import hashlib
import json
import os
import time
import uuid
import sys
from threading import Lock
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from collage.paths import relocated_path, relocated_values

_cache_names_lock = Lock()


def cache_directory(root, prefix, identity, manifest, *, create=True):
    """Use short serial names; full request metadata determines cache reuse."""
    root = Path(root)
    with _cache_names_lock:
        # Include old hash-named directories so an upgrade cannot resubmit a request.
        for folder in sorted(root.glob(prefix+'_*')):
            record = folder/manifest
            if record.is_file() and read(record) == identity:
                return folder
        if not create:
            return None
        root.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            folder = root/f'{prefix}_{number:02d}'
            try:
                folder.mkdir()
            except FileExistsError:
                number += 1
                continue
            save(folder/manifest, identity)
            return folder


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return relocated_values(json.loads(Path(relocated_path(path)).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f'Invalid number: {s}'))))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(relocated_path(path)).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def event(run, name, **data):
    from observation import emit
    emit(run, name, **data)


@contextmanager
def timed(run, name):
    from observation import operation
    started = time.monotonic()
    with operation(run, name):
        event(run, name + '_started')
        try:
            yield
        except Exception as exc:
            event(run, name + '_failed', elapsed_seconds=round(time.monotonic()-started, 3), error=str(exc))
            raise
        else:
            event(run, name + '_finished', elapsed_seconds=round(time.monotonic()-started, 3))


@contextmanager
def locked(run):
    """Only one writer per task; never silently remove a stale lock."""
    path = Path(run) / '.write.lock'
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps({'pid': os.getpid(), 'at': now(), 'request_id':os.environ.get('COLLAGE_REQUEST_ID')}))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def verify_source(record):
    p = Path(relocated_path(record['file']))
    if not p.is_file() or sha(p) != record['sha256']:
        raise ValueError(f'Source changed or missing: {p}')
    return p


def inside(run, relative):
    root = Path(run).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Output escapes task directory')
    return path
