
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from collage.paths import relocated_path, relocated_values


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return relocated_values(json.loads(Path(relocated_path(path)).read_text(encoding='utf-8-sig')))
    except FileNotFoundError:
        return {} if default is None else default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(relocated_path(path)).read_bytes()).hexdigest()


@contextmanager
def exclusive(path):
    """OS-released lock: a crashed worker never needs a stale-lock deletion race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('已有工作进程正在处理此任务') from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('已有工作进程正在处理此任务') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
