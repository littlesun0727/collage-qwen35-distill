"""Durable optional observation; production never depends on a web server."""
import json
import os
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from common import now, read, save, sha

_context = {}
_thread_lock = threading.RLock()
STAGES = {'prepare': 1, 'validate': 2, 'reveal-plan': 2, 'build': 3,
          'render': 3, 'screen': 4, 'apply': 5, 'review': 5, 'recover': 5, 'generate': 5,
          'revision-prepare':5,'revision-commit':5,'revision-recover':5}


def enabled():
    return os.environ.get('COLLAGE_OBSERVE', '1') != '0'


def warn(run, exc):
    message = 'Observation incomplete: ' + type(exc).__name__
    print(message, file=sys.stderr)
    try:
        save(Path(run)/'observability/warning.json', {'at': now(), 'message': message})
    except OSError:
        pass


@contextmanager
def event_lock(run):
    """OS locks release on process death, unlike a stale exclusive-create file."""
    p = Path(run)/'observability/events.lock'
    p.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock, p.open('a+b') as f:
        f.seek(0, 2)
        if not f.tell():
            f.write(b'0'); f.flush()
        deadline = time.monotonic() + 5
        while True:
            try:
                f.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Event writer busy')
                time.sleep(.02)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def emit(run, name, **data):
    if not enabled():
        return
    try:
        root = Path(run)
        row = {'schema_version': 'collage-events/1', 'event_id': uuid.uuid4().hex,
               'run_id': root.name, 'at': now(), 'event': name, 'source': 'script',
               **_context, **data}
        with event_lock(root):
            with (root/'events.jsonl').open('ab') as f:
                # A preceding interrupted append must not consume the next valid event.
                f.write(b'\n' + json.dumps(row, ensure_ascii=False).encode('utf-8') + b'\n')
                f.flush()
    except Exception as exc:
        warn(run, exc)


@contextmanager
def operation(run, command):
    previous = dict(_context)
    _context.update(attempt_id=uuid.uuid4().hex, command=command, stage=STAGES.get(command, 3))
    stop = threading.Event()
    def pulse():
        while not stop.is_set():
            try:
                save(Path(run)/'observability/heartbeat.json',
                     {'at': now(), 'pid': os.getpid(), 'active': True, **_context})
            except Exception as exc:
                warn(run, exc)
            stop.wait(3)
    thread = threading.Thread(target=pulse, daemon=True) if enabled() else None
    if thread:
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread:
            thread.join(timeout=4)
            try:
                save(Path(run)/'observability/heartbeat.json', {'at': now(), 'active': False, **_context})
            except Exception as exc:
                warn(run, exc)
        _context.clear(); _context.update(previous)


def evidence(run, path):
    """Copy operation intent before the input file can be reused for another round."""
    if not enabled():
        return
    try:
        out = Path(run)/'observability/operations'/(_context.get('attempt_id', uuid.uuid4().hex)+'.json')
        save(out, read(path))
        _context['evidence'] = str(out.relative_to(run)).replace('\\', '/')
    except Exception as exc:
        warn(run, exc)


def differences(before, after):
    old = {o['id']: o for o in before.get('objects', [])}
    changes = []
    keys = ('kind', 'bbox', 'rotation', 'style', 'binding', 'method', 'extracted_offset', 'gate', 'text', 'editor_transform', 'photo_window_spec')
    current={o['id'] for o in after.get('objects',[])}
    for oid,obj in old.items():
        if oid not in current:
            changes.append({'id':oid,'label':obj.get('label',oid),'bbox':obj.get('bbox'),
                            'fields':{'removed':{'before':True,'after':False}}})
    for obj in after.get('objects', []):
        delta = {k: {'before': old.get(obj['id'], {}).get(k), 'after': obj.get(k)}
                 for k in keys if old.get(obj['id'], {}).get(k) != obj.get(k)}
        if delta:
            changes.append({'id': obj['id'], 'label': obj.get('label', obj['id']),
                            'bbox': obj.get('bbox'), 'fields': delta})
    if before.get('layer_order') != after.get('layer_order'):
        changes.append({'id': 'layer_order', 'label': '图层前后顺序',
                        'fields': {'layer_order': {'before': before.get('layer_order'), 'after': after.get('layer_order')}}})
    return changes


def publish(run):
    """Publish complete immutable render evidence before the next command mutates it."""
    if not enabled():
        return
    try:
        run = Path(run); result = read(run/'result.json'); scene = read(run/'scene.json')
        if result.get('scene_sha256') != sha(run/'scene.json') or result.get('final_sha256') != sha(run/'final.png'):
            raise ValueError('Snapshot render mismatch')
        versions = run/'observability/versions'; versions.mkdir(parents=True, exist_ok=True)
        manifests = sorted(versions.glob('*/manifest.json'))
        previous = read(manifests[-1]) if manifests else {}
        if previous.get('render_id') == result['render_id']:
            emit(run, 'version_published', version_id=previous['version_id'], render_id=result['render_id'])
            return
        identifier = f'{len(manifests)+1:04d}-' + uuid.uuid4().hex[:8]
        staging = versions/('.pending-' + identifier); staging.mkdir()
        artifacts = {}
        def copy_file(key, path):
            path = Path(path)
            if not path.is_file():
                return
            target = staging/'evidence'/(f'{len(artifacts):04d}'+path.suffix.lower())
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            artifacts[key] = {'path': target.relative_to(staging).as_posix(), 'sha256': sha(target)}
        for name in ('final.png', 'scene.json', 'result.json'):
            shutil.copyfile(run/name, staging/name)
            artifacts[name] = {'path': name, 'sha256': sha(staging/name)}
        for name in ('analysis.json', 'bindings.json', 'photo-windows.json', 'previews/analysis-boxes.png', 'previews/photo-windows.png', 'previews/comparison.png'):
            copy_file(name, run/name)
        for record in result.get('resources', []):
            copy_file('material:'+record['id'], record['file'])
            meta=record.get('metadata', {})
            if meta.get('foreground'):
                copy_file('processed:'+record['id'], meta['foreground'])
            source=meta.get('source', {})
            if isinstance(source, dict) and source.get('raw_file'):
                copy_file('raw:'+record['id'], source['raw_file'])
        for i, path in enumerate(result.get('extraction_sheets', [])):
            copy_file('sheet:'+str(i), path)
        for path in (run/'previews').glob('review-detail-*.png'):
            copy_file(path.relative_to(run).as_posix(), path)
        intent = read(run/_context['evidence']) if _context.get('evidence') else {}
        if intent:
            save(staging/'intent.json', intent)
        old_scene = read(manifests[-1].parent/'scene.json') if manifests else {}
        manifest = {'schema_version': 'collage-versions/1', 'version_id': identifier,
                    'parent_version_id': previous.get('version_id'), 'render_id': result['render_id'],
                    'at': now(), 'command': _context.get('command', 'render'),
                    'attempt_id': _context.get('attempt_id'), 'artifacts': artifacts,
                    'changes': differences(old_scene, scene) if old_scene else [], 'intent': intent}
        save(staging/'manifest.json', manifest)
        staging.rename(versions/identifier)
        emit(run, 'version_published', version_id=identifier, render_id=result['render_id'])
    except Exception as exc:
        warn(run, exc)


def checkpoint(run, stage, status, summary, analysis_only=False):
    """Record actual agent work; require evidence for terminal stage declarations."""
    if not enabled():
        return {'recorded': False, 'reason': 'Observation disabled'}
    run = Path(run)
    if not (run/'input.json').is_file():
        raise ValueError('Prepare the task first')
    result = read(run/'result.json') if (run/'result.json').exists() else {}
    if status == 'complete' and stage == 2:
        from validate import analysis_check, bindings_check
        analysis = analysis_check(read(run/'analysis.json'), read(run/'input.json'))
        bindings_check(read(run/'bindings.json'), analysis, read(run/'prepared/catalog.json'))
        if not (run/'previews/analysis-boxes.png').exists():
            raise ValueError('Validate and inspect the boxes first')
    if status == 'complete' and stage == 6 and not analysis_only:
        if not result.get('visual_review') or not result.get('exported'):
            raise ValueError('Delivery needs a reviewed current render')
        if result['final_sha256'] != sha(run/'final.png') or result['scene_sha256'] != sha(run/'scene.json'):
            raise ValueError('Delivery render changed')
    artifacts = {}
    if status == 'complete' and stage == 2:
        folder = run/'observability/checkpoints'/uuid.uuid4().hex
        folder.mkdir(parents=True)
        for name in ('analysis.json', 'bindings.json', 'photo-windows.json', 'previews/analysis-boxes.png', 'previews/photo-windows.png'):
            if name in {'photo-windows.json','previews/photo-windows.png'} and not (run/name).is_file():
                continue  # Legacy tasks have no separate window stage.
            target = folder/Path(name).name; shutil.copyfile(run/name, target)
            artifacts[name] = target.relative_to(run).as_posix()
    controller=__import__('os').environ.get('COLLAGE_CONTROLLER')
    emit(run, 'workflow_stage' if controller else 'agent_stage', source=controller or 'agent', stage=stage, status=status, summary=summary,
         analysis_only=analysis_only, artifacts=artifacts, render_id=result.get('render_id'))
    return {'recorded': True, 'stage': stage, 'status': status}
