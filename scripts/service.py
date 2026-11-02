"""One-shot Linux service control; no supervisor, cron, or background restart loop."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'service-model.json'
STATE = ROOT / 'logs/service.json'
MODELS = ('Qwen3.5-4B', 'Qwen3.5-9B')

def config():
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {
        'model': 'Qwen3.5-9B', 'gpu': '0', 'port': 8000}

def identity(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None

def healthy(cfg):
    try:
        base = 'http://127.0.0.1:%d' % cfg['port']
        with urlopen(base + '/health', timeout=3) as r:
            if r.status != 200:
                return False
        with urlopen(base + '/v1/models', timeout=3) as r:
            return any(x['id'] == cfg['model'] for x in json.load(r)['data'])
    except Exception:
        return False

def control(action, new_config=None):
    STATE.parent.mkdir(exist_ok=True)
    with (STATE.parent/'service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg = config()
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        pid = state.get('pid')
        owned = bool(pid and identity(pid) == state.get('start_ticks'))
        if owned:
            try:
                owned = os.getpgid(pid) == pid
            except ProcessLookupError:
                owned = False
        ready = healthy(cfg)
        if ready and owned:
            state['was_ready'] = True
            STATE.write_text(json.dumps(state))
        if action == 'status':
            return {**state, **cfg, 'status': 'ready' if ready else 'starting' if owned else 'stopped'}
        if action == 'configure':
            if owned or ready:
                raise RuntimeError('Stop the current service before switching models.')
            CONFIG.write_text(json.dumps(new_config, indent=2))
            return {'status': 'configured', **new_config}
        if action == 'stop':
            if ready and not owned:
                raise RuntimeError('Refusing to stop a service not launched by this project.')
        elif ready:
            return {'status': 'ready', **cfg}
        if action == 'restart' and owned and not state.get('was_ready') and time.time()-state.get('started_at', 0)<900:
            return {**state, **cfg, 'status': 'starting'}
        if action in ('stop', 'restart') and owned:
            os.killpg(pid, signal.SIGTERM)
            for _ in range(50):
                if identity(pid) != state['start_ticks']:
                    break
                time.sleep(.2)
            if identity(pid) == state['start_ticks']:
                os.killpg(pid, signal.SIGKILL)
            owned = False
        if action == 'stop':
            STATE.unlink(missing_ok=True)
            return {'status': 'stopped', **cfg}
        if owned:
            return {**state, **cfg, 'status': 'starting'}
        manifest = ROOT / 'models' / cfg['model'] / 'download-manifest.json'
        if not manifest.exists():
            raise RuntimeError('Download and verify the selected model first.')
        if not (ROOT/'.venv/bin/vllm').is_file():
            raise RuntimeError('Run bash scripts/setup.sh first.')
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', cfg['port'])) == 0:
                raise RuntimeError('Port is occupied; refusing duplicate startup.')
        env = dict(os.environ, GPU_ID=str(cfg['gpu']), PORT=str(cfg['port']), MODEL_NAME=cfg['model'])
        with (ROOT/'logs/server.log').open('ab') as out:
            proc = subprocess.Popen(['bash', str(ROOT/'scripts/serve.sh')], cwd=ROOT,
                env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True)
        state = {'pid':proc.pid, 'start_ticks':identity(proc.pid), 'started_at':time.time(), **cfg}
        STATE.write_text(json.dumps(state))
        return {**state, 'status':'starting'}

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['configure','ensure','status','restart','stop'])
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--gpu', default='0', help='Single physical GPU index, default 0')
    p.add_argument('--port', type=int, default=8000)
    a=p.parse_args()
    if a.action=='configure' and not a.model:
        p.error('configure requires --model')
    if not a.gpu.isdigit() or not 1 <= a.port <= 65535:
        p.error('Use a single nonnegative GPU index and a valid port')
    try:
        print(json.dumps(control(a.action, {'model':a.model,'gpu':a.gpu,'port':a.port})))
    except Exception as e:
        print(json.dumps({'status':'error','error':str(e)}))
        raise SystemExit(1)
