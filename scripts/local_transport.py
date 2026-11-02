"""Local HTTP inference, with bounded request-triggered service recovery."""
import json
from http.client import HTTPException
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

def control(action):
    result = subprocess.run([sys.executable, str(ROOT/'scripts/service.py'), action],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stdout or result.stderr)
    return json.loads(result.stdout)

def ready(model, timeout=900):
    state = control('status')
    if state['model'] != model:
        raise ValueError('Configure the service for %s first; current model is %s' % (model,state['model']))
    deadline=time.monotonic()+timeout
    if state['status']=='starting' and (state.get('was_ready') or time.time()-state.get('started_at',0)>timeout):
        state=control('restart')
    while state['status']!='ready':
        if time.monotonic()>deadline:
            raise TimeoutError('Startup timed out; inspect logs/server.log')
        state=control('ensure')
        if state['status']=='ready':
            break
        time.sleep(5)
        state=control('status')
    return 'http://127.0.0.1:%s/v1/chat/completions' % state['port']

def infer(body, retries=2):
    last=None
    for attempt in range(retries+1):
        url=ready(body['model'])
        try:
            request=Request(url, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=600) as response:
                return json.load(response)
        except HTTPError as e:
            last='HTTP %d: %s' % (e.code,e.read(2048).decode(errors='replace'))
            if e.code not in (408,429) and e.code<500:
                raise RuntimeError(last) from e
        except (URLError,OSError,HTTPException,json.JSONDecodeError) as e:
            last=str(e)
        if attempt<retries:
            time.sleep(5)
    raise RuntimeError('Inference failed after bounded retries: '+str(last))
