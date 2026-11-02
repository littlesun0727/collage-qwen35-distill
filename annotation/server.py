"""Local annotation server. Start: python -m annotation.server --dataset PATH"""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import threading
from urllib.parse import unquote, urlparse
import uuid

from PIL import Image
from eval.schema import parse_structure, strict_loads, validate_occlusions
from .diagnostics import CHECKS, candidates, diagnose

ASSETS = Path(__file__).with_name('static')


class Problem(Exception):
    def __init__(self, message, status=400):
        self.status = status
        super().__init__(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def read_json(path, default=None):
    return strict_loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def atomic(path, data):
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root/'manifest.json')
        self.entries = {e['test_id']: e for e in self.manifest['cases']}
        if len(self.entries) != len(self.manifest['cases']):
            raise ValueError('Duplicate sample IDs')
        self.lock = threading.RLock()
        self.thumbnails = {}

    def contained(self, path):
        p = (self.root/path).resolve()
        if not p.is_relative_to(self.root):
            raise Problem('路径超出标注目录', 403)
        return p

    def entry(self, key):
        if key not in self.entries:
            raise Problem('找不到这张图片', 404)
        return self.entries[key]

    def files(self, key):
        e = self.entry(key)
        gt = self.contained(e['structure.json']['path'])
        return gt, gt.with_name('review.json'), self.root/'occlusions.jsonl'

    def occlusions(self):
        p = self.root/'occlusions.jsonl'
        rows = [strict_loads(line) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()] if p.exists() else []
        keys = [r['id'] for r in rows]
        if len(keys) != len(set(keys)):
            raise Problem('遮挡清单出现重复 ID，请先修复', 422)
        return rows

    def state(self, key):
        gt, rev, _ = self.files(key)
        structure = read_json(gt)
        review = read_json(rev, {'id': key, 'status': 'pending', 'notes': '', 'checks': {c: False for c in CHECKS}})
        pairs = next((r.get('occlusion_pairs') for r in self.occlusions() if r['id'] == key), None)
        state = {'structure': structure, 'review': review, 'occlusion_pairs': pairs}
        revision = hashlib.sha256(encoded(state)).hexdigest()
        return state, revision

    def get(self, key):
        with self.lock:
            e = self.entry(key)
            state, revision = self.state(key)
            seedref = e.get('teacher.structure.json') or e.get('assistant.structure.json')
            seed = read_json(self.contained(seedref['path'])) if seedref else None
            with Image.open(self.contained(e['image'])) as image:
                size = image.size
            draft = state['review'].get('occlusion_draft', []) if state['occlusion_pairs'] is None else state['occlusion_pairs']
            return dict(state, id=key, revision=revision, title=e.get('title', ''), file=e.get('cover_file', key),
                        width=size[0], height=size[1], seed=seed, draft_pairs=draft,
                        source='现场补充' if e.get('assistant.structure.json') else '教师初始结果',
                        issues=diagnose(state['structure'], state['review']), candidates=candidates(state['structure']))

    def index(self):
        with self.lock:
            result = []
            for key, e in self.entries.items():
                state, _ = self.state(key)
                issues = diagnose(state['structure'], state['review'])
                result.append({'id': key, 'title': e.get('title', ''), 'file': e.get('cover_file', key),
                               'status': state['review'].get('status', 'pending'),
                               'objects': len(state['structure'].get('objects', [])),
                               'errors': sum(i['severity']=='error' for i in issues),
                               'warnings': sum(i['severity']=='warning' and not i['acknowledged'] for i in issues),
                               'checks': sum(bool(v) for v in state['review'].get('checks', {}).values()),
                               'source': '现场补充' if e.get('assistant.structure.json') else '教师结果'})
            return {'dataset': self.manifest['dataset'], 'root': str(self.root), 'cases': result}

    def save(self, key, payload):
        with self.lock:
            before, revision = self.state(key)
            if payload.get('revision') != revision:
                raise Problem('文件已在其他页面或编辑器中改动。请先刷新并核对，当前草稿仍保留。', 409)
            structure, review = payload.get('structure'), payload.get('review')
            if not isinstance(structure, dict) or not isinstance(review, dict):
                raise Problem('缺少结构或审核信息', 422)
            parsed = parse_structure(structure)
            if not parsed.valid:
                raise Problem('请先修正结构错误：'+'；'.join(parsed.errors), 422)
            if structure.get('coordinate_mode', 'normalized_1000') != 'normalized_1000':
                raise Problem('本标注台只支持 normalized_1000 坐标', 422)
            if len(parsed.objects)>160:
                raise Problem('每张最多 160 个元素', 422)
            pairs = payload.get('draft_pairs', [])
            try:
                validate_occlusions(pairs, parsed)
            except ValueError as exc:
                raise Problem('遮挡关系与图层冲突：'+str(exc), 422)
            review = dict(review, id=key, occlusion_draft=pairs)
            checks = review.get('checks', {})
            if not isinstance(checks, dict) or any(not isinstance(checks.get(c, False), bool) for c in CHECKS):
                raise Problem('审核状态格式错误', 422)
            review['checks'] = {c: checks.get(c, False) for c in CHECKS}
            if review.get('status') == 'reviewed':
                if not all(review['checks'].values()) or any(i['severity']=='error' for i in diagnose(structure, review)):
                    raise Problem('请完成七项人工检查并修正结构错误，再标记审核完成', 422)
            else:
                review['status'] = 'in_progress'
            review['updated_at'] = datetime.now(timezone.utc).isoformat()
            pairs_out = pairs if review['checks']['occlusion_pairs'] else None
            rows = self.occlusions()
            for row in rows:
                if row['id']==key:
                    row['occlusion_pairs'] = pairs_out
                    break
            else:
                rows.append({'id': key, 'occlusion_pairs': pairs_out})
            files = self.files(key)
            updates = [encoded(structure), encoded(review), b''.join(encoded(r).replace(b'\n', b'')+b'\n' for r in rows)]
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:6]
            history = self.root/'.annotation-history'/key
            history.mkdir(parents=True, exist_ok=True)
            atomic(history/(stamp+'.json'), encoded(dict(before, revision=revision, saved_at=datetime.now(timezone.utc).isoformat())))
            previous = [p.read_bytes() if p.exists() else None for p in files]
            try:
                for p, data in zip(files, updates):
                    atomic(p, data)
            except Exception:
                for p, data in zip(files, previous):
                    if data is None:
                        p.unlink(missing_ok=True)
                    else:
                        atomic(p, data)
                raise
            return self.get(key)

    def history(self, key, stamp=None):
        self.entry(key)
        folder = self.root/'.annotation-history'/key
        if stamp:
            if not stamp.endswith('.json') or Path(stamp).name != stamp or '/' in stamp or '\\' in stamp:
                raise Problem('历史版本名称无效', 400)
            p = folder/stamp
            if not p.exists():
                raise Problem('历史版本不存在', 404)
            return read_json(p)
        return [{'file': p.name, 'saved_at': read_json(p).get('saved_at')} for p in sorted(folder.glob('*.json'), reverse=True)][:100]

    def image(self, key, thumb=False):
        path = self.contained(self.entry(key)['image'])
        if not thumb:
            return path.read_bytes(), mimetypes.guess_type(path)[0] or 'image/png'
        if key not in self.thumbnails:
            with Image.open(path) as image:
                image = image.convert('RGB')
                image.thumbnail((180, 180))
                buf = io.BytesIO()
                image.save(buf, 'JPEG', quality=78)
                self.thumbnails[key] = buf.getvalue()
        return self.thumbnails[key], 'image/jpeg'


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data, status=200, content_type='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = encoded(data)
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def dispatch(self):
            origin = self.headers.get('Origin')
            if origin and origin != 'http://'+self.headers.get('Host', ''):
                raise Problem('只允许从本地标注页面访问', 403)
            path = [unquote(p) for p in urlparse(self.path).path.split('/') if p]
            if self.command=='GET':
                if not path or path in [['app.js'], ['style.css']]:
                    name = path[0] if path else 'index.html'
                    return self.reply((ASSETS/name).read_bytes(), content_type=mimetypes.guess_type(name)[0]+'; charset=utf-8')
                if path == ['api', 'cases']:
                    return self.reply(store.index())
                if path == ['api', 'health']:
                    return self.reply({'ok': True, 'dataset': str(store.root)})
                if len(path)>=3 and path[:2]==['api', 'cases']:
                    key = path[2]
                    if len(path)==3:
                        return self.reply(store.get(key))
                    if len(path)==4 and path[3] in ('image', 'thumb'):
                        data, mime = store.image(key, path[3]=='thumb')
                        return self.reply(data, content_type=mime)
                    if path[3]=='history' and len(path) in (4, 5):
                        return self.reply(store.history(key, path[4] if len(path)==5 else None))
            if self.command=='POST' and len(path)==4 and path[:2]==['api', 'cases']:
                if self.headers.get_content_type()!='application/json':
                    raise Problem('请求必须使用 JSON', 415)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2_000_000:
                    raise Problem('请求大小无效', 413)
                data = strict_loads(self.rfile.read(length).decode('utf-8'))
                if path[3]=='save':
                    return self.reply(store.save(path[2], data))
                if path[3]=='check':
                    store.entry(path[2])
                    return self.reply({'issues': diagnose(data['structure'], data.get('review')), 'candidates': candidates(data['structure'])})
            raise Problem('找不到此页面', 404)

        def do_GET(self):
            try:
                self.dispatch()
            except ConnectionError:
                pass  # A tab may close after a successful save or image request.
            except Problem as exc:
                self.reply({'error': str(exc)}, exc.status)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply({'error': '数据格式错误：'+str(exc)}, 422)
            except Exception as exc:
                self.reply({'error': '操作未完成：'+str(exc)}, 500)

        do_POST = do_GET
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(r'data/annotations'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(Store(args.dataset)))
    print(f'Collage review: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
