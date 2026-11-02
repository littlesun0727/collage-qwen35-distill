from copy import deepcopy
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from http.server import ThreadingHTTPServer
from PIL import Image
import pytest

from annotation.diagnostics import CHECKS, diagnose
from annotation.server import Problem, Store, encoded, handler_for


def fixture_dataset(root):
    structure = {'coordinate_mode': 'normalized_1000', 'background': {'type': 'solid_color', 'color': '#FFFFFF'},
                 'objects': [
                     {'id': 'photo_a', 'kind': 'photo', 'bbox': [20, 30, 620, 700], 'description': '一张完整照片'},
                     {'id': 'sticker', 'kind': 'overlay', 'bbox': [450, 500, 800, 850], 'description': '一枚装饰贴纸'}],
                 'layer_order': ['photo_a', 'sticker']}
    cases = []
    for key in ('a', 'b'):
        folder = root/'cases'/key
        folder.mkdir(parents=True)
        (folder/'structure.json').write_bytes(encoded(structure))
        (folder/'teacher.structure.json').write_bytes(encoded(structure))
        (folder/'review.json').write_bytes(encoded({'id': key, 'status': 'pending', 'notes': '',
                                                  'checks': {c: False for c in CHECKS}}))
        Image.new('RGB', (640, 800), '#dedbd2').save(folder/'original.png')
        cases.append({'test_id': key, 'title': '测试 '+key, 'cover_file': key+'.png',
                      'image': f'cases/{key}/original.png', 'structure.json': {'path': f'cases/{key}/structure.json'},
                      'teacher.structure.json': {'path': f'cases/{key}/teacher.structure.json'}})
    (root/'manifest.json').write_bytes(encoded({'dataset': 'annotation-test', 'cases': cases}))
    (root/'occlusions.jsonl').write_text(''.join(json.dumps({'id': k, 'occlusion_pairs': None})+'\n' for k in ('a', 'b')), encoding='utf-8')
    return Store(root)


@pytest.fixture
def store(tmp_path):
    return fixture_dataset(tmp_path)


def payload(data):
    return deepcopy({k: data[k] for k in ('revision', 'structure', 'review', 'draft_pairs')})


def test_save_roundtrip_backup_and_immutable_teacher(store):
    original = store.get('a')
    p = payload(original)
    p['structure']['objects'][0]['bbox'][0] = 40
    p['draft_pairs'] = [['photo_a', 'sticker']]
    result = store.save('a', p)
    assert result['structure']['objects'][0]['bbox'][0] == 40
    assert result['review']['status'] == 'in_progress'
    assert result['occlusion_pairs'] is None  # partial relation review stays a draft
    assert result['draft_pairs'] == [['photo_a', 'sticker']]
    assert result['seed'] == original['seed']
    history = store.history('a')
    assert len(history) == 1
    assert store.history('a', history[0]['file'])['structure'] == original['structure']
    with pytest.raises(Problem, match='其他页面'):
        store.save('a', p)


def test_two_cases_do_not_overwrite_other_occlusions(store):
    a, b = payload(store.get('a')), payload(store.get('b'))
    for p in (a, b):
        p['review']['checks']['occlusion_pairs'] = True
        p['draft_pairs'] = [['photo_a', 'sticker']]
    store.save('a', a)
    store.save('b', b)
    assert all(row['occlusion_pairs'] == [['photo_a', 'sticker']] for row in store.occlusions())


def test_completion_requires_explicit_checks_and_rejects_bad_boxes(store):
    p = payload(store.get('a'))
    p['review']['status'] = 'reviewed'
    with pytest.raises(Problem, match='七项'):
        store.save('a', p)
    p['review']['checks'] = {c: True for c in CHECKS}
    p['structure']['objects'][0]['bbox'][0] = -1
    with pytest.raises(Problem, match='结构错误'):
        store.save('a', p)
    p['structure']['objects'][0]['bbox'][0] = 20
    result = store.save('a', p)
    assert result['review']['status'] == 'reviewed'
    assert result['occlusion_pairs'] == []


def test_pair_order_and_path_boundaries(store):
    p = payload(store.get('a'))
    p['draft_pairs'] = [['sticker', 'photo_a']]
    with pytest.raises(Problem, match='遮挡关系'):
        store.save('a', p)
    with pytest.raises(Problem):
        store.contained('../outside.json')
    with pytest.raises(Problem):
        store.get('../a')
    with pytest.raises(Problem):
        store.history('a', '../structure.json')


def test_write_failure_rolls_back_all_annotation_files(store, monkeypatch):
    from annotation import server
    p = payload(store.get('a'))
    p['structure']['objects'][0]['bbox'][0] = 99
    before = {f: f.read_bytes() for f in store.files('a')}
    real_atomic, count = server.atomic, 0
    def fail_once(path, data):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError('simulated disk error')
        real_atomic(path, data)
    monkeypatch.setattr(server, 'atomic', fail_once)
    with pytest.raises(OSError):
        store.save('a', p)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_duplicate_issue_is_a_hypothesis_and_acknowledgement_expires(store):
    s = store.get('a')['structure']
    s['objects'][1] = dict(s['objects'][0], id='photo_b')
    s['layer_order'] = ['photo_a', 'photo_b']
    issue = next(i for i in diagnose(s) if i['code']=='near-duplicate')
    assert issue['severity']=='warning'
    review = {'acknowledged_issues': [issue['key']]}
    assert next(i for i in diagnose(s, review) if i['code']=='near-duplicate')['acknowledged']
    s['objects'][0]['bbox'][0] += 1
    assert not next(i for i in diagnose(s, review) if i['code']=='near-duplicate')['acknowledged']


def test_http_reads_and_rejects_cross_origin_writes(store):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/cases') as r:
            assert len(json.load(r)['cases'])==2
        request = Request(base+'/api/cases/a/save', encoded(payload(store.get('a'))),
                          {'Content-Type': 'application/json', 'Origin': 'https://example.com'})
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 403
        with urlopen(base+'/api/cases/a/image') as r:
            assert r.headers['Content-Type']=='image/png'
    finally:
        server.shutdown()
        server.server_close()
