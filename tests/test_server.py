"""Web server: info, estimate, job lifecycle with live events, video download."""
import time

import pytest

pytest.importorskip('fastapi')
from tests.test_pipeline_e2e import tiny_store  # noqa: F401  (fixture)


def test_server_job_lifecycle(tiny_store, tmp_path, monkeypatch):
    monkeypatch.setenv('TINYH3_HOME', str(tmp_path / 'home'))
    from fastapi.testclient import TestClient
    from tinyh3.server.app import create_app
    client = TestClient(create_app(subprocess=False))
    with client:
        info = client.get('/api/info').json()
        assert '768p' in info['resolutions'] and list(info['samplers']) == ['quality', 'balance', 'fast', 'fast-high']
        assert info['upscale'] == ['none', '1080p', '2k']
        est = client.post('/api/estimate', json={'resolution': '480p', 'seconds': 5}).json()
        assert est['width'] == 864 and est['height'] == 480 and est['frames'] == 124
        est = client.post('/api/estimate', json={'resolution': '768p', 'seconds': 5, 'sampler': 'fast-high',
                                                 'upscale': '2k'}).json()
        assert est['render'] == [1024, 576] and (est['width'], est['height']) == (2520, 1440)
        assert client.post('/api/jobs', json={'prompt': '   '}).status_code == 400
        with client.websocket_connect('/api/events') as ws:
            job = client.post('/api/jobs', json={'prompt': 'w1 w2', 'resolution': '360p', 'aspect': '1:1',
                                                 'sampler': 'fast', 'seed': 9, 'model': str(tiny_store)}).json()
            seen = set()
            deadline = time.time() + 300
            while time.time() < deadline:
                e = ws.receive_json()
                if e.get('job') != job['id']:
                    continue
                seen.add(e.get('stage') or e.get('status'))
                if e.get('type') == 'status' and e['status'] in ('done', 'failed'):
                    break
        rec = client.get('/api/jobs').json()[0]
        assert rec['status'] == 'done', rec.get('error')
        assert {'text', 'denoise', 'decode'} <= seen
        r = client.get(f"/api/videos/{job['id']}")
        assert r.status_code == 200 and r.headers['content-type'] == 'video/mp4' and len(r.content) > 1000
        assert client.delete(f"/api/jobs/{job['id']}?remove_file=true").json() == {'ok': True}
        assert client.get('/api/jobs').json() == []


def _wait_done(client, job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = next(j for j in client.get('/api/jobs').json() if j['id'] == job_id)
        if rec['status'] in ('done', 'failed', 'cancelled'):
            return rec
        time.sleep(0.5)
    raise TimeoutError(job_id)


def test_server_tasks_and_uploads(tiny_store, tmp_path, monkeypatch):
    monkeypatch.setenv('TINYH3_HOME', str(tmp_path / 'home'))
    import io
    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient
    from tinyh3.server.app import create_app

    def png(w=96, h=64, seed=0):
        buf = io.BytesIO()
        Image.fromarray((np.random.RandomState(seed).rand(h, w, 3) * 255).astype('uint8')).save(buf, 'PNG')
        return buf.getvalue()

    client = TestClient(create_app(subprocess=False))
    with client:
        info = client.get('/api/info').json()
        assert list(info['tasks']) == ['t2v', 'i2v', 'flf2v', 'ref2v'] and info['ref_limits']['image'] == 9
        # uploads: kind from extension, binary body, listing back
        assert client.post('/api/uploads?name=x.exe', content=b'zz').status_code == 400
        assert client.post('/api/uploads?name=a.png', content=b'').status_code == 400
        a = client.post('/api/uploads?name=a.png', content=png()).json()
        assert a['kind'] == 'image' and a['id'].endswith('.png')
        assert client.get(f"/api/uploads/{a['id']}").content[:4] == b'\x89PNG'
        assert client.get('/api/uploads/..%2Fjobs.sqlite3').status_code in (400, 404)

        # validation
        assert client.post('/api/jobs', json={'prompt': 'w1', 'task': 'i2v'}).status_code == 400
        assert client.post('/api/jobs', json={'prompt': 'w1', 'task': 'ref2v', 'references': []}).status_code == 400
        aud = client.post('/api/uploads?name=s.wav', content=b'RIFFxxxx').json()
        assert client.post('/api/jobs', json={'prompt': 'w1', 'task': 'ref2v', 'references': [
            {'kind': 'audio', 'id': aud['id']}]}).status_code == 400
        assert client.post('/api/jobs', json={'prompt': 'w1', 'task': 'ref2v', 'references': [
            {'kind': 'video', 'id': a['id']}]}).status_code == 400   # a png is not a video

        # estimates grow with references (a simulated RTX 3070 only for these two requests)
        monkeypatch.setenv('TINYH3_SIMULATE', '8,16')
        base = client.post('/api/estimate', json={'resolution': '480p', 'task': 't2v'}).json()
        ref = client.post('/api/estimate', json={'resolution': '480p', 'task': 'ref2v', 'references': [
            {'kind': 'image', 'id': a['id']}, {'kind': 'image', 'id': a['id']}]}).json()
        monkeypatch.delenv('TINYH3_SIMULATE')
        assert ref['steps'] == 4 and ref['est']['denoise_s'] > base['est']['denoise_s']

        # i2v end to end through the queue
        job = client.post('/api/jobs', json={'prompt': 'w1 w2', 'task': 'i2v', 'first': a['id'], 'resolution': '360p',
                                             'aspect': '1:1', 'sampler': 'fast', 'seed': 5,
                                             'model': str(tiny_store)}).json()
        rec = _wait_done(client, job['id'])
        assert rec['status'] == 'done', rec.get('error')
        assert rec['request']['task'] == 'i2v' and rec['request']['first'].endswith(a['id'])
        assert client.get(f"/api/videos/{job['id']}").status_code == 200


def test_cross_origin_writes_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('TINYH3_HOME', str(tmp_path / 'home'))
    from fastapi.testclient import TestClient
    from tinyh3.server.app import create_app
    client = TestClient(create_app(subprocess=False))
    with client:
        evil = {'Origin': 'https://evil.example'}
        assert client.post('/api/jobs', json={'prompt': 'w1'}, headers=evil).status_code == 403
        assert client.post('/api/uploads?name=a.png', content=b'x', headers=evil).status_code == 403
        assert client.delete('/api/jobs/abc', headers=evil).status_code == 403
        assert client.get('/api/info', headers=evil).status_code == 200            # reads stay open
        same = {'Origin': 'http://testserver'}                                       # the page's own origin
        assert client.post('/api/jobs', json={'prompt': '  '}, headers=same).status_code == 400
