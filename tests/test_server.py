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
