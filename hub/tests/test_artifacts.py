"""The hub serves artifacts/ (static review pages) with seekable video."""
import json

from fastapi.testclient import TestClient

from hub.app import Hub, create_app


def client(tmp_path):
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts/clip.mp4').write_bytes(bytes(range(256)))
    (tmp_path / 'artifacts/review').mkdir()
    (tmp_path / 'artifacts/review/index.html').write_text('<p>review</p>')
    (tmp_path / 'private.txt').write_text('outside artifacts')
    reg = tmp_path / 'apps.json'
    reg.write_text(json.dumps({'apps': []}))
    return TestClient(create_app(Hub(registry=reg, local=None, autostart=False), artifacts=tmp_path / 'artifacts'))


def test_browser_can_request_video_ranges(tmp_path):
    with client(tmp_path) as c:
        r = c.get('/artifacts/clip.mp4', headers={'Range': 'bytes=10-19'})
        assert r.status_code == 206
        assert r.headers['content-range'] == 'bytes 10-19/256'
        assert r.headers['accept-ranges'] == 'bytes'
        assert r.content == bytes(range(10, 20))
        r = c.head('/artifacts/clip.mp4')
        assert r.headers['content-length'] == '256' and r.content == b''
        assert c.get('/artifacts/review/').text == '<p>review</p>'


def test_files_outside_artifacts_are_not_served(tmp_path):
    with client(tmp_path) as c:
        for url in ('/artifacts/../private.txt', '/artifacts/%2e%2e/private.txt', '/artifacts/..%2fprivate.txt'):
            assert 'outside artifacts' not in c.get(url).text
