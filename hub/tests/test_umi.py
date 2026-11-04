"""hub/apps/umi.py: the relay to the excavator Pi's dashboard, against a fake Pi (a local http.server)."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from hub.app import Hub, create_app

TOKEN = 'pi-secret'


class FakePi(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    seen: list = []
    gate = threading.Event()

    def log_message(self, *a):
        pass

    def reply(self, code, body=b'', kind='application/json', headers=()):
        body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def record(self, body=b''):
        FakePi.seen.append({'method': self.command, 'path': self.path, 'headers': dict(self.headers), 'body': body})

    def do_GET(self):
        self.record()
        if self.headers.get('Cookie') != f'umi_access={TOKEN}':
            return self.reply(401, b'Open the private dashboard link.', 'text/plain')
        if self.path == '/api/status':
            return self.reply(200, {'camera': {'live': True}})
        if self.path.startswith('/preview.jpg'):
            return self.reply(200, b'\xff\xd8jpeg', 'image/jpeg', [('X-Frame-Sequence', '42'), ('X-Frame-Age-Ms', '3.5')])
        if self.path == '/file':
            data = bytes(range(100))
            rng = self.headers.get('Range')
            if rng:
                a, b = (int(x) for x in rng.split('=')[1].split('-'))
                return self.reply(206, data[a:b + 1], 'application/octet-stream', [('Content-Range', f'bytes {a}-{b}/100')])
            return self.reply(200, data, 'application/octet-stream')
        if self.path == '/login':
            return self.reply(303, b'', headers=[('Location', '/'), ('Set-Cookie', f'umi_access={TOKEN}; Path=/')])
        if self.path == '/stream.mjpg':          # like the Pi: no length, ends when the connection closes
            self.close_connection = True
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(b'--frame\r\nfirst\r\n')
            self.wfile.flush()
            FakePi.gate.wait(10)
            self.wfile.write(b'--frame\r\nsecond\r\n')
            return
        self.reply(404, {'error': 'Not found'})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.record(body)
        if self.headers.get('X-UMI-Action') != 'recording' or self.headers.get('Content-Type') != 'application/json':
            return self.reply(403, {'error': 'Use the dashboard recording controls.'})
        self.reply(409, {'error': 'busy', 'got': json.loads(body)})


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def pi():
    FakePi.seen, FakePi.gate = [], threading.Event()
    srv = ThreadingHTTPServer(('127.0.0.1', 0), FakePi)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_port}'
    FakePi.gate.set()
    srv.shutdown()
    srv.server_close()


def make_hub(tmp_path: Path, urls, token=True) -> Hub:
    tok = tmp_path / 'token'
    if token:
        tok.write_text(TOKEN + '\n')
    reg = tmp_path / 'apps.json'
    reg.write_text(json.dumps({'apps': [{'id': 'pi-dashboard', 'module': 'umi', 'view': 'umi', 'name': 'Pi',
                                         'options': {'urls': urls,
                                                     'token_files': [str(tmp_path / 'missing'), str(tok)]}}]}))
    return Hub(registry=reg, local=None, autostart=False)


@pytest.fixture
def client(tmp_path, pi):
    dead = f'http://127.0.0.1:{free_port()}'           # tried first, refuses connections
    h = make_hub(tmp_path, [dead, pi])
    with TestClient(create_app(h)) as c:
        assert h.start('pi-dashboard')['ok']
        yield c, pi


def test_relays_with_token_cookie_and_falls_through_to_reachable_url(client):
    c, pi = client
    r = c.get('/api/pi-dashboard/api/status', headers={'Cookie': 'hub_session=x'})
    assert r.status_code == 200 and r.json() == {'camera': {'live': True}}
    last = FakePi.seen[-1]
    assert last['path'] == '/api/status'
    assert last['headers']['Cookie'] == f'umi_access={TOKEN}'       # the browser's own cookies stay here
    s = c.get('/api/pi-dashboard/status').json()
    assert s['reachable'] and s['upstream'] == pi and s['token'] and s['authorized'] is True
    assert s['token_file'].endswith('token')


def test_post_headers_body_status_and_media_headers_pass_through(client):
    c, _ = client
    r = c.post('/api/pi-dashboard/api/recordings/start', content=b'{"label": "x"}',
               headers={'Content-Type': 'application/json', 'X-UMI-Action': 'recording'})
    assert r.status_code == 409 and r.json() == {'error': 'busy', 'got': {'label': 'x'}}
    assert c.post('/api/pi-dashboard/api/recordings/start', json={}).status_code == 403
    r = c.get('/api/pi-dashboard/preview.jpg?after=41')
    assert r.status_code == 200 and r.content == b'\xff\xd8jpeg'
    assert r.headers['X-Frame-Sequence'] == '42' and r.headers['X-Frame-Age-Ms'] == '3.5'
    assert FakePi.seen[-1]['path'] == '/preview.jpg?after=41'
    assert c.get('/api/pi-dashboard/nope').json() == {'error': 'Not found'}


def test_range_redirect_and_cookie_handling(client):
    c, _ = client
    r = c.get('/api/pi-dashboard/file', headers={'Range': 'bytes=10-19'})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))
    assert r.headers['Content-Range'] == 'bytes 10-19/100'
    r = c.get('/api/pi-dashboard/login', follow_redirects=False)
    assert r.status_code == 303 and r.headers['Location'] == '/api/pi-dashboard/'
    assert 'set-cookie' not in r.headers                            # the token never reaches the browser


def test_unreachable_and_missing_token(tmp_path, pi):
    h = make_hub(tmp_path, [f'http://127.0.0.1:{free_port()}'])
    with TestClient(create_app(h)) as c:
        h.start('pi-dashboard')
        r = c.get('/api/pi-dashboard/api/status')
        assert r.status_code == 502 and 'not reachable' in r.json()['error']
        assert c.get('/api/pi-dashboard/status').json()['reachable'] is False
        info = c.get('/api/hub/apps').json()['apps'][0]['info']
        assert info['reachable'] == 'no' and info['token cookie'] == 'yes'
    (tmp_path / 'token').unlink()
    h = make_hub(tmp_path, [pi], token=False)
    with TestClient(create_app(h)) as c:
        h.start('pi-dashboard')
        assert c.get('/api/pi-dashboard/api/status').status_code == 401
        s = c.get('/api/pi-dashboard/status').json()
        assert s['reachable'] and not s['token'] and s['authorized'] is False


def test_stopped_app_is_503(tmp_path, pi):
    h = make_hub(tmp_path, [pi])
    with TestClient(create_app(h)) as c:
        r = c.get('/api/pi-dashboard/api/status')
        assert r.status_code == 503 and 'stopped' in r.json()['error']


def test_stream_is_relayed_as_it_arrives(tmp_path, pi):
    """A real server: the first MJPEG part must reach the browser while the Pi is still sending."""
    h = make_hub(tmp_path, [pi])
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(h), host='127.0.0.1', port=port, log_level='warning'))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline and t.is_alive()
            time.sleep(0.02)
        assert h.start('pi-dashboard')['ok']
        t0 = time.monotonic()
        with httpx.stream('GET', f'http://127.0.0.1:{port}/api/pi-dashboard/stream.mjpg', timeout=5) as r:
            assert r.status_code == 200
            assert r.headers['content-type'].startswith('multipart/x-mixed-replace')
            chunks = r.iter_raw()
            got = b''
            while b'first' not in got:
                got += next(chunks)
            assert time.monotonic() - t0 < 3 and not FakePi.gate.is_set()   # streamed, not buffered
            FakePi.gate.set()
            got += b''.join(chunks)
        assert got == b'--frame\r\nfirst\r\n--frame\r\nsecond\r\n'
    finally:
        FakePi.gate.set()
        server.should_exit = True
        t.join(10)


def test_page_modules_are_served(client):
    c, _ = client
    for path in ('/web/data-collection/app.js', '/web/data-collection/latest-poll.js', '/web/data-collection/style.css',
                 '/web/views/umi/view.js', '/web/views/umi/style.css'):
        r = c.get(path)
        assert r.status_code == 200, path
    assert 'export function mount' in c.get('/web/data-collection/app.js').text


def test_urls_and_token_default_to_the_local_hosts_block(tmp_path, monkeypatch):
    """No urls / token_files in apps.json: hosts.pi of configs/hub/local.json (policy.data.local_config), the same
    block the inference app and the policy runner read (DEFAULT_PI_HOST)."""
    from hub.apps import umi
    from digital_twin.hardware import pi
    from policy.data import local_config
    assert pi.DEFAULT_PI_HOST == ','.join(local_config.DEFAULT_PI_URLS)       # tests see no local.json
    local = tmp_path / 'local.json'
    local.write_text(json.dumps({'hosts': {'pi': {'urls': ['http://169.254.10.2:8000', 'http://pi.example:8000/'],
                                                  'token_file': str(tmp_path / 'tok')}}}))
    monkeypatch.setenv('UMI_EXCAVATOR_LOCAL_CONFIG', str(local))
    relay = umi.start(None)
    assert relay.urls == ['http://169.254.10.2:8000', 'http://pi.example:8000']
    assert relay.token_files == [tmp_path / 'tok']
    explicit = umi.start(None, urls=['http://other:8000'], token_files=[str(tmp_path / 'x')])
    assert (explicit.urls, explicit.token_files) == (['http://other:8000'], [tmp_path / 'x'])
