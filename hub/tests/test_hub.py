import json
import sys
import types

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from hub.app import Hub, create_app


class Counter:
    def __init__(self, start=0):
        self.n, self.closed = start, False

    def info(self):
        return {'n': self.n}

    def close(self):
        self.closed = True


def make_module(name, fail=False):
    mod = types.ModuleType(f'hub.apps.{name}')

    def start(hub, start=0):
        if fail:
            raise RuntimeError('boom')
        return Counter(start)

    def routes(get):
        r = APIRouter()

        @r.post('/inc')
        def inc():
            c = get()
            c.n += 1
            return {'n': c.n}

        @r.get('/peer')
        def peer():
            return {'peer': mod.hub.service('a').n}
        return r

    def start_with_hub(hub, **kw):
        mod.hub = hub
        return start(hub, **kw)
    mod.start, mod.routes = start_with_hub, routes
    sys.modules[mod.__name__] = mod
    return mod


@pytest.fixture
def hub(tmp_path):
    make_module('fake_a')
    make_module('fake_b', fail=True)
    reg = tmp_path / 'apps.json'
    reg.write_text(json.dumps({'apps': [
        {'id': 'a', 'module': 'fake_a', 'name': 'A', 'options': {'start': 5}, 'first': True},
        {'id': 'b', 'module': 'fake_b', 'name': 'B', 'landing': True},
    ]}))
    h = Hub(registry=reg, local=tmp_path / 'local.json', autostart=False)
    with TestClient(create_app(h)) as c:
        h.start_all()
        yield h, c


def test_routes_call_the_service_directly(hub):
    h, c = hub
    assert c.post('/api/a/inc').json() == {'n': 6}
    assert h.service('a').n == 6
    assert c.get('/api/a/peer').json() == {'peer': 6}


def test_listing_status_and_errors(hub):
    h, c = hub
    apps = {a['id']: a for a in c.get('/api/hub/apps').json()['apps']}
    assert apps['a']['status'] == 'running' and apps['a']['info'] == {'n': 5}
    assert apps['b']['status'] == 'crashed' and 'boom' in apps['b']['error']
    r = c.post('/api/b/inc')
    assert r.status_code == 503 and 'failed' in r.json()['error']


def test_apps_cannot_be_stopped_over_http(hub):
    h, c = hub
    svc = h.service('a')
    for action in ('start', 'stop', 'restart', 'settings'):
        assert c.post(f'/api/hub/apps/a/{action}', json={'options': {}}).status_code == 404
    assert h.service('a') is svc and not svc.closed
    assert c.post('/api/a/inc').json() == {'n': 6}


def test_apps_start_with_the_hub_and_stop_when_it_exits(tmp_path):
    make_module('fake_a')
    reg = tmp_path / 'apps.json'
    reg.write_text(json.dumps({'apps': [{'id': 'a', 'module': 'fake_a', 'name': 'A'}]}))
    (tmp_path / 'local.json').write_text(json.dumps({'apps': {'a': {'options': {'start': 41}}}}))
    h = Hub(registry=reg, local=tmp_path / 'local.json')
    assert h.landing == 'a'                                                # no "landing" app: the first loaded
    with TestClient(create_app(h)) as c:
        svc = h.service('a', wait=5)
        assert svc is not None and c.post('/api/a/inc').json() == {'n': 42}    # local.json options apply
    assert svc.closed and h.service('a') is None


def test_shell_routes_and_unknown_api(hub):
    _, c = hub
    for path in ('/a', '/data-review/whatever'):
        r = c.get(path)
        assert r.status_code == 200 and 'shell.js' in r.text
    r = c.get('/', follow_redirects=False)
    assert r.is_redirect and r.headers['location'] == '/b'                 # the "landing": true app
    assert c.get('/api/hub/apps').json()['landing'] == 'b'
    assert c.get('/api/nope').status_code == 404
    assert c.get('/web/shell.js').status_code == 200
    assert c.get('/web/theme.css').status_code == 200


def test_host_and_origin_guard(hub):
    _, c = hub
    assert c.get('/api/hub/apps', headers={'Host': 'evil.example'}).status_code == 403
    assert c.post('/api/a/inc', headers={'Origin': 'http://evil.example'}).status_code == 403
    assert c.post('/api/a/inc', headers={'Origin': 'http://testserver'}).status_code == 200
