"""The hub's gpio app (hub/apps/gpio.py) over the in-process GPIO bridge, against the fake ESP32 of
digital_twin/tests/fake_esp32.py: HTTP routes, the websocket protocol, and a remote session through a running hub."""
import json
import socket
import threading
import time

import pytest

from digital_twin.hardware import gpio
from digital_twin.tests.fake_esp32 import FakeEsp32, fake, recv_until, wait_for


@pytest.fixture
def hub(tmp_path):
    from fastapi.testclient import TestClient
    from hub.app import Hub, create_app
    FakeEsp32.instances.clear()
    reg = tmp_path / 'apps.json'
    (tmp_path / 'hardware_map.json').write_text('{"joints": {"boom": {"positive": {"pin": 33, "label": "boom up"}}}}\n')
    reg.write_text(json.dumps({'apps': [{'id': 'gpio', 'name': 'GPIO bridge', 'first': True, 'options': {
        'target': 'fake://esp32', 'hardware_map': str(tmp_path / 'hardware_map.json')}}]}))
    h = Hub(registry=reg, local=None, autostart=False)
    with TestClient(create_app(h)) as c:
        h.start_all()
        b = h.service('gpio')
        assert wait_for(lambda: b.online and b._last_state is not None)
        yield h, c
        h.stop_all()


def test_hub_http_routes(hub):
    h, c = hub
    assert c.get('/api/gpio/health').json() == {'esp32': True, 'target': 'fake://esp32'}
    assert c.get('/api/gpio/labels').json() == {'33': 'boom up'}
    apps = c.get('/api/hub/apps').json()['apps']
    assert apps[0]['status'] == 'running' and apps[0]['info']['esp32'] is True
    assert h.service('gpio').trigger(33, 20)['type'] == 'ack'
    assert c.get('/api/gpio/recordings').status_code == 404


def test_hub_websocket_speaks_the_old_protocol(hub):
    h, c = hub
    with c.websocket_connect('/api/gpio/ws') as a, c.websocket_connect('/api/gpio/ws') as b:
        greet = [a.receive_json() for _ in range(3)]
        assert [m['type'] for m in greet] == ['bridge', 'labels', 'state']
        for _ in range(3):
            b.receive_json()
        a.send_text(json.dumps({'cmd': 'press', 'pin': 33, 'autorelease': True, 'id': 1}))
        b.send_text(json.dumps({'cmd': 'trigger', 'pin': 25, 'ms': 50, 'id': 1}))

        def until(ws, pred):
            for _ in range(50):
                m = ws.receive_json()
                if pred(m):
                    return m
            raise AssertionError('no message')
        assert until(a, lambda m: m['type'] == 'ack') == {'type': 'ack', 'cmd': 'press', 'pin': 33, 'id': 1}
        assert until(b, lambda m: m['type'] == 'ack') == {'type': 'ack', 'cmd': 'trigger', 'pin': 25, 'id': 1}
        assert until(b, lambda m: m['type'] == 'state' and any(p['pin'] == 33 and p['pressed'] for p in m['pins']))
        a.send_text('not json')
        assert until(a, lambda m: m['type'] == 'error') == {'type': 'error', 'error': 'invalid JSON'}
    assert wait_for(lambda: not fake().pressed[33])                  # a's autorelease hold, released on close
    assert h.service('gpio').info()['sessions'] == 0


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def test_remote_session_through_a_running_hub(hub):
    """CLI tools outside the hub: gpio.connect('host:port') over the real websocket."""
    import uvicorn
    from hub.app import create_app
    h, _ = hub
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(h), host='127.0.0.1', port=port, log_level='warning',
                                           lifespan='off'))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        assert wait_for(lambda: server.started, 5)
        host = f'127.0.0.1:{port}'
        assert gpio.health(host)['esp32'] is True
        with gpio.connect(host) as s:
            assert isinstance(s, gpio.RemoteSession)
            assert recv_until(s, lambda m: m['type'] == 'bridge')['esp32'] is True
            for _ in range(3):                                             # fresh reads with an unchanged state
                assert len(s.request({'cmd': 'state', 'fresh': True})['pins']) == 8
            # acknowledgements survive interleaved state updates
            assert s.request({'cmd': 'trigger', 'pin': 25, 'ms': 40})['cmd'] == 'trigger'
            assert s.request({'cmd': 'release', 'pin': 25})['cmd'] == 'release'
            assert next(p for p in s.request({'cmd': 'state', 'fresh': True})['pins'] if p['pin'] == 25)['pressed'] is False
            s.request({'cmd': 'trigger', 'pin': 33, 'ms': 60})
        time.sleep(0.15)
        with gpio.connect(host) as o:                                      # released after the requester left
            assert next(p for p in o.request({'cmd': 'state', 'fresh': True})['pins'] if p['pin'] == 33)['pressed'] is False
        # the policy executor over the websocket, as policy.runner uses it with --bridge-host
        from digital_twin.hardware.excavator import BridgeExecutor, load_hw
        ex = BridgeExecutor(load_hw(), host, dry_run=True, log=lambda *_: None)
        assert len(ex.probe_latency(n=2)) == 2
        ex = BridgeExecutor(load_hw(), host, dry_run=False, pace=False, log=lambda *_: None)
        ex.connect()
        ex.close()
    finally:
        server.should_exit = True
        t.join(5)


