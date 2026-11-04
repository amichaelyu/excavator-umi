"""The in-process GPIO bridge (digital_twin.hardware.gpio) against a fake ESP32 (fake_esp32.py; the hub's gpio app:
hub/tests/test_gpio.py). Nothing touches Bluetooth or hardware.
"""
import json
import threading
import time

import pytest

from digital_twin.hardware import gpio
from digital_twin.sim.pins import pin
from digital_twin.tests.fake_esp32 import FakeEsp32, drain, fake, recv_until, wait_for


@pytest.fixture
def bridge(tmp_path):
    FakeEsp32.instances.clear()
    hw = tmp_path / 'hardware_map.json'
    hw.write_text(json.dumps({'joints': {'boom': {'positive': {'pin': 33, 'label': 'boom up'},
                                                  'negative': {'pin': 32, 'label': 'boom down'}, 'sign': 1}}}))
    b = gpio.GpioBridge('fake://esp32', hardware_map_path=hw, recordings_dir=tmp_path / 'rec', log=lambda *_: None).start()
    assert wait_for(lambda: b.online and b._last_state is not None)
    yield b
    b.close()


# ------------------------------------------------------------------------------------------- bridge
def test_greeting_matches_the_old_websocket(bridge):
    with bridge.session('t') as s:
        msgs = drain(s)
    assert [m['type'] for m in msgs] == ['bridge', 'recording', 'labels', 'state']
    assert msgs[0] == {'type': 'bridge', 'esp32': True, 'target': 'fake://esp32'}
    assert msgs[1] == {'type': 'recording', 'active': False}
    assert msgs[2]['labels'] == {'33': 'boom up', '32': 'boom down'}
    assert len(msgs[3]['pins']) == 8 and 'id' not in msgs[3]
    assert bridge.status() == {'esp32': True, 'target': 'fake://esp32', 'recording': {'active': False}}


def test_ids_are_remapped_per_session(bridge):
    a, b = bridge.session('a'), bridge.session('b')
    drain(a); drain(b)
    a.send({'cmd': 'trigger', 'pin': 25, 'ms': 30, 'id': 1})
    b.send({'cmd': 'trigger', 'pin': 26, 'ms': 30, 'id': 1})
    ra = recv_until(a, lambda m: m['type'] == 'ack')
    rb = recv_until(b, lambda m: m['type'] == 'ack')
    assert (ra['id'], ra['pin']) == (1, 25) and (rb['id'], rb['pin']) == (1, 26)
    up = [m['id'] for m in fake().rx if m['cmd'] == 'trigger']
    assert len(set(up)) == 2 and all(isinstance(i, int) for i in up)
    # request() picks out its own reply; the id it used never leaks to recv()
    assert a.request({'cmd': 'release', 'pin': 25}) == {'type': 'ack', 'cmd': 'release', 'pin': 25}
    assert a.request({'cmd': 'trigger', 'pin': 99, 'ms': 5, 'id': 'x'}) == \
        {'type': 'error', 'error': 'pin not controllable', 'id': 'x'}
    a.close(); b.close()


def test_state_broadcasts_fan_out_only_on_change(bridge):
    a, b = bridge.session(), bridge.session()
    drain(a); drain(b)
    a.request({'cmd': 'press', 'pin': 33})
    for s in (a, b):
        st = recv_until(s, lambda m: m['type'] == 'state')
        assert next(p for p in st['pins'] if p['pin'] == 33)['pressed'] is True
    fake().emit(fake().state())                         # a heartbeat reply repeating the same state
    fake().emit({**fake().state(), 'id': 12345})        # ... or an unrelated id
    assert [m for m in drain(b) if m['type'] == 'state'] == []
    a.request({'cmd': 'release', 'pin': 'all'})
    assert recv_until(b, lambda m: m['type'] == 'state')
    a.close(); b.close()


def test_fresh_state_goes_to_the_firmware_and_cached_state_does_not(bridge):
    with bridge.session() as s:
        drain(s)
        n = len(fake().rx)
        cached = s.request({'cmd': 'state', 'id': 4})
        assert cached['id'] == 4 and 'fresh' not in cached and len(fake().rx) == n
        for _ in range(3):                              # unchanged state: every fresh read still answered
            r = s.request({'cmd': 'state', 'fresh': True, 'id': 5})
            assert r['type'] == 'state' and r['fresh'] is True and r['source'] == 'firmware' and r['id'] == 5
            assert len(r['pins']) == 8
        sent = [m for m in fake().rx[n:] if m['cmd'] == 'state']
        assert len(sent) == 3 and all(m['fresh'] is True and isinstance(m['id'], int) for m in sent)


def test_autorelease_holds_released_when_the_session_closes(bridge):
    s = bridge.session('holder')
    drain(s)
    assert s.request({'cmd': 'press', 'pin': 33, 'autorelease': True})['type'] == 'ack'
    assert s.request({'cmd': 'press', 'pin': 'D25', 'autorelease': True})['type'] == 'ack'
    assert s.request({'cmd': 'press', 'pin': 26})['type'] == 'ack'               # no autorelease: stays
    assert s.holds == {'33', 'd25'}
    s.close()
    assert wait_for(lambda: not fake().pressed[33] and not fake().pressed[25])
    assert fake().pressed[26]
    rel = [m for m in fake().rx if m['cmd'] == 'release']
    assert {m['pin'] for m in rel} == {33, 'd25'} and all('id' not in m for m in rel)
    with pytest.raises(ConnectionError):
        s.send({'cmd': 'state'})


def test_pin_released_after_a_triggering_session_disconnects(bridge):
    s = bridge.session()
    s.request({'cmd': 'trigger', 'pin': 33, 'ms': 60})
    s.close()
    time.sleep(0.15)
    with bridge.session() as o:
        st = o.request({'cmd': 'state', 'fresh': True})
    assert next(p for p in st['pins'] if p['pin'] == 33)['pressed'] is False


def test_reply_timeout(bridge, monkeypatch):
    monkeypatch.setattr(gpio, 'REPLY_TIMEOUT_S', 0.2)
    fake().silent = True
    with bridge.session() as s:
        t0 = time.monotonic()
        r = s.request({'cmd': 'trigger', 'pin': 25, 'ms': 10, 'id': 9}, timeout=3)
    assert r == {'type': 'error', 'id': 9, 'error': 'ESP32 did not reply'} and time.monotonic() - t0 < 2.0
    out = bridge.release_all(timeout=0.1)                # E-STOP never raises
    assert out['ok'] is False and 'TimeoutError' in out['error'] and 'ms' in out
    with pytest.raises(TimeoutError):
        bridge.trigger(25, 10, timeout=0.1)


def test_disconnect_fails_pending_and_reconnects(bridge, monkeypatch):
    monkeypatch.setattr(gpio, 'RECONNECT_S', 0.1)
    first = fake()
    s = bridge.session()
    drain(s)
    first.silent = True
    s.send({'cmd': 'trigger', 'pin': 25, 'ms': 10, 'id': 1})
    time.sleep(0.05)
    first.drop()
    assert recv_until(s, lambda m: m['type'] == 'error')['error'] == 'ESP32 disconnected'
    assert recv_until(s, lambda m: m['type'] == 'bridge') == {'type': 'bridge', 'esp32': False, 'target': 'fake://esp32'}
    assert not bridge.online
    assert s.request({'cmd': 'trigger', 'pin': 25, 'ms': 10}) == {'type': 'error', 'error': 'ESP32 not connected'}
    assert bridge.release_all() == {'ok': False, 'sent': False, 'error': 'ESP32 not connected', 'ms': pytest.approx(0, abs=50)}
    assert recv_until(s, lambda m: m['type'] == 'bridge' and m['esp32'])
    assert fake() is not first and bridge.online
    assert bridge.trigger(26, 10)['type'] == 'ack'
    assert bridge.release_all()['ok'] is True
    s.close()


def test_watchdog_drops_a_silent_link(tmp_path, monkeypatch):
    monkeypatch.setattr(gpio, 'RECONNECT_S', 60)
    monkeypatch.setattr(gpio, 'HEARTBEAT_S', 0.05)
    monkeypatch.setattr(gpio, 'UPSTREAM_TIMEOUT_S', 0.3)
    b = gpio.GpioBridge('fake://esp32', hardware_map_path=tmp_path / 'hw.json', recordings_dir=tmp_path / 'r',
                        log=lambda *_: None).start()
    try:
        assert wait_for(lambda: b.online)
        n = len(fake().rx)
        assert wait_for(lambda: len(fake().rx) > n)      # heartbeat: a plain state query
        assert fake().rx[-1] == {'cmd': 'state'}
        fake().silent = True                             # the ESP32 goes quiet
        assert wait_for(lambda: not b.online, 3)
    finally:
        b.close()


def test_labels_validation_and_file(bridge, tmp_path):
    with bridge.session() as s:
        drain(s)
        s.send({'cmd': 'set_label', 'pin': 99, 'label': 'x', 'id': 3})
        assert s.recv(1) == {'type': 'error', 'id': 3, 'error': 'unknown pin'}
        s.send({'cmd': 'set_label', 'pin': 'abc', 'label': 'x'})
        assert s.recv(1) == {'type': 'error', 'error': 'unknown pin'}
        s.send({'cmd': 'set_label', 'pin': '33', 'label': '  claw down ' + 'x' * 50})
        m = s.recv(1)
        assert m['type'] == 'labels' and m['labels']['33'] == ('claw down ' + 'x' * 50)[:40]
        s.send({'cmd': 'set_label', 'pin': 32, 'label': ''})            # a joint pin keeps a name
        assert s.recv(1)['labels']['32'] == 'boom down'
        s.send({'cmd': 'set_label', 'pin': 26, 'label': 'claw'})        # an ESP32 pin no joint uses has none
        assert s.recv(1)['labels'] == {'33': ('claw down ' + 'x' * 50)[:40], '32': 'boom down'}
        s.send_text('{nope')
        assert s.recv(1) == {'type': 'error', 'error': 'invalid JSON'}
        s.send_text('[1]')
        assert s.recv(1) == {'type': 'error', 'error': 'expected a JSON object'}
    saved = (tmp_path / 'hardware_map.json').read_text()
    hw = json.loads(saved)
    assert 'pins' not in hw and saved.endswith('}\n') and '\n      "positive": {' in saved
    assert hw['joints'] == {'boom': {'positive': {'pin': 33, 'label': ('claw down ' + 'x' * 50)[:40]},
                                     'negative': {'pin': 32, 'label': 'boom down'}, 'sign': 1}}   # the rest is kept
    assert bridge.labels_dict() == {'32': 'boom down', '33': ('claw down ' + 'x' * 50)[:40]}


def test_recorder_jsonl_and_csv(bridge):
    with bridge.session('rec') as s:
        drain(s)
        s.send({'cmd': 'record_start', 'name': '  Dig test #3  '})
        st = recv_until(s, lambda m: m['type'] == 'recording')
        assert st['active'] and st['name'] == 'Dig test #3' and st['id'].endswith('_dig-test-3')
        s.send({'cmd': 'trigger', 'pin': 33, 'ms': 40, 'id': 'a'})
        assert recv_until(s, lambda m: m['type'] == 'ack')['id'] == 'a'
        assert wait_for(lambda: not fake().pressed[33])
        time.sleep(0.05)
        s.send({'cmd': 'set_label', 'pin': 32, 'label': 'claw down'})
        s.send({'cmd': 'record_stop'})
        recv_until(s, lambda m: m['type'] == 'recordings_changed')
    rid = st['id']
    lines = [json.loads(x) for x in bridge.recording_path(rid).read_text().splitlines()]
    types = [e['type'] for e in lines]
    assert types[0] == 'meta' and types[1] == 'snapshot' and types[-1] == 'end'
    meta = lines[0]
    assert meta['esp32Url'] == 'fake://esp32' and meta['esp32Connected'] is True and meta['startedAt'].endswith('Z')
    assert {'33': 'boom up', '32': 'boom down'} == meta['labels']
    assert next(p for p in lines[1]['pins'] if p['pin'] == 33)['label'] == 'boom up'
    cmd = next(e for e in lines if e['type'] == 'command')
    assert cmd['req'] == 'a' and cmd['cmd'] == 'trigger' and cmd['pin'] == 33 and cmd['ms'] == 40
    reply = next(e for e in lines if e['type'] == 'reply')
    assert reply['ok'] is True and 'error' not in reply and isinstance(reply['latency_ms'], int)
    pins = [e for e in lines if e['type'] == 'pin']
    assert [(e['pin'], e['pressed']) for e in pins] == [(33, True), (33, False)] and pins[0]['label'] == 'boom up'
    assert {'type', 'pin', 'label'} <= set(next(e for e in lines if e['type'] == 'label'))
    assert lines[-1]['events'] == len(lines)
    assert all(isinstance(e['t'], int) and isinstance(e['rel'], (int, float)) for e in lines)
    info = bridge.recordings()[0]
    assert info['id'] == rid and info['name'] == 'Dig test #3' and info['complete'] is True
    assert info['events'] == len(lines) and info['durationMs'] == lines[-1]['durationMs']
    text = bridge.recording_csv(rid)
    rows = text.splitlines()
    assert rows[0] == 't_iso,t,rel,type,client,cmd,pin,name,pressed,ms,ok,error,latency_ms,esp32,extra'
    assert len(rows) == len(lines) + 1 and text.endswith('\n')
    assert rows[1].split(',')[3] == 'meta' and '""labels""' in rows[1]           # extra: quoted JSON
    trig = next(r for r in rows if ',command,' in r).split(',')
    assert trig[5:10] == ['trigger', '33', '', '', '40']
    assert bridge.recording_csv('../x') is None and bridge.recording_path('missing') is None


def test_in_process_helpers(bridge):
    assert bridge.trigger(25, 30)['type'] == 'ack'
    with pytest.raises(RuntimeError, match='ms must be'):
        bridge.trigger(25, 0)
    r = bridge.release_all()
    assert r['ok'] and r['sent'] and r['reply']['type'] == 'ack' and r['ms'] >= 0
    assert bridge.state()['fresh'] is True


def test_connect_with_a_bridge_object(bridge):
    with gpio.connect(bridge) as s:
        assert isinstance(s, gpio.Session)
        assert s.request({'cmd': 'trigger', 'pin': 25, 'ms': 10})['type'] == 'ack'
    assert gpio.describe(bridge) == 'in-process GPIO bridge (fake://esp32)'
    assert gpio.bridge_url('localhost:8888') == 'ws://localhost:8888/api/gpio/ws'
    assert gpio.bridge_url('ws://h:1/ws') == 'ws://h:1/ws'


def test_bridge_executor_and_pulse_backend_run_in_process(bridge):
    from digital_twin.hardware.bridge import BridgeExecutor, load_hw
    from digital_twin.hardware.commands import BridgePulseBackend
    hw = load_hw()
    ex = BridgeExecutor(hw, bridge, dry_run=False, pace=False, log=lambda *_: None)
    ex.connect()
    pulse = [{'t_start_s': 0.0, 'joint': 'boom', 'direction': 1, 'duration_ms': 20, 'gpio': pin(hw, 'boom', 1)}]
    assert ex.execute(pulse)['sent'] == 1
    assert len(ex.probe_latency(n=2)) == 2
    ex.close()
    sched = {'pulses': [{**pulse[0], 'label': 'up', 'sign_verified': False}], 'gap_ms': 0}
    res = BridgePulseBackend(bridge, dry_run=False, i_understand_open_loop=True, record=True,
                             log=lambda *_: None).execute(sched)
    assert res['sent'] == 1 and [f['cmd'] for f in res['frames']] == ['record_start', 'trigger', 'release', 'record_stop']
    assert bridge.recordings()[0]['name'] == 'goto_pose'


# ------------------------------------------------------------------------------------------- calibrate
def test_calibration_presses_every_joint_on_the_bridge_and_stops(bridge):
    """hardware.calibrate's LiveArm on the hub's in-process bridge: the arm joints and the base turn pass the pin
    check (phase homing), the stop event refuses the next press, and close() releases everything."""
    from digital_twin.hardware.bridge import load_hw
    from digital_twin.hardware.calibrate import LiveArm
    io = object.__new__(LiveArm)                        # no Pi: only the bridge side
    stop = threading.Event()
    io.hw, io.log, io.yes, io.stop, io.executor, io._bridge_target = load_hw(), lambda *_: None, True, stop, None, bridge
    io.stream, io.sensors = None, type('S', (), {'close': lambda self: None})()
    hw = io.hw
    for joint, d in (('slew', 1), ('stick', -1), ('bucket', 1)):
        io.press(joint, d, 20)
        gpio = pin(hw, joint, d)
        assert wait_for(lambda: any(m.get('cmd') == 'trigger' and m.get('pin') == gpio for m in fake().rx))
    stop.set()
    with pytest.raises(KeyboardInterrupt):
        io.press('boom', 1, 20)
    io.close()
    assert wait_for(lambda: [m['cmd'] for m in fake().rx if m['cmd'] != 'state'][-1] == 'release')
    assert wait_for(lambda: bridge.info()['sessions'] == 0)
