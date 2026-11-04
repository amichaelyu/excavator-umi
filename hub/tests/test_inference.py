"""Inference app (hub.apps.inference: routes over hub.apps.inference.core.Inference) against fakes for the Pi dashboard,
the hub's GPIO bridge and the Lambda CLOAK policy server. Runs under .venv (MuJoCo, no policy server).

Requests go through the real hub (hub.app.create_app with a one-app registry whose service is an Inference built
here) with FastAPI's TestClient; job events are read from the Inference event bus (TestClient buffers whole
responses, so the SSE endpoint itself is tested once against a real uvicorn server).

Every test uses a temp copy of configs/hardware/hardware_map.json with the calibration state it needs
(:func:`write_map`), so the tests do not depend on what the real map says today. The dashboard
has no simulation and no dry run: jobs send their pulses to the fake bridge and read the fake Pi (its arm IMUs
included).

    .venv/bin/python -m pytest -q hub/tests/test_inference.py
"""
import http.client
import io
import json
import math
import queue
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from hub.apps import _common
from hub.apps.inference import core as inference_core
from hub.apps.inference import jobs as inference_jobs
from hub.apps.inference import pi as inference_pi
from hub.apps.inference import util as inference_util
from digital_twin.paths import HARDWARE_MAP
from digital_twin.policy import runner as pr
from digital_twin.policy import cloak as cloak_policy
from digital_twin.tests.test_cloak_policy import FakeCloak, with_md
from digital_twin.tests.test_policy_runner import FakeCloakClient
from hub import app as hub_app
from hub.apps import inference as hub_inference

REPO = Path(__file__).resolve().parents[2]
VIEW_DIR = REPO / 'hub' / 'web' / 'views' / 'inference'
TOKEN = 'test-token-123'
LAMBDA_URL = 'ws://127.0.0.1:8001'
RELEASE_ALL = {'cmd': 'release', 'pin': 'all'}


# ------------------------------------------------------------------ fake Pi dashboard
def _jpeg(w=640, h=360, color=(40, 120, 200)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (w, h), color).save(buf, 'JPEG')
    return buf.getvalue()


class FakePi(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.jpeg = _jpeg()
        self.requests = []
        self.imus = None                                  # {name: entry}: the per-link IMUs of /api/status

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(h):
                self.requests.append(h.path)
                if f'umi_access={TOKEN}' not in (h.headers.get('Cookie') or ''):
                    h.send_response(401); h.send_header('Content-Length', '0'); h.end_headers(); return
                if h.path == '/api/status':
                    body = json.dumps({'uptime_s': 12.0,
                                       'camera': {'live': True, 'age_s': 0.02, 'fps': 59.8, 'frames': 1000},
                                       'imu': {'live': True, 'age_s': 0.01, 'hz': 142.0,
                                               'sample': {'accel': [-0.07, 0.98, -0.20], 'gyro': [0.1, -0.2, 0.3],
                                                          'read_monotonic_ns': 1, 'clipped': False}},
                                       **({'imus': self.imus} if self.imus else {})}).encode()
                    ctype = 'application/json'
                elif h.path == '/preview.jpg':
                    body, ctype = self.jpeg, 'image/jpeg'
                else:
                    h.send_response(404); h.send_header('Content-Length', '0'); h.end_headers(); return
                h.send_response(200); h.send_header('Content-Type', ctype)
                h.send_header('Content-Length', str(len(body))); h.end_headers(); h.wfile.write(body)

        super().__init__(('127.0.0.1', 0), H)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def host(self):
        return f'127.0.0.1:{self.server_port}'

    def close(self):
        self.shutdown(); self.server_close()


# ------------------------------------------------------------------ fake GPIO bridge (the hub's gpio service)
class FakeSession:
    """GpioBridge.session(): the bridge's JSON protocol as dicts; acks trigger / release, answers state."""

    def __init__(self, bridge):
        self.bridge, self.q, self._id = bridge, queue.Queue(), 0
        self.q.put({'type': 'bridge', 'esp32': True})
        self.q.put({'type': 'state', 'pins': {}})

    def send(self, msg):
        self.bridge.messages.append({k: v for k, v in msg.items() if k != 'id'})
        if msg.get('cmd') in ('release', 'trigger'):
            self.q.put({'type': 'ack', 'id': msg.get('id'), 'cmd': msg['cmd'], 'pin': msg.get('pin')})
        elif msg.get('cmd') == 'state':
            self.q.put({'type': 'state', 'id': msg.get('id'), 'pins': {}})

    def recv(self, timeout=None):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def request(self, msg, timeout=5.0):
        self._id += 1
        msg = {**msg, 'id': msg.get('id', f'fake-{self._id}')}
        self.send(msg)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            r = self.recv(0.05)
            if r is not None and r.get('id') == msg['id']:
                return r
        raise TimeoutError('no reply')

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


try:                                                    # gpio.connect() & co. dispatch on isinstance(GpioBridge)
    from digital_twin.hardware.gpio import GpioBridge as _GpioBridge
except ImportError:                                     # pragma: no cover
    _GpioBridge = object


class FakeBridge(_GpioBridge):
    """GpioBridge's API in-process without an ESP32: status / release_all / trigger / session. Records every
    command sent (without ids)."""
    target = 'ble://fake-gpio'
    online = True

    def __init__(self):                                 # no link, no threads: GpioBridge.__init__ is not run
        self.messages = []

    def status(self):
        return {'esp32': self.online, 'target': self.target}

    def release_all(self, timeout=2.0):
        self.messages.append(dict(RELEASE_ALL))
        return {'ok': True, 'sent': True, 'reply': {'type': 'ack', 'cmd': 'release'}, 'ms': 1.0}

    def trigger(self, pin, ms, timeout=3.0):
        self.messages.append({'cmd': 'trigger', 'pin': int(pin), 'ms': int(ms)})
        return {'type': 'ack', 'cmd': 'trigger', 'pin': int(pin)}

    def session(self):
        return FakeSession(self)

    def close(self):
        pass


# ------------------------------------------------------------------ helpers
def write_map(path, signs_verified=False):
    """Temp hardware map = the real one with a controlled pin-sign state."""
    hw = json.loads(HARDWARE_MAP.read_text())
    for j in hw['joints'].values():
        j['sign_verified'] = signs_verified
    Path(path).write_text(json.dumps(hw, indent=2))
    return hw


def write_registry(path):
    """Temp local checkpoint registry with one Lambda-served CLOAK π0.5 checkpoint."""
    Path(path).write_text(json.dumps({'checkpoints': [{
        'id': 'cloak-test-1000', 'label': 'CLOAK · test · step 1000', 'backend': 'cloak',
        'profile': 'pi05_tabletop_excavator', 'policy_space': 'calibrated', 'gateway_url': LAMBDA_URL, 'mode': 'baseline',
        'state': 'succeeded', 'step': 1000, 'optimizer_updates': 1000, 'base_model': 'pi05_base', 'default': True,
        'task': 'dirt', 'prompt': 'scoop the dirt', 'joints': ['slew', 'boom', 'stick', 'bucket']}]}))


class SlowCloak(FakeCloakClient):
    """infer() blocks until released (a policy server call that never returns)."""
    release = threading.Event()
    calls = 0

    def infer(self, image, state, mask=None):
        SlowCloak.calls += 1
        SlowCloak.release.wait(30)
        return super().infer(image, state, mask)


TEST_MODULE = '_inference_under_test'


def make_hub(tmp: Path, factory):
    """A one-app hub whose 'inference' app is hub/apps/inference/__init__.py's router over Inference objects built by ``factory(hub)``."""
    mod = types.ModuleType(f'hub.apps.{TEST_MODULE}')
    mod.start = lambda hub: factory(hub)
    mod.routes = hub_inference.routes
    sys.modules[mod.__name__] = mod
    reg = tmp / 'apps.json'
    reg.write_text(json.dumps({'apps': [{'id': 'inference', 'module': TEST_MODULE, 'name': 'Inference'}]}))
    return hub_app.Hub(registry=reg, local=tmp / 'local.json', autostart=False)


class Base(unittest.TestCase):
    stop_timeout = 6.0
    signs_verified = False

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='inference-dash-'))
        self.pi, self.bridge = FakePi(), FakeBridge()
        self.token = self.tmp / 'token'; self.token.write_text(TOKEN + '\n')
        self.hw_path = self.tmp / 'hardware_map.json'
        write_map(self.hw_path, self.signs_verified)
        self.hw_orig = self.hw_path.read_text()
        self.runs = self.tmp / 'runs'
        self.registry = self.tmp / 'local_checkpoints.json'
        write_registry(self.registry)
        self.hub = make_hub(self.tmp, self.make_inference)
        self.client = TestClient(hub_app.create_app(self.hub))
        self.client.__enter__()
        self.assertTrue(self.hub.start('inference')['ok'])
        self.inference = self.hub.service('inference')

    def make_inference(self, hub):
        inference = inference_core.Inference(pi_host=self.pi.host, bridge=self.bridge, token_file=self.token, hardware_map=self.hw_path,
                     runs_dir=self.runs, poll=False, local_registry=self.registry,
                     stop_timeout_s=self.stop_timeout, cal_root=self.tmp / 'hwcal')
        inference.hub = hub
        return inference

    def tearDown(self):
        self.client.__exit__(None, None, None)            # lifespan end: the hub stops (closes) the app
        self.pi.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def req(self, method, path, body=None, headers=None):
        """(status, headers, body) for /api/<x> = the inference app's /api/inference/<x>."""
        if path.startswith('/api/'):
            path = '/api/inference/' + path[len('/api/'):]
        h = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode(); h.setdefault('Content-Type', 'application/json')
        r = self.client.request(method, path, content=data, headers=h)
        return r.status_code, r.headers, r.content

    def get_json(self, path):
        st, _h, body = self.req('GET', path)
        self.assertEqual(st, 200, body)
        return json.loads(body)

    def post(self, path, body=None):
        st, _h, raw = self.req('POST', path, body if body is not None else {})
        return st, json.loads(raw) if raw else None

    def wait_idle(self, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.inference.job
            if j is None or not j.running:
                if j is not None and j.thread is not None:
                    j.thread.join(5)
                return j
            time.sleep(0.05)
        self.fail('job did not finish')


class BusReader:
    """The events a page gets over SSE, read straight from the Inference event bus."""

    def __init__(self, inference):
        self.inference, self.q, self.events = inference, inference.bus.subscribe(), []

    def read_until(self, pred, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                ev = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.events.append(ev)
            if pred(ev):
                return ev
        raise AssertionError(f'event condition not met; got types {[e["type"] for e in self.events][-30:]}')

    def close(self):
        self.inference.bus.unsubscribe(self.q)


# ------------------------------------------------------------------ tests
class SensorAndMediaTests(Base):
    def test_sensors_camera_and_imu_pitch(self):
        s = self.get_json('/api/sensors?refresh=1')
        self.assertTrue(s['pi']['ok'])
        self.assertAlmostEqual(s['pi']['camera']['fps'], 59.8)
        self.assertEqual(s['pi']['imu']['accel_g'], [-0.07, 0.98, -0.20])
        self.assertIsInstance(s['pi']['imu']['pitch_deg'], float)        # the bucket link pitch (arm IMU calibration)
        self.assertEqual(s['pi']['imu']['pitch_source'], 'arm IMU calibration')
        self.assertTrue(s['bridge']['ok'] and s['bridge']['esp32'])
        self.assertNotIn('locked_pins', s['bridge'])                  # no pins are locked
        self.assertNotIn(TOKEN, json.dumps(s))

    def test_camera_proxy_and_observation_preview(self):
        st, h, body = self.req('GET', '/api/camera.jpg')
        self.assertEqual((st, h['Content-Type']), (200, 'image/jpeg'))
        self.assertEqual(body, self.pi.jpeg)
        st, h, body = self.req('GET', '/api/observation.jpg')
        self.assertEqual(st, 200)
        self.assertTrue(h['X-Observation'].startswith('preview'))
        img = _common.decode_jpeg(body)
        self.assertEqual(img.shape, (224, 224, 3))                         # the policy server's letterbox

    def test_camera_offline_is_502_not_estop(self):
        self.pi.close()
        self.inference.pi._frame = (0.0, 0, b'', '')
        st, _h, body = self.req('GET', '/api/camera.jpg')
        self.assertEqual(st, 502)
        self.assertIn('error', json.loads(body))
        self.assertEqual(self.bridge.messages, [])
        self.pi = FakePi()                       # for tearDown

    def test_twin_frame_jpeg(self):
        st, h, body = self.req('GET', '/api/twin_frame?camera=side')
        self.assertEqual((st, h['Content-Type']), (200, 'image/jpeg'), body[:200])
        self.assertTrue(body.startswith(b'\xff\xd8'))
        self.assertEqual(self.req('GET', '/api/twin_frame?camera=nope')[0], 400)

    def test_run_file_range(self):
        d = self.runs / '20260101T000000Z-policy'; (d / 'frames').mkdir(parents=True)
        blob = bytes(range(256)) * 4
        (d / 'observations.mp4').write_bytes(blob)
        (d / 'frames' / '00000.jpg').write_bytes(self.pi.jpeg)
        (d / 'run.json').write_text(json.dumps({'steps': 3, 'latency_ms': {'median': 12.0}, 'events': [1]}))
        (d / 'job.json').write_text(json.dumps({'kind': 'policy', 'status': 'done', 'live': False}))
        st, h, body = self.req('GET', f'/api/runs/{d.name}/files/observations.mp4', headers={'Range': 'bytes=10-19'})
        self.assertEqual(st, 206)
        self.assertEqual(h['Content-Range'], f'bytes 10-19/{len(blob)}')
        self.assertEqual(body, blob[10:20])
        st, h, body = self.req('GET', f'/api/runs/{d.name}/files/observations.mp4')
        self.assertEqual((st, len(body), h['Accept-Ranges']), (200, len(blob), 'bytes'))
        self.assertEqual(self.req('GET', f'/api/runs/{d.name}/files/observations.mp4',
                                  headers={'Range': 'bytes=99999-'})[0], 416)
        st, h, body = self.req('GET', f'/api/runs/{d.name}/files/frames/00000.jpg')
        self.assertEqual((st, h['Content-Type']), (200, 'image/jpeg'))
        self.assertEqual(self.req('GET', f'/api/runs/{d.name}/files/..%2F..%2Fhardware_map.json')[0], 404)
        runs = self.get_json('/api/runs')['runs']
        self.assertEqual(runs[0]['id'], d.name)
        self.assertEqual(runs[0]['summary']['steps'], 3)
        self.assertNotIn('events', runs[0]['summary'])

    def test_foreign_host_and_origin_rejected(self):
        self.assertEqual(self.req('GET', '/api/state', headers={'Host': 'evil.example'})[0], 403)
        st, _h, _b = self.req('POST', '/api/stop', {}, headers={'Origin': 'http://evil.example'})
        self.assertEqual(st, 403)
        self.assertEqual(self.req('POST', '/api/stop', None, headers={'Content-Type': 'text/plain',
                                                                     'Content-Length': '0'})[0], 415)

    def test_view_module_is_served_by_the_hub(self):
        for p, ct in (('/web/views/inference/view.js', 'text/javascript'), ('/web/views/inference/style.css', 'text/css')):
            r = self.client.get(p)
            self.assertEqual(r.status_code, 200, p)
            self.assertTrue(r.headers['content-type'].startswith(ct), r.headers['content-type'])
        js = (VIEW_DIR / 'view.js').read_text()
        self.assertNotIn('http://cdn', js)
        self.assertNotIn('document.getElementById', js)                # DOM lookups scoped to the view root
        self.assertIn('export async function mount(root, ctx)', js)
        self.assertIn('[data-view="inference"] {', (VIEW_DIR / 'style.css').read_text())   # scoped stylesheet
        self.assertFalse((REPO / 'hub' / 'apps' / 'ops').exists() or (REPO / 'hub' / 'web' / 'views' / 'ops').exists())   # renamed: inference

    def test_sse_endpoint_streams_hello_and_events_then_unsubscribes(self):
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(hub_app.create_app(self.hub), host='127.0.0.1', port=0,
                                               log_level='warning', lifespan='off'))
        threading.Thread(target=server.run, daemon=True).start()
        t0 = time.time()
        while not server.started and time.time() - t0 < 10:
            time.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        try:
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            c.request('GET', '/api/inference/events')
            r = c.getresponse()
            self.assertEqual((r.status, r.getheader('Content-Type').split(';')[0]), (200, 'text/event-stream'))

            def next_event():
                data = []
                while True:
                    line = r.fp.readline().decode().rstrip('\n')
                    if line.startswith('data: '):
                        data.append(line[6:])
                    elif line == '' and data:
                        return json.loads('\n'.join(data))
            hello = next_event()
            self.assertEqual(hello['type'], 'hello')
            self.assertFalse(hello['data']['state']['busy'])
            self.assertIn('twin_state', hello['data']['sensors'])
            self.assertEqual(self.inference.bus.n_subscribers, 1)
            self.inference.bus.publish('log', {'job': None, 'line': 'hello page'})
            ev = next_event()
            self.assertEqual((ev['type'], ev['data']['line']), ('log', 'hello page'))
            c.close()
            t0 = time.time()
            while self.inference.bus.n_subscribers and time.time() - t0 < 10:   # a closed page is noticed (deadman)
                time.sleep(0.05)
            self.assertEqual(self.inference.bus.n_subscribers, 0)
        finally:
            server.should_exit = True


def arm_imu_cal(ref=None):
    """calibration.arm_imus for three IMUs mounted alike: joint axis = IMU z, up = IMU y at ``ref``."""
    ref = ref or {'boom': 0.5, 'stick': 1.0, 'bucket': -0.7}
    return {'session': 'test-arm', 'reference_q_rad': ref,
            'imus': {n: {'axis_imu': [0, 0, 1], 'up_ref_imu': [0, 1, 0]} for n in ref}}


def imu_entry(link_turn, age_s=0.01, live=True):
    """A Pi ``imus`` entry for a link turned ``link_turn`` rad about its axis from the reference pose:
    the up reading turns the other way in the IMU frame."""
    up = [math.sin(link_turn), math.cos(link_turn), 0.0]
    return {'live': live, 'age_s': age_s, 'hz': 90.0, 'accel_mean': up,
            'sample': {'accel': up, 'gyro': [0.0, 0.0, 0.0]}}


class TwinStateTests(Base):
    def set_map(self, **kw):
        hw = json.loads(self.hw_path.read_text())
        hw['joints']['stick']['status'] = kw.pop('stick', 'locked')
        hw['calibration']['arm_imus'] = kw.pop('arm_imus', None)
        tp = kw.pop('true_pose', None)
        if tp is None:
            hw.pop('true_pose', None)                                # model convention unless a test asks
        else:
            hw['true_pose'] = tp
        self.hw_path.write_text(json.dumps(hw, indent=2))
        return hw

    def test_true_pose_offsets_shift_boom_and_stick_only(self):
        off = {'boom': -0.664, 'stick': -0.119}
        hw = self.set_map(true_pose={'offset_rad': off})
        st = self.get_json('/api/twin_state')
        self.assertAlmostEqual(st['joints']['boom']['rad'], hw['homing']['joints']['boom']['home_value_rad'] + off['boom'])
        self.assertAlmostEqual(st['joints']['stick']['rad'], hw['locked_joints']['stick']['value_rad'] + off['stick'])
        self.assertAlmostEqual(st['joints']['bucket']['rad'], hw['homing']['joints']['bucket']['home_value_rad'])
        self.assertIn('true pose', st['convention'])
        hw = self.set_map(true_pose={'offset_rad': off, 'display_deg': {
            'boom_underside_from_horizontal': {'add_deg': 55.04, 'of': ['boom']},
            'stick_from_horizontal': {'add_deg': -43.14, 'of': ['boom', 'stick']}}})
        st = self.get_json('/api/twin_state')
        b, k = st['joints']['boom'], st['joints']['stick']
        self.assertAlmostEqual(b['deg'], math.degrees(b['rad']) + 55.04)                # what a phone on the boom reads
        self.assertAlmostEqual(k['deg'], math.degrees(b['rad'] + k['rad']) - 43.14)     # 0 = flat

    def test_without_arm_calibration_the_moving_joints_are_assumed_and_not_live(self):
        hw = self.set_map()
        st = self.get_json('/api/twin_state')
        self.assertEqual(list(st['joints']), ['slew', 'boom', 'stick', 'bucket'])
        src = {n: j['source'] for n, j in st['joints'].items()}
        self.assertEqual(src, {'slew': 'locked', 'boom': 'assumed', 'stick': 'locked', 'bucket': 'assumed'})
        self.assertAlmostEqual(st['joints']['stick']['rad'], hw['locked_joints']['stick']['value_rad'])
        self.assertAlmostEqual(st['joints']['boom']['rad'], hw['homing']['joints']['boom']['home_value_rad'])
        self.assertAlmostEqual(st['joints']['boom']['deg'], math.degrees(st['joints']['boom']['rad']))
        self.assertFalse(st['live'])
        self.assertIsNone(st['age_s'])
        self.assertFalse(st['arm_imus']['calibrated'])
        self.assertIn('calibration.arm_imus', st['arm_imus']['error'])
        self.assertIn('twin_state', self.get_json('/api/sensors?refresh=1'))

    def test_link_imus_measure_boom_stick_and_bucket(self):
        self.set_map(arm_imus=arm_imu_cal())
        self.pi.imus = {'boom': imu_entry(0.1), 'stick': imu_entry(0.1 + 0.2), 'bucket': imu_entry(0.0, age_s=0.05)}
        s = self.inference.refresh_sensors(bridge=False)['twin_state']   # the fake bridge is slow: samples would age
        j = s['joints']
        self.assertEqual({n: v['source'] for n, v in j.items()},
                         {'slew': 'locked', 'boom': 'imu', 'stick': 'imu', 'bucket': 'imu'})
        self.assertAlmostEqual(j['boom']['rad'], 0.5 + 0.1, places=6)
        self.assertAlmostEqual(j['stick']['rad'], 1.0 + 0.2, places=6)
        self.assertAlmostEqual(j['bucket']['rad'], -0.7 - 0.3, places=6)
        self.assertTrue(s['live'])
        self.assertGreaterEqual(j['bucket']['age_s'], 0.05)       # the older of its link and parent link
        self.assertTrue(s['arm_imus']['calibrated'])
        self.assertEqual(s['arm_imus']['session'], 'test-arm')
        self.assertEqual(sorted(s['arm_imus']['imus']), ['boom', 'bucket', 'stick'])
        later = self.inference.live_twin.snapshot(now=time.monotonic() + 2.0)   # no new samples: stale
        self.assertFalse(later['live'])
        self.assertEqual(later['joints']['boom']['source'], 'assumed')

    def test_a_job_estimate_is_dead_reckoned_and_ages(self):
        self.set_map(stick='available')
        self.inference.set_estimate({'boom': 0.3, 'bucket': -0.2, 'stick': 0.9}, 'dead reckoning (commanded)')
        st = self.get_json('/api/twin_state')
        self.assertEqual({n: j['source'] for n, j in st['joints'].items()},
                         {'slew': 'locked', 'boom': 'dead_reckoned', 'stick': 'dead_reckoned', 'bucket': 'dead_reckoned'})
        self.assertAlmostEqual(st['joints']['bucket']['rad'], -0.2)
        self.assertTrue(st['live'])
        self.assertEqual(st['job_source'], 'dead reckoning (commanded)')
        with self.inference._lock:
            self.inference.q_est_t -= 5.0                                # the job stopped updating it 5 s ago
        st = self.get_json('/api/twin_state')
        self.assertFalse(st['live'])
        self.assertGreater(st['age_s'], 4.9)

    def test_twin_state_jpeg(self):
        self.set_map()
        st, h, body = self.req('GET', '/api/twin_state.jpg?camera=side')
        self.assertEqual((st, h['Content-Type']), (200, 'image/jpeg'), body[:200])
        self.assertTrue(body.startswith(b'\xff\xd8'))
        self.assertEqual(h['X-Twin-State'], 'stale')
        self.assertEqual(self.req('GET', '/api/twin_state.jpg?camera=nope')[0], 400)


class CalibrationAndCheckpointTests(Base):
    def test_calibration_status_reflects_hardware_map(self):
        g = self.get_json('/api/calibration')
        c = g['map']
        self.assertFalse(c['joints']['boom']['sign_verified'])
        self.assertTrue(c['arm_imus']['calibrated'])                       # the example map's link-IMU calibration
        self.assertTrue(c['slew']['calibrated'])
        self.assertEqual(c['joints']['stick']['positive'], json.loads(self.hw_path.read_text())['joints']['stick']['positive'])
        self.assertIn('pin signs ✗ (boom, stick, bucket not verified)', g['summary']['text'])
        self.assertEqual((g['last'], g['last_sim']), (None, None))
        self.assertTrue(g['manual_prerequisites'] and g['motion'])
        hw = json.loads(self.hw_path.read_text())
        for j in ('boom', 'stick', 'bucket'):
            hw['joints'][j].pop('sign_verified')
        hw['calibration']['arm_imus']['session'] = '20261104T101500Z'
        hw['calibration']['speeds']['bucket'] = {'speed_measured_rad_s': {'positive': 1.1}, 'dead_time_s': {'positive': 0.12, 'negative': -0.3}}
        hw['slew']['sign_verified'] = False
        self.hw_path.write_text(json.dumps(hw))
        g = self.get_json('/api/calibration')
        c, text = g['map'], g['summary']['text']
        self.assertEqual(c['joints']['bucket']['speed_deg_s'], {'out': round(math.degrees(1.1), 1)})
        self.assertEqual(c['joints']['bucket']['dead_time_s'], {'out': 0.12})      # a negative fit is not a dead time
        for piece in ('arm IMUs ✓ (session 20261104T101500Z)', 'ranges ✓', 'speeds ✗ (bucket-in missing)', 'base turn ✗',
                      'pin signs ✓'):
            self.assertIn(piece, text)
        self.assertFalse(g['summary']['ready_for_live_policy'])
        self.assertEqual(g['summary']['calibrated_at'], '2026-11-04 10:15:00Z')

    def test_checkpoints_list_the_local_registry(self):
        c = self.get_json('/api/checkpoints?refresh=1')
        self.assertEqual(c['source'], 'local registry')
        self.assertIsNone(c['error'])
        (r,) = c['runs']
        self.assertEqual((r['backend'], r['profile'], r['policy_space']), ('cloak', 'pi05_tabletop_excavator', 'calibrated'))
        self.assertEqual(r['checkpoints'][0]['gateway_url'], LAMBDA_URL)
        self.assertEqual((c['default']['gateway_url'], c['default']['backend']), (LAMBDA_URL, 'cloak'))
        cfg = self.get_json('/api/config')
        self.assertEqual(sorted(cfg['profiles']), ['pi05_tabletop_excavator'])
        self.assertEqual(cfg['profiles']['pi05_tabletop_excavator']['joints'], ['slew', 'boom', 'stick', 'bucket'])
        self.assertEqual((cfg['default_prompt'], cfg['defaults']['hz'], cfg['defaults']['start_pose']),
                         ('scoop the dirt', 10, 'none'))
        self.assertEqual(cfg['known_prompts'], ['scoop the bottle cap', 'scoop the dirt'])


def fake_run(root, name, status='done', log='1/6 check\n6/6 done', **cal):
    """A hardware.calibrate run dir: calibration.json + log.txt."""
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'calibration.json').write_text(json.dumps({'mode': 'sim' if name.endswith('-sim') else 'live', 'status': status,
                                                    'calibration': cal}))
    (d / 'log.txt').write_text(log + '\n')
    return d


class CalibrationStatusTests(Base):
    def test_last_runs_and_the_log(self):
        root = self.tmp / 'hwcal'
        fake_run(root, '20261104T090000Z', ranges_rad={'boom': 0.8}, speeds_rad_s={'boom': {'negative': 0.6}},
                 dead_time_s={'boom': {'negative': 0.11}}, signs={'boom': {'sign': 1}, 'bucket': {'sign': -1}},
                 failures=['moving the stick turned the bucket IMU the opposite way'],
                 slew={'sign': 1, 'slew': {'left': {'yaw_rate_deg_s': 57.0}, 'right': {'yaw_rate_deg_s': -58.0}}})
        fake_run(root, '20261104T080000Z')
        fake_run(root, '20261104T100000Z-sim', log='simulated run', slew={'error': 'base turn not confirmed'})
        (root / 'notes').mkdir()
        hw = json.loads(self.hw_path.read_text())
        hw['calibration']['arm_imus']['session'] = '20261104T090000Z'
        self.hw_path.write_text(json.dumps(hw))
        g = self.get_json('/api/calibration')
        last, sim = g['last'], g['last_sim']
        self.assertEqual((last['name'], last['status'], last['at_text'], last['in_map']),
                         ('20261104T090000Z', 'done', '2026-11-04 09:00:00Z', True))
        self.assertEqual(last['ranges_deg'], {'boom': round(math.degrees(0.8), 1)})
        self.assertEqual(last['speeds_deg_s'], {'boom': {'down': round(math.degrees(0.6), 1)}})
        self.assertEqual(last['dead_time_s'], {'boom': {'down': 0.11}})
        self.assertEqual((last['signs']['bucket'], last['slew']['left'], last['slew']['sign']), (-1, 57.0, 1))
        self.assertEqual(len(last['failures']), 1)
        self.assertEqual((sim['mode'], sim['in_map'], sim['slew']), ('sim', False, {'error': 'base turn not confirmed'}))
        r = self.get_json('/api/calibration/report')                       # the newest run, simulated or live
        self.assertEqual((r['text'], r['run']['name']), ('simulated run\n', '20261104T100000Z-sim'))

    def test_no_run_yet(self):
        self.assertEqual(self.get_json('/api/calibration/report')['error'], 'no calibration run yet')

    def test_plan_gating(self):
        # always on the machine (no simulated job, no arming); only the manual prerequisites gate it
        plan = self.inference.calibration_plan({'prereqs_ok': True})
        self.assertEqual((plan['visits'], plan['slew']), (2, True))
        argv = plan['argv']
        for f in ('--live', '--i-understand-open-loop', '--yes'):
            self.assertIn(f, argv)
        self.assertEqual(argv[argv.index('--artifacts') + 1], str(self.tmp / 'hwcal'))
        self.assertEqual(argv[argv.index('--pi-host') + 1], self.pi.host)
        self.assertEqual(argv[argv.index('--token-file') + 1], str(self.token))
        argv = self.inference.calibration_plan({'visits': 3, 'slew': False, 'prereqs_ok': True})['argv']
        self.assertEqual((argv[argv.index('--visits') + 1], '--no-slew' in argv), ('3', True))
        for body in ({'visits': 9, 'prereqs_ok': True}, {'backend': 'sim', 'prereqs_ok': True}):
            self.assertEqual(self.post('/api/calibrate', body)[0], 400, body)
        st, err = self.post('/api/calibrate', {})
        self.assertEqual(st, 400)
        self.assertIn('manual prerequisites', err['error'])
        self.assertFalse(self.inference.job is not None and self.inference.job.running)
        self.assertEqual(self.bridge.messages, [])


class JobBase(Base):
    """Jobs drive the machine: the fake Pi reports live boom / stick / bucket IMUs at the reference pose of a test
    arm IMU calibration (the runner refuses a run without them), and pulses go to the fake bridge."""

    def setUp(self):
        super().setUp()
        from unittest import mock
        hw = json.loads(self.hw_path.read_text())
        hw['calibration']['arm_imus'] = arm_imu_cal()
        self.hw_path.write_text(json.dumps(hw, indent=2))
        self.pi.imus = {n: imu_entry(0.0) for n in ('boom', 'stick', 'bucket')}
        self._cloak = mock.patch.object(cloak_policy, 'CloakPolicy', FakeCloakClient)
        self._cloak.start()
        SlowCloak.release.clear()

    def tearDown(self):
        try:
            SlowCloak.release.set()
            if self.inference.job is not None and self.inference.job.running:
                self.inference.stop('test teardown'); self.wait_idle()
        finally:
            self._cloak.stop()
            super().tearDown()

    def policy_body(self, **kw):
        return {'profile': 'pi05_tabletop_excavator', 'gateway_url': LAMBDA_URL,
                'prompt': 'scoop the dirt', 'chunk': 3, 'hz': 10, 'max_seconds': 30, 'max_steps': 4, **kw}

    def logs(self, events):
        return [e['data']['line'] for e in events if e['type'] == 'log']

    def triggers(self):
        return [m for m in self.bridge.messages if m.get('cmd') == 'trigger']


class PolicyJobTests(JobBase):
    """Policy runs on the (fake) machine: pulses sent, the state measured by the arm IMUs."""

    def test_sse_streams_a_policy_run_with_the_real_camera(self):
        sse = BusReader(self.inference)
        try:
            self.assertFalse(self.get_json('/api/state')['busy'])
            st, job = self.post('/api/policy/start', self.policy_body(start_pose='0,25,55,-30'))
            self.assertEqual(st, 200, job)
            self.assertFalse({'live', 'mode'} & set(job) | {'mode', 'imu'} & set(job['params']))
            end = sse.read_until(lambda e: e['type'] == 'job_end', timeout=120)
        finally:
            sse.close()
        self.assertEqual(end['data']['status'], 'done', end['data'].get('error'))
        types_ = [e['type'] for e in sse.events]
        for t in ('state', 'log', 'progress', 'pulses', 'start_pose', 'step', 'estimate'):
            self.assertIn(t, types_)
        self.assertNotIn('homing', types_)                             # a policy run never homes
        phases = [e['data']['phase'] for e in sse.events if e['type'] == 'progress']
        for ph in ('setup', 'start_pose', 'infer', 'step', 'policy'):
            self.assertIn(ph, phases)
        self.assertNotIn('homing', phases)
        self.assertLess(phases.index('start_pose'), phases.index('infer'))
        steps = [e['data'] for e in sse.events if e['type'] == 'step']
        self.assertEqual([s['step'] for s in steps], [0, 1, 2, 3])
        s = steps[-1]
        for k in ('latency_ms', 'state', 'actions', 'targets', 'n_pulses', 'skipped_total', 'est_pitch_rad'):
            self.assertIn(k, s)
        self.assertEqual(s['estimate'], 'arm IMUs (measured)')
        self.assertTrue(any('IMU: boom, stick and bucket IMUs measure the joint state' in l for l in self.logs(sse.events)))
        # every pulse shown went to the bridge, in order (no dry run)
        pulse_ev = [e['data'] for e in sse.events if e['type'] == 'pulses']
        phases_ = {p['phase'] for p in pulse_ev}
        self.assertTrue({'start_pose', 'policy'} <= phases_ and 'homing' not in phases_)
        shown = [p['gpio'] for ev in pulse_ev for p in ev['pulses']]
        self.assertEqual([m['pin'] for m in self.triggers()], shown)
        self.assertTrue(shown)
        # the observation is the fake Pi camera frame, exactly as sent (native, + its state)
        st, h, body = self.req('GET', '/api/observation.jpg')
        self.assertEqual(st, 200)
        self.assertTrue(h['X-Observation'].startswith('sent'))
        self.assertIn('state=', h['X-Observation'])
        img = _common.decode_jpeg(body)
        self.assertEqual(img.shape[2], 3)
        mid = img[img.shape[0] // 2, img.shape[1] // 2].astype(int)   # FakePi colour (40, 120, 200), not a render
        self.assertLess(np.abs(mid - [40, 120, 200]).max(), 12, mid)
        self.assertIn('/preview.jpg', self.pi.requests)
        # estimated pose view + label
        self.assertEqual(self.inference.q_source, 'arm IMUs (measured)')
        self.assertEqual(self.req('GET', '/api/twin_frame')[0], 200)
        # record
        d = self.runs / job['id']
        for f in ('run.json', 'job.json', 'steps.csv', 'pulses.csv', 'log.txt'):
            self.assertTrue((d / f).is_file(), f)
        runs = self.get_json('/api/runs')['runs']
        self.assertEqual((runs[0]['id'], runs[0]['status'], runs[0]['summary']['steps']), (job['id'], 'done', 4))
        self.assertEqual(self.get_json(f"/api/runs/{job['id']}")['job']['params']['prompt'], 'scoop the dirt')

    def test_a_run_without_live_arm_imus_is_refused_before_any_pulse(self):
        self.pi.imus = None                                            # the Pi reports no arm IMUs
        st, job = self.post('/api/policy/start', self.policy_body())
        self.assertEqual(st, 200, job)                                 # the runner refuses on its worker
        j = self.wait_idle(30)
        self.assertEqual(j.status, 'error')
        self.assertIn('run refused: the arm IMUs give the joint state', j.error)
        self.assertEqual(self.triggers(), [])
        self.assertIn(RELEASE_ALL, self.bridge.messages)                # ended early: release-all

    def test_only_registered_lambda_checkpoints_start(self):
        captured = {}
        self.inference._launch_runner = lambda job, args: captured.setdefault('args', args) and job.info()
        for url in ('wss://api.pi-fleet.com/v1/models/ckpt-1', 'ws://127.0.0.1:9999'):
            st, err = self.post('/api/policy/start', self.policy_body(gateway_url=url))
            self.assertEqual(st, 400, err)
            self.assertIn('local_checkpoints.json', err['error'])
        self.assertEqual(self.post('/api/policy/start', self.policy_body(hz=5))[0], 400)
        self.assertEqual(self.post('/api/policy/start', self.policy_body(profile='excavator_joint_targets'))[0], 400)
        for pose in ('10,20', 'training'):                                     # 4 angles or none
            self.assertEqual(self.post('/api/policy/start', self.policy_body(start_pose=pose))[0], 400, pose)
        self.assertNotIn('args', captured)
        st, job = self.post('/api/policy/start', self.policy_body())
        self.assertEqual(st, 200, job)
        self.assertEqual((captured['args'].gateway_url, captured['args'].profile, captured['args'].start_pose),
                         (LAMBDA_URL, 'pi05_tabletop_excavator', 'none'))
        self.assertEqual(job['params']['policy_space'], 'calibrated')
        self.assertNotIn('self_mask', job['params'])
        self.inference.job.status = 'done'

    def test_policy_start_has_no_homing_imu_choice_or_arming(self):
        captured = {}
        def launch(job, args):
            captured['args'] = args
            return job.info()
        self.inference._launch_runner = launch
        st, job = self.post('/api/policy/start', self.policy_body(quick_calibration=True, home=True, home_stick=False,
                                                                  imu='off', live=False, arm_token='x'))   # ignored
        self.assertEqual(st, 200, job)
        self.assertFalse({'home', 'home_stick', 'quick_calibration', 'imu', 'mode', 'live'} & set(job['params']))
        self.assertFalse(captured['args'].home_only)
        for gone in ('home', 'no_quick_calibration', 'home_stick', 'imu', 'live', 'i_understand_open_loop', 'no_pace',
                     'probe_latency'):
            self.assertFalse(hasattr(captured['args'], gone), gone)
        self.assertIs(captured['args'].bridge, self.bridge)
        self.inference.job.status = 'done'

    def test_simulation_is_refused_and_the_removed_routes_are_gone(self):
        for body in (self.policy_body(backend='sim'), self.policy_body(camera='sim')):
            st, err = self.post('/api/policy/start', body)
            self.assertEqual(st, 400, err)
            self.assertIn('no simulation', err['error'])
        self.assertEqual(self.post('/api/calibrate', {'backend': 'sim', 'prereqs_ok': True})[0], 400)
        for gone in ('/api/home', '/api/arm', '/api/disarm'):               # Quick home and ARM LIVE are gone
            self.assertEqual(self.post(gone, {})[0], 404, gone)
        cfg = self.get_json('/api/config')
        self.assertEqual(cfg['backend'], 'bridge')
        self.assertFalse({'modes', 'pace', 'arm_ttl_s', 'stick_homing_blocked'} & set(cfg))
        self.assertFalse({'armed', 'arm_expires_in_s'} & set(self.get_json('/api/state')))
        self.assertEqual(self.bridge.messages, [])

    def _wait_not_busy(self, limit):
        t0 = time.time()
        while self.get_json('/api/state')['busy'] and time.time() - t0 < limit + 5:
            time.sleep(0.05)
        return time.time() - t0

    def test_stop_interrupts_a_blocked_infer(self):
        cloak_policy.CloakPolicy = SlowCloak                     # undone by the patcher in tearDown
        st, job = self.post('/api/policy/start', self.policy_body(start_pose='none'))
        self.assertEqual(st, 200, job)
        t0 = time.time()
        while SlowCloak.calls == 0 and time.time() - t0 < 30:
            time.sleep(0.05)
        self.assertGreater(SlowCloak.calls, 0)          # the worker is inside infer()
        self.assertEqual(self.post('/api/stop')[1]['stopped'], True)
        self.assertLess(self._wait_not_busy(3.0), 2.0)          # cooperative: no need for the hard timeout
        state = self.get_json('/api/state')
        self.assertEqual(state['job']['status'], 'stopped')
        self.assertEqual(json.loads((self.runs / job['id'] / 'job.json').read_text())['status'], 'stopped')
        self.assertIn(RELEASE_ALL, self.bridge.messages)

    def test_stop_hard_timeout_abandons_an_unresponsive_worker(self):
        self.inference.stop_timeout_s = 1.5
        stuck = threading.Event()

        def unresponsive_step(runner, k):                       # ignores every cancel check
            stuck.set(); SlowCloak.release.wait(30)
            return {}
        from unittest import mock
        with mock.patch.object(inference_jobs.DashboardRunner, 'step', unresponsive_step):
            st, job = self.post('/api/policy/start', self.policy_body(start_pose='none'))
            self.assertEqual(st, 200, job)
            self.assertTrue(stuck.wait(20))
            self.assertEqual(self.post('/api/stop')[1]['stopped'], True)
            elapsed = self._wait_not_busy(1.5)
            self.assertLess(elapsed, 4.0)
            state = self.get_json('/api/state')
            self.assertEqual(state['job']['status'], 'killed')
            self.assertIn('did not stop', state['job']['error'])
            self.assertIn(RELEASE_ALL, self.bridge.messages)
            # the slot is free; the abandoned worker cannot overwrite the result when it finally returns
            SlowCloak.release.set()
            self.inference.job.thread.join(10)
        self.assertEqual(self.inference.job.status, 'killed')
        self.assertEqual(json.loads((self.runs / job['id'] / 'job.json').read_text())['status'], 'killed')

    def test_start_pose_progress_then_stop_interrupts_the_worker(self):
        sse = BusReader(self.inference)
        try:
            st, job = self.post('/api/policy/start', self.policy_body(start_pose='0,0,30,60'))
            self.assertEqual(st, 200, job)
            self.assertEqual(job['status'], 'running')
            prog = sse.read_until(lambda e: e['type'] == 'progress' and e['data']['phase'] == 'start_pose'
                                  and (e['data']['elapsed_s'] or 0) > 0.3, timeout=20)
            self.assertGreater(prog['data']['planned_s'], 0.5)          # the pulses are waited out on the machine
            self.assertTrue(self.get_json('/api/state')['busy'])
            self.assertEqual(self.post('/api/policy/start', self.policy_body())[0], 409)
            t0 = time.time()
            st, r = self.post('/api/stop')
            self.assertEqual((st, r['stopped']), (200, True))
            end = sse.read_until(lambda e: e['type'] == 'job_end', timeout=10)
        finally:
            sse.close()
        self.assertLess(time.time() - t0, 2.0)                   # cooperative cancel, not the hard timeout
        self.assertEqual(end['data']['status'], 'stopped')
        self.assertFalse(self.get_json('/api/state')['busy'])
        self.assertEqual(json.loads((self.runs / job['id'] / 'job.json').read_text())['status'], 'stopped')
        self.assertIn(RELEASE_ALL, self.bridge.messages)
        self.assertFalse(self.post('/api/stop')[1]['stopped'])

    def test_emergency_stop_releases_all_and_aborts_job(self):
        st, job = self.post('/api/policy/start', self.policy_body(start_pose='0,0,30,60'))
        self.assertEqual(st, 200, job)
        time.sleep(0.3)
        r = self.inference.emergency_stop('test')
        self.assertTrue(r['bridge_release']['ok'], r)
        self.assertIn(RELEASE_ALL, self.bridge.messages)
        j = self.wait_idle(15)
        self.assertEqual(j.status, 'stopped')
        self.assertTrue(j.estopped)
        self.bridge.messages.clear()
        self.inference.emergency_stop('test')                      # nothing running: still releases
        self.assertEqual([m['cmd'] for m in self.bridge.messages], ['release'])

    def test_start_just_starts_whatever_the_pin_signs(self):
        # this map: every pin sign unverified; no arming and no calibration gate besides the arm IMUs
        st, job = self.post('/api/policy/start', self.policy_body(start_pose='0,0,30,60'))
        self.assertEqual(st, 200, job)
        self.assertTrue(self.post('/api/stop')[1]['stopped'])
        self.assertIn(self.wait_idle(15).status, ('stopped', 'done'))
        st, job = self.post('/api/policy/start', self.policy_body(start_pose='none', max_steps=1))   # and again
        self.assertEqual(st, 200, job)
        self.assertEqual(self.wait_idle(30).status, 'done')

    def test_page_puts_policy_first_and_calibration_collapsed(self):
        html = js = (VIEW_DIR / 'view.js').read_text()             # the markup lives in the view module
        self.assertLess(html.index('id="policy-card"'), html.index('id="cal-card"'))
        self.assertRegex(html, r'<details class="card calpanel" id="cal-card">')      # collapsed (no open attr)
        self.assertNotIn('allow-unverified', html + js)
        for gone in ('btn-quick-home', 'Quick home', 'home-stick', 'stick_homing_blocked', 'id="imu"', 'arm-btn',
                     'ARM LIVE', 'takeArming', 'arm_token', 'pill-armed', 'btn-cal-sim', 'Start (dry run)', 'DRY RUN'):
            self.assertNotIn(gone, js, gone)
        # Motor profiles retain their busy-only gate.
        self.assertIn("$('btn-start').disabled = busy;", js)
        self.assertNotRegex(js, r"btn-start.*disabled.*(?:calibrat|sign_verified)")
        opts = {o for a in pr.build_parser()._actions for o in a.option_strings}
        for flag in ('--allow-unverified-signs', '--home-check', '--home-tol-deg', '--policy-backend'):
            self.assertNotIn(flag, opts)

    def test_filter_runner_argv_drops_unknown_flags(self):
        kept, dropped = inference_jobs.filter_runner_argv(['--home-only', '--allow-unverified-signs', '--max-seconds', '5',
                                                     '--home-tol-deg', '3'])
        self.assertEqual(kept, ['--home-only', '--max-seconds', '5'])
        self.assertEqual(dropped, ['--allow-unverified-signs', '--home-tol-deg'])

    def test_server_error_triggers_emergency_stop(self):
        def boom():
            raise RuntimeError('synthetic failure')
        self.inference.state = boom
        st, _h, body = self.req('GET', '/api/state')
        self.assertEqual(st, 500)
        self.assertIn(b'synthetic failure', body)
        self.assertIn(RELEASE_ALL, self.bridge.messages)


class LiveCalibrationJobTests(JobBase):
    """The calibration job (always on the machine) calls hardware.calibrate.run in the worker thread with the hub's
    bridge and the job's stop event (a fake run here: the real one needs the Pi's link IMUs)."""

    def start_live(self, run):
        self.inference.cal_run = run
        st, job = self.post('/api/calibrate', {'prereqs_ok': True, 'visits': 1})       # no arming: it just starts
        self.assertEqual(st, 200, job)
        self.assertEqual((job['kind'], job['params']), ('calibrate', {'visits': 1, 'slew': True, 'command': job['params']['command']}))
        return job

    def test_gets_the_hub_bridge_the_stop_event_and_the_live_flags(self):
        calls = []

        def run(args, log=print, bridge=None, stop=None):
            calls.append((args, bridge, stop))
            log('1/6 check: streaming the boom, stick and bucket IMUs')
            return 0
        sse = BusReader(self.inference)
        try:
            self.start_live(run)
            end = sse.read_until(lambda e: e['type'] == 'job_end', timeout=30)
        finally:
            sse.close()
        self.assertEqual(end['data']['status'], 'done', end['data'].get('error'))
        ((args, bridge, stop),) = calls
        self.assertIs(bridge, self.bridge)
        self.assertIs(stop, self.inference.job.stop)
        self.assertTrue(args.live and args.i_understand_open_loop and args.yes and args.slew)
        self.assertEqual((args.visits, args.pi_host, args.hardware_map), (1, self.pi.host, str(self.hw_path)))
        self.assertTrue(any('1/6 check' in l for l in self.logs(sse.events)))
        self.assertEqual(end['data']['summary']['returncode'], 0)

    def test_a_failed_run_ends_with_an_error_and_releases(self):
        self.start_live(lambda args, log=print, bridge=None, stop=None: 1)
        j = self.wait_idle(30)
        self.assertEqual(j.status, 'error')
        self.assertIn('code 1', j.error)
        self.assertIn(RELEASE_ALL, self.bridge.messages)                       # a live job that did not finish

    def test_stop_aborts_a_running_calibration_then_release_all(self):
        started = threading.Event()

        def run(args, log=print, bridge=None, stop=None):
            started.set()
            if stop.wait(20):
                log('interrupted: releasing all pins')                   # what run() does on the stop event
                return 1
            return 0
        self.start_live(run)
        self.assertTrue(started.wait(20))
        t0 = time.time()
        self.assertTrue(self.post('/api/stop')[1]['stopped'])
        j = self.wait_idle(10)
        self.assertLess(time.time() - t0, 3.0)
        self.assertEqual((j.status, j.summary['returncode']), ('stopped', 1))
        self.assertIn(RELEASE_ALL, self.bridge.messages)

    def test_stop_interrupts_the_real_routine(self):
        # the CLI's simulated arm stands in for the machine (the hub job itself always passes --live)
        self.inference.calibration_plan = lambda body: {'visits': 2, 'slew': True,
                                                  'argv': ['--hardware-map', str(self.hw_path), '--artifacts',
                                                           str(self.tmp / 'hwcal')]}
        orig = inference_core.Inference._begin

        def begin(kind, params):                                          # stop set before the worker starts
            job = orig(self.inference, kind, params)
            job.stop.set()
            return job
        self.inference._begin = begin
        st, job = self.post('/api/calibrate', {})
        self.assertEqual(st, 200, job)
        j = self.wait_idle(30)
        self.assertEqual((j.status, j.summary['returncode']), ('stopped', 1))
        self.assertEqual(self.get_json('/api/calibration')['last_sim']['status'], 'interrupted')


class HubAppTests(Base):
    def test_start_without_the_gpio_app_reports_the_bridge_unavailable(self):
        class NoGpio:
            def service(self, app_id, wait=0.0):
                return None
        inference = hub_inference.start(NoGpio(), pi_host=self.pi.host, token_file=str(self.token),
                            hardware_map=str(self.hw_path), runs_dir=str(self.runs),
                            calibration_root=str(self.tmp / 'hwcal'))
        try:
            self.assertIsNone(inference.bridge)
            b = inference.poll_bridge()
            self.assertFalse(b['ok'])
            self.assertIn('unavailable', b['error'])
            self.assertEqual(inference.info()['bridge'], 'none')
            r = inference.emergency_stop('test')                      # never raises without a bridge
            self.assertFalse(r['bridge_release']['ok'])
            with self.assertRaises(inference_util.Refused) as cm:
                inference.start_calibrate({'prereqs_ok': True})
            self.assertEqual(cm.exception.status, 503)
            self.assertIsNone(inference.job)
        finally:
            inference.close()

    def test_the_bridge_is_looked_up_on_every_use(self):
        class Gpio:
            svc = None

            def service(self, app_id, wait=0.0):
                return self.svc if app_id == 'gpio' else None
        hub = Gpio()
        inference = hub_inference.start(hub, pi_host=self.pi.host, token_file=str(self.token), hardware_map=str(self.hw_path),
                            runs_dir=str(self.runs))
        try:
            self.assertFalse(inference.poll_bridge()['ok'])
            hub.svc = self.bridge                                # the gpio app (re)started
            b = inference.poll_bridge()
            self.assertTrue(b['ok'] and b['esp32'])
            self.assertEqual(b['target'], 'ble://fake-gpio')
            self.assertTrue(inference.emergency_stop('test')['bridge_release']['ok'])
            self.assertIn(RELEASE_ALL, self.bridge.messages)
        finally:
            inference.close()

    def test_reset_returns_the_policy_to_its_starting_state_in_place(self):
        old = self.inference
        old.obs = (b'jpeg', 'job', time.time(), None, '')
        st, r = self.post('/api/reset', {})
        self.assertEqual(st, 200, r)
        self.assertTrue(r['reset'] and r['cleared'])
        self.assertTrue(r['stop']['bridge_release']['ok'])
        self.assertIn(RELEASE_ALL, self.bridge.messages)
        self.assertIs(self.hub.service('inference'), old)              # the app is not restarted
        self.assertFalse(old._closing.is_set())
        self.assertIsNone(old.job)
        self.assertIsNone(old.obs)
        state = self.get_json('/api/state')
        self.assertNotIn('armed', state)
        self.assertIsNone(state['job'])

    def test_unknown_routes_and_bad_bodies(self):
        self.assertEqual(self.req('GET', '/api/nope')[0], 404)
        self.assertEqual(self.req('POST', '/api/nope', {})[0], 404)
        st, _h, body = self.req('POST', '/api/stop', headers={'Content-Type': 'application/json'}, body=None)
        self.assertEqual(st, 200, body)                         # empty body = {}
        r = self.client.post('/api/inference/stop', content=b'[1]', headers={'Content-Type': 'application/json'})
        self.assertEqual((r.status_code, r.json()['error']), (400, 'body must be a JSON object'))
        r = self.client.post('/api/inference/stop', content=b'{', headers={'Content-Type': 'application/json'})
        self.assertEqual(r.status_code, 400)
        r = self.client.post('/api/inference/stop', content=b'x' * (hub_inference.BODY_MAX + 1), headers={'Content-Type': 'application/json'})
        self.assertEqual(r.status_code, 413)


def dead_port():
    import socket
    sk = socket.socket(); sk.bind(('127.0.0.1', 0)); port = sk.getsockname()[1]; sk.close()
    return port                                   # nothing listens: connections are refused


class PiHostListTests(JobBase):
    def test_first_host_refuses_second_is_used(self):
        dead = f'127.0.0.1:{dead_port()}'
        spec = f'{dead},{self.pi.host}'
        pi = inference_pi.PiClient(spec, self.tmp / 'token', probe_timeout=1.0)
        self.assertEqual(pi.spec, spec)
        t0 = time.time()
        st = pi.status()
        self.assertLess(time.time() - t0, 2.5)
        self.assertEqual(st['camera']['fps'], 59.8)
        self.assertEqual(pi.base, f'http://{self.pi.host}')
        self.assertFalse(pi.last_probe[f'http://{dead}']['ok'])
        self.assertEqual(pi.link()['kind'], 'network')
        code, body, _ = pi.frame()
        self.assertEqual((code, body), (200, self.pi.jpeg))

    def test_reprobe_when_the_active_host_fails(self):
        other = FakePi()
        try:
            pi = inference_pi.PiClient(f'{other.host},{self.pi.host}', self.tmp / 'token')
            pi.status()
            self.assertEqual(pi.base, f'http://{other.host}')          # first in order wins
            other.close()
            pi.status()                                                 # fails, re-probes, retries
            self.assertEqual(pi.base, f'http://{self.pi.host}')
        finally:
            try:
                other.close()
            except Exception:
                pass

    def test_dashboard_with_a_list_and_verbatim_passthrough(self):
        spec = f'127.0.0.1:{dead_port()},{self.pi.host}'
        self.inference.pi = inference_pi.PiClient(spec, self.tmp / 'token')
        s_ = self.get_json('/api/sensors?refresh=1')
        self.assertTrue(s_['pi']['ok'])
        self.assertEqual(s_['pi']['link']['host'], self.pi.host)
        self.assertIsNotNone(s_['pi']['link']['status_ms'])
        self.assertEqual(self.req('GET', '/api/camera.jpg')[0], 200)
        captured = {}
        self.inference._launch_runner = lambda job, args: captured.setdefault('args', args) and job.info()
        self.post('/api/policy/start', self.policy_body())
        self.assertEqual(captured['args'].pi_host, spec)                # verbatim list to the runner
        self.inference.job.status = 'done'
        argv = self.inference.calibration_plan({'prereqs_ok': True})['argv']
        self.assertEqual(argv[argv.index('--pi-host') + 1], spec)       # and to hardware.calibrate


# ------------------------------------------------------------------------------------------- CLOAK checkpoints
def cloak_entry(url, **kw):
    return {'id': 'cloak-dirt-base', 'label': 'Dirt · base π0.5', 'backend': 'cloak', 'profile': 'pi05_tabletop_excavator',
            'policy_space': 'calibrated', 'gateway_url': url, 'state': 'configured', 'step': None,
            'optimizer_updates': 0, 'base_model': 'pi05_base', 'task': 'dirt', 'prompt': 'scoop the dirt',
            'mode': 'pretrained', 'default': True, **kw}


class CloakDashboardTests(Base):
    """pi05_tabletop_excavator dashboard runs end to end against a local CLOAK server double."""
    def setUp(self):
        super().setUp()
        hw = json.loads(self.hw_path.read_text())                  # live arm IMUs: the runner needs them
        hw['calibration']['arm_imus'] = arm_imu_cal()
        self.hw_path.write_text(json.dumps(hw, indent=2))
        self.pi.imus = {n: imu_entry(0.0) for n in ('boom', 'stick', 'bucket')}
        self.model = FakeCloak(metadata=with_md(weights='pretrained_initialization', mode='baseline'))
        self.addCleanup(self.model.close)
        self.inference.local_registry = self.tmp / 'checkpoints.json'
        self.inference.local_registry.write_text(json.dumps({'checkpoints': [cloak_entry(self.model.url)]}))

    def test_full_run_of_a_base_cloak_policy(self):
        body = {'profile': 'pi05_tabletop_excavator', 'gateway_url': self.model.url, 'prompt': 'scoop the dirt', 'chunk': 3,
                'hz': 10, 'max_seconds': 30, 'max_steps': 1}
        st, result = self.post('/api/policy/start', body)
        self.assertEqual(st, 200, result)
        self.assertEqual(result['params']['start_pose'], 'none')                  # the default: standard pose
        job = self.wait_idle()
        self.assertEqual(job.status, 'done', job.error)
        self.assertEqual(job.summary['steps'], 1)
        self.assertEqual(job.summary['policy_space'], 'calibrated')
        req = self.model.requests[0]
        self.assertEqual(req['state'].shape, (4,))
        self.assertEqual(req['rgb'].shape[:2], req['embodiment_mask'].shape)     # native frame, mask alongside
        self.assertFalse(req['embodiment_mask'].any())                            # baseline: nothing masked
        self.assertIn(RELEASE_ALL, self.bridge.messages)                         # sent to the machine, then released

    def test_add_and_remove_a_lambda_endpoint(self):
        new = FakeCloak(metadata=with_md(mode='cloak'))
        self.addCleanup(new.close)
        add = lambda **kw: self.post('/api/checkpoints/add', {'label': 'CLOAK · new run', 'gateway_url': new.url,
                                                                'prompt': 'scoop the bottle cap', **kw})
        self.assertEqual(add(gateway_url=self.model.url)[0], 400)                 # already registered
        self.assertEqual(add(gateway_url='http://127.0.0.1:1')[0], 400)           # not ws:// / wss://
        for md in ({'action_horizon': 10, 'action_contract': {'joints': ['boom', 'bucket'], 'units': 'radians',
                                                             'type': 'absolute next joint target', 'frequency_hz': 10}},
                   with_md(policy_space='handheld_reconstructed')):
            bad = FakeCloak(metadata=md)
            self.addCleanup(bad.close)
            st, err = add(gateway_url=bad.url)
            self.assertEqual(st, 400, err)                                        # wrong contract on the wire
            self.assertIn('not the expected 4-joint CLOAK policy', err['error'])
        self.assertEqual(add(gateway_url='ws://127.0.0.1:1')[0], 502)             # nothing listening
        st, entry = add()
        self.assertEqual(st, 200, entry)
        self.assertEqual((entry['backend'], entry['profile'], entry['policy_space'], entry['mode'], entry['prompt']),
                         ('cloak', 'pi05_tabletop_excavator', 'calibrated', 'cloak', 'scoop the bottle cap'))
        run = next(r for r in self.get_json('/api/checkpoints')['runs'] if r['key'] == entry['id'])
        self.assertEqual(run['checkpoints'][0]['gateway_url'], new.url)
        self.assertEqual(self.post('/api/checkpoints/remove', {'checkpoint_id': entry['id']})[0], 200)
        self.assertNotIn(entry['id'], [r['key'] for r in self.get_json('/api/checkpoints')['runs']])
        self.assertEqual(self.post('/api/checkpoints/remove', {'checkpoint_id': entry['id']})[0], 404)
        self.assertEqual(self.bridge.messages, [])

    def test_start_uses_only_registered_endpoints_and_real_joint_space(self):
        captured = {}
        self.inference._launch_runner = lambda job, args: captured.setdefault('args', args) and job.info()
        body = {'profile': 'pi05_tabletop_excavator', 'gateway_url': self.model.url, 'prompt': 'scoop the dirt', 'chunk': 3,
                'hz': 10, 'max_seconds': 30}
        st, result = self.post('/api/policy/start', {**body, 'gateway_url': 'wss://api.pi-fleet.com/v1/models/x'})
        self.assertEqual(st, 400, result)
        st, result = self.post('/api/policy/start', body)
        self.assertEqual(st, 200, result)
        self.inference.job.status = 'done'
        runner = pr.PolicyRunner(captured['args'], log=lambda *_: None)
        self.assertEqual(runner.policy_space, 'calibrated')
        self.assertFalse(hasattr(runner.args, 'live'))                         # every run moves the machine
        for gone in ('/api/checkpoints/preview', '/api/checkpoints/example.png?id=cloak-dirt-base'):
            self.assertEqual(self.req('POST' if 'preview' in gone else 'GET', gone, {} if 'preview' in gone else None)[0],
                             404)
        self.assertEqual(self.bridge.messages, [])


if __name__ == '__main__':
    unittest.main()
