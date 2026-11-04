import http.client
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from recorder.server import JPEGFrames, Server, State, decode_imu, decode_mpu6050, imu_worker, load_imu_config, preview_worker


class JPEGTests(unittest.TestCase):
    # Length-delimited APP data includes thumbnail-like marker bytes.
    payload = b"\xff\xd8" + b"\xff\xe0\x00\x08AB\xff\xd8\xff\xd9" + b"\xff\xda\x00\x02" + b"abc\xff\x00d\xff\xd0e\xff\xd9"

    def test_arbitrary_chunk_boundaries_and_multiple_frames(self):
        for size in (1, 2, 3, 7, 16, 1000):
            parser = JPEGFrames()
            frames = []
            data = b"unrelated startup output\n" + self.payload * 3
            for i in range(0, len(data), size):
                frames.extend(parser.feed(data[i:i+size]))
            self.assertEqual(frames, [self.payload] * 3)

    def test_partial_frame_is_not_emitted(self):
        parser = JPEGFrames()
        self.assertEqual(parser.feed(self.payload[:-1]), [])
        self.assertEqual(parser.feed(self.payload[-1:]), [self.payload])

    def test_corrupt_frame_resynchronizes(self):
        parser = JPEGFrames()
        self.assertEqual(parser.feed(b"\xff\xd8\xff\xda\x00\x02broken" + self.payload), [self.payload])


class SensorTests(unittest.TestCase):
    def test_signed_data_scales_and_axis_order(self):
        sample = decode_imu(struct.pack('<6h', -100, 200, -300, 8197, -8197, 0))
        self.assertEqual(sample['gyro'], [-1.7500000000000002, 3.5000000000000004, -5.250000000000001])
        self.assertAlmostEqual(sample['accel'][0], 1.000034)
        self.assertAlmostEqual(sample['accel'][1], -1.000034)
        self.assertFalse(sample['clipped'])

    def test_clipping_flag(self):
        self.assertTrue(decode_imu(struct.pack('<6h', 0, 0, -32768, 0, 0, 0))['clipped'])

    def test_mpu6050_is_big_endian_and_skips_temperature(self):
        sample = decode_mpu6050(struct.pack('>7h', 8192, -8192, 4096, 32767, 131, -655, 0))
        self.assertEqual(sample['accel'], [1.0, -1.0, 0.5])
        self.assertAlmostEqual(sample['gyro'][0], 2.0)
        self.assertAlmostEqual(sample['gyro'][1], -10.0)
        self.assertFalse(sample['clipped'])                              # the temperature word is not a reading
        self.assertTrue(decode_mpu6050(struct.pack('>7h', 0, 0, 0, 0, 0, 0, -32768))['clipped'])

    def test_mpu6050_worker_configures_reads_and_restores(self):
        regs = {0x75: 0x68, 0x6B: 0x40, 0x19: 0, 0x1A: 0, 0x1B: 0, 0x1C: 0, 0x38: 0, 0x3A: 1}
        state = State()
        state.configure_imus({'name': 'bucket', 'bus': 1, 'address': 0x6A, 'model': 'lsm6ds'},
                             [{'name': 'boom', 'bus': 5, 'address': 0x68, 'model': 'mpu6050'}])

        class Bus:
            def __init__(self, number):
                self.number = number
            def read_byte_data(self, address, reg):
                return regs[reg]
            def write_byte_data(self, address, reg, value):
                regs[reg] = value
            def read_i2c_block_data(self, address, reg, length):
                assert (address, reg, length) == (0x68, 0x3B, 14)
                state.stop.set()
                return list(struct.pack('>7h', 0, 8192, 0, 0, 0, 0, 0))
            def close(self):
                pass

        with mock.patch.dict(sys.modules, {'smbus2': types.SimpleNamespace(SMBus=Bus)}):
            imu_worker(state, {'name': 'boom', 'bus': 5, 'address': 0x68, 'model': 'mpu6050'})
        self.assertEqual(state.extra_imus['boom']['sample']['accel'], [0.0, 1.0, 0.0])
        self.assertEqual(state.extra_imus['boom']['model'], 'mpu6050')   # kept for the worker the server starts
        self.assertEqual(regs[0x6B], 0x40)                               # put back to sleep as found
        self.assertEqual(regs[0x1C], 0)

    def test_stale_samples_cannot_look_live(self):
        state = State()
        state.frame_received(JPEGTests.payload)
        state.sample_received({'accel': [0, 0, 1], 'gyro': [0, 0, 0]})
        self.assertTrue(state.snapshot()['imu']['live'])
        with state.condition:
            state.imu_times[-1] = time.monotonic() - 2
            state.camera_times[-1] = time.monotonic() - 3
        snapshot = state.snapshot()
        self.assertFalse(snapshot['camera']['live'])
        self.assertFalse(snapshot['imu']['live'])
        self.assertIsNone(snapshot['imu']['sample'])


class ExtraIMUTests(unittest.TestCase):
    def test_missing_config_is_the_bucket_imu_only(self):
        primary, others = load_imu_config(Path('/nonexistent/imus.json'))
        self.assertEqual((primary['name'], primary['bus'], primary['address']), ('bucket', 1, 0x6A))
        self.assertEqual(others, [])

    def test_bucket_config_parses_and_rejects_bad_entries(self):
        primary, others = load_imu_config(Path(__file__).parents[1] / 'devices' / 'imus-bucket.json')
        self.assertEqual(primary['name'], 'bucket')
        self.assertEqual({e['name']: (e['bus'], e['address'], e['model']) for e in others},
                         {'stick': (3, 0x68, 'mpu6050'), 'boom': (5, 0x68, 'mpu6050')})
        self.assertEqual(primary['model'], 'lsm6ds')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'imus.json'
            for bad in ([{'name': 'a', 'bus': 1, 'address': '0x6a'}, {'name': 'b', 'bus': 1, 'address': 0x6A}],
                        [{'name': 'a', 'bus': 1, 'address': '0x1c'}],
                        [{'name': 'a', 'bus': 1, 'address': '0x68'}],                      # an MPU6050 address, model lsm6ds
                        [{'name': 'a', 'bus': 1, 'address': '0x6a', 'model': 'mpu6050'}],
                        [{'name': 'a', 'bus': 1, 'address': '0x68', 'model': 'bmi160'}],
                        [{'name': 'a', 'bus': 1, 'address': '0x6a'}, {'name': 'a', 'bus': 3, 'address': '0x6a'}], []):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_imu_config(path)

    def test_extra_imus_are_reported_but_never_recorded_or_primary(self):
        state = State()
        state.configure_imus({'name': 'bucket', 'bus': 1, 'address': 0x6A},
                             [{'name': 'boom', 'bus': 5, 'address': 0x6A}])
        state.sample_received({'accel': [0, 1, 0], 'gyro': [0, 0, 0]}, 'boom')
        snapshot = state.snapshot()
        self.assertFalse(snapshot['imu']['live'])                     # the primary saw nothing
        self.assertTrue(snapshot['imus']['boom']['live'])
        self.assertEqual(snapshot['imus']['boom']['sample']['accel'], [0, 1, 0])
        for k in range(20):
            state.sample_received({'accel': [0, 1 + k % 2 * 0.1, 0], 'gyro': [0, 0, 0]}, 'boom')
        boom = state.snapshot()['imus']['boom']
        self.assertEqual(boom['mean_samples'], 12)
        self.assertAlmostEqual(boom['accel_mean'][1], 1.05)
        self.assertFalse(snapshot['imus']['bucket']['live'])
        with state.condition:
            state.extra_imus['boom']['times'][-1] = time.monotonic() - 2
        self.assertIsNone(state.snapshot()['imus']['boom']['sample'])
        state.extra_imu_error('boom', 'unplugged')
        state.sample_received({'accel': [0, 0, 1], 'gyro': [0, 0, 0]})
        snapshot = state.snapshot()
        self.assertTrue(snapshot['imus']['bucket']['live'] and snapshot['imu']['live'])
        self.assertEqual(snapshot['imus']['boom']['error'], 'unplugged')


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = State()
        cls.server = Server(('127.0.0.1', 0), cls.state, 'test-token')
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.state.stop.set()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(2)

    def request(self, path, auth=True):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        conn.request('GET', path, headers={'Cookie': 'umi_access=test-token'} if auth else {})
        response = conn.getresponse()
        body = response.read()
        status, headers = response.status, dict(response.getheaders())
        conn.close()
        return status, headers, body

    def setUp(self):
        self.state = State()
        self.server.state = self.state

    def test_no_auth_no_device_data(self):
        for path in ('/', '/api/status', '/api/imu/recent', '/events', '/stream.mjpg', '/snapshot.jpg', '/preview.jpg', '/api/recordings/status', '/api/recordings/download'):
            self.assertEqual(self.request(path, False)[0], 401)

    def test_private_link_sets_cookie_and_cleans_url(self):
        status, headers, _ = self.request('/?token=test-token', False)
        self.assertEqual(status, 303)
        self.assertEqual(headers['Location'], '/')
        self.assertIn('HttpOnly', headers['Set-Cookie'])
        self.assertEqual(self.request('/?token=wrong', False)[0], 401)

    def test_static_files_and_no_traversal(self):
        for path in ('/', '/app.js', '/pi.js', '/recording.js', '/latest-poll.js', '/style.css'):
            self.assertEqual(self.request(path)[0], 200)
        for path in ('/../access-token', '/server.py', '/access-token', '/calibrate', '/target.svg', '/api/calibration/status'):
            self.assertEqual(self.request(path)[0], 404)

    def test_recording_requires_auth_and_explicit_action_header(self):
        for headers,expected in [({},401),({'Cookie':'umi_access=test-token'},403),
                                 ({'Cookie':'umi_access=test-token','Content-Type':'application/json','X-UMI-Action':'calibration'},403),
                                 ({'Cookie':'umi_access=test-token','Content-Type':'application/json','X-UMI-Action':'recording'},409)]:
            conn=http.client.HTTPConnection(*self.server.server_address,timeout=3)
            conn.request('POST','/api/recordings/start',body='{}',headers=headers)
            response=conn.getresponse();response.read();conn.close()
            self.assertEqual(response.status,expected)

    def test_status_does_not_fabricate_samples(self):
        status, _, body = self.request('/api/status')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertFalse(data['imu']['live'])
        self.assertIsNone(data['imu']['sample'])
        self.assertEqual(self.request('/snapshot.jpg')[0], 503)
        self.assertEqual(self.request('/preview.jpg')[0], 503)

    def test_imu_history_preserves_samples_but_never_reports_stale_data(self):
        for i in range(900):
            self.state.sample_received({'read_monotonic_ns': i, 'gyro': [0, 0, 0],
                                       'accel': [0, 0, 1], 'clipped': False})
        result = json.loads(self.request('/api/imu/recent')[2])
        self.assertTrue(result['live'])
        self.assertEqual([s['read_monotonic_ns'] for s in result['samples']], list(range(68, 900)))
        self.state.imu_times[-1] = time.monotonic()-1
        result = json.loads(self.request('/api/imu/recent')[2])
        self.assertFalse(result['live'])
        self.assertEqual(result['samples'], [])

    def test_extra_imu_history_by_name(self):
        state = State()
        state.configure_imus({'name': 'bucket', 'bus': 1, 'address': 0x6A, 'model': 'lsm6ds'},
                             [{'name': 'boom', 'bus': 5, 'address': 0x68, 'model': 'mpu6050'}])
        server = Server(('127.0.0.1', 0), state, 'test-token')
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for i in range(900):
                state.sample_received({'read_monotonic_ns': i, 'gyro': [0, 0, 0], 'accel': [0, 1, 0],
                                       'clipped': False}, 'boom')
            conn = http.client.HTTPConnection(*server.server_address, timeout=3)
            conn.request('GET', '/api/imu/recent?imu=boom', headers={'Cookie': 'umi_access=test-token'})
            result = json.loads(conn.getresponse().read())
            self.assertTrue(result['live'])
            self.assertEqual([s['read_monotonic_ns'] for s in result['samples']], list(range(68, 900)))
            conn.request('GET', '/api/imu/recent?imu=nope', headers={'Cookie': 'umi_access=test-token'})
            self.assertEqual(conn.getresponse().status, 404)
            conn.close()
        finally:
            state.stop.set(); server.shutdown(); server.server_close()

    def test_preview_returns_latest_only_and_rejects_old_or_duplicate_images(self):
        self.state.frame_received(b'original-720p')
        self.state.preview = (b'first', 1, time.monotonic(), 640, 360)
        status, headers, body = self.request('/preview.jpg')
        self.assertEqual((status, body), (200, b'first'))
        self.assertEqual(headers['X-Frame-Sequence'], '1')
        self.assertGreaterEqual(float(headers['X-Frame-Age-Ms']), 0)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(self.request('/preview.jpg?after=1')[0], 204)
        for sequence in range(2, 50):
            self.state.preview = (str(sequence).encode(), sequence, time.monotonic(), 640, 360)
        status, headers, body = self.request('/preview.jpg?after=1')
        self.assertEqual(body, b'49')
        self.assertEqual(headers['X-Frame-Sequence'], '49')
        self.state.preview = (b'old', 50, time.monotonic() - 1, 640, 360)
        self.assertEqual(self.request('/preview.jpg')[0], 204)
        self.assertEqual(self.request('/snapshot.jpg')[2], b'original-720p')
        self.state.error('camera', 'Disconnected')
        self.assertEqual(self.request('/preview.jpg')[0], 503)

    def test_preview_recovers_after_server_sequence_restarts(self):
        self.state.frame_received(b'raw')
        self.state.preview = (b'restarted', 1, time.monotonic(), 640, 360)
        self.assertEqual(self.request('/preview.jpg?after=99999')[2], b'restarted')

    def test_preview_encoder_preserves_snapshot_and_original_timestamp(self):
        from PIL import Image
        output = io.BytesIO()
        Image.new('RGB', (1280, 720), (128, 32, 64)).save(output, format='JPEG')
        original = output.getvalue()
        self.state.frame_received(original)
        received_at = self.state.camera_times[-1]
        worker = threading.Thread(target=preview_worker, args=(self.state,), daemon=True)
        worker.start()
        try:
            time.sleep(.06)
            self.assertIsNone(self.state.preview, 'Do not encode when nobody is viewing')
            self.request('/preview.jpg')
            deadline = time.monotonic() + 2
            while self.state.preview is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIsNotNone(self.state.preview)
            frame, sequence, timestamp, width, height = self.state.preview
            self.assertEqual((width, height, sequence, timestamp), (640, 360, 1, received_at))
            with Image.open(io.BytesIO(frame)) as image:
                self.assertEqual(image.size, (640, 360))
            self.assertEqual(self.state.frame, original)
            self.assertLess(len(frame), len(original))
        finally:
            self.state.stop.set()
            with self.state.condition:
                self.state.condition.notify_all()
            worker.join(2)

    def test_events_deliver_json(self):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        conn.request('GET', '/events', headers={'Cookie': 'umi_access=test-token'})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        line = response.fp.readline()
        self.assertIn('camera', json.loads(line.decode()[6:]))
        conn.close()
