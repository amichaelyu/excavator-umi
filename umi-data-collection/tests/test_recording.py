import csv
import http.client
import io
import json
from pathlib import Path
import queue
import shutil
import tarfile
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import av
from PIL import Image
from recorder.recording import Recorder
from recorder.download import RecordingDownload
from recorder.server import Server, State
from test_time_alignment import frame


def jpeg():
    output = io.BytesIO()
    Image.new('RGB', (64, 48), (60, 140, 90)).save(output, 'JPEG')
    return output.getvalue()


def feed(recorder):
    base = time.monotonic_ns()
    for index, delta in enumerate((0, 50_000_000, 120_321_000)):
        meta = dict(frame(base+delta), receipt_monotonic_ns=base+delta+2_000_000,
                    sequence=index, width=64, height=48, camera_on_imu_host_clock_ns=base+delta+20_234_616,
                    camera_to_imu_offset_ns=20_234_616, time_calibration_id='test-profile')
        recorder.put('frame', (jpeg(), meta))
        recorder.put('imu', {'read_monotonic_ns':base+delta, 'read_duration_ns':1_000_000,
                            'gyro':[1,2,3], 'accel':[0,0,1], 'clipped':False})
    return base


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.recorder = Recorder(self.root)

    def tearDown(self):
        self.recorder.shutdown()
        if self.recorder.thread: self.recorder.thread.join(10)
        self.temp.cleanup()

    def start(self):
        return self.recorder.start({}, duration=None, mode='task', label='Dig dirt')

    def finish(self):
        self.recorder.stop(); self.recorder.thread.join(10)
        self.assertFalse(self.recorder.thread.is_alive())
        return self.recorder.snapshot()

    def test_long_take_mp4_timing_alignment_and_persistent_history(self):
        started = self.start()
        with self.recorder.lock: self.recorder.started -= 3600
        base = feed(self.recorder)
        self.assertEqual(self.recorder.snapshot()['status'], 'recording')
        saved = self.finish()
        self.assertEqual(saved['status'], 'ready', saved)
        self.assertEqual(saved['video_status'], 'ready', saved)
        self.assertGreater(saved['elapsed_s'], 3600)
        self.assertEqual(self.recorder.snapshot()['elapsed_s'], saved['elapsed_s'])
        with av.open(str(self.recorder.video(started['id']))) as video:
            times = [float(f.pts*f.time_base) for f in video.decode(video=0)]
        self.assertEqual(len(times), 3)
        for actual, expected in zip(times, (0, .05, .120321)): self.assertAlmostEqual(actual, expected, places=6)
        with (self.root/started['id']/'frames.csv').open() as stream: rows = list(csv.DictReader(stream))
        self.assertEqual(int(rows[0]['camera_timestamp_ns']), base)
        self.assertEqual(int(rows[0]['camera_on_imu_host_clock_ns']), base+20_234_616)
        self.assertFalse((self.root/(started['id']+'.tar')).exists())
        next_id = self.start()['id']
        with self.assertRaises(ValueError): self.recorder.stop(started['id'])
        self.assertEqual(self.recorder.snapshot()['id'], next_id)
        self.assertEqual(self.recorder.snapshot()['status'], 'recording')
        feed(self.recorder); self.finish()
        reopened = Recorder(self.root)
        self.assertEqual(len(reopened.list_saved()), 2)
        self.assertTrue(reopened.video(started['id']).is_file())

    def test_camera_and_imu_disconnect_save_captured_data(self):
        for device in ('camera', 'imu'):
            with self.subTest(device=device):
                self.start(); feed(self.recorder)
                state = State(); state.recorder = self.recorder; state.error(device, 'Device removed')
                result = self.finish()
                self.assertEqual(result['video_status'], 'ready', result)
                self.assertIn(device.upper() if device == 'imu' else 'Camera', result['error'])
                self.assertTrue(result['raw_data_saved'])
                metadata = self.recorder.entry(result['id'])
                self.assertEqual(metadata['frames'], 3)

    def test_silent_imu_stall_stops_take(self):
        self.recorder.health = lambda: {'camera':{'live':True}, 'imu':{'live':False}}
        self.start(); self.recorder.thread.join(5)
        result = self.recorder.snapshot()
        self.assertEqual(result['status'], 'ready')
        self.assertIn('IMU', result['error']); self.assertEqual(result['video_status'], 'error')

    def test_raw_camera_imu_capture_skips_encoder_and_can_start_next_take(self):
        with patch('recorder.video_export.check_video_support', side_effect=AssertionError('encoder dependency accessed')), \
             patch('recorder.video_export.export_video', side_effect=AssertionError('encoder called')):
            self.recorder.start({}, duration=None, mode='task', label='hwcal speeds', export_video=False)
            base = feed(self.recorder)
            result = self.finish()
        self.assertEqual(result['status'], 'ready', result)
        self.assertFalse(result['video_export_requested'])
        self.assertEqual(result['video_status'], 'none')
        self.assertEqual(result['warning'], '')
        self.assertTrue(result['raw_data_saved'])
        files = self.recorder.data_files(result['id'])
        self.assertEqual({p.name for p in files}, {'metadata.json', 'frames.csv', 'imu.csv', 'video.mjpg'})
        folder = self.root/result['id']
        with (folder/'frames.csv').open() as f:
            frames = list(csv.DictReader(f))
        self.assertEqual(int(frames[0]['camera_timestamp_ns']), base)
        self.assertEqual(len(frames), 3)
        self.assertEqual((folder/'video.mjpg').read_bytes(), jpeg()*3)
        self.assertEqual(len((folder/'imu.csv').read_text().splitlines()), 4)
        self.assertFalse((folder/'video.mp4').exists())
        self.recorder.start({}, duration=None, mode='task', export_video=False)
        feed(self.recorder)
        self.assertEqual(self.finish()['status'], 'ready')

    def test_encoder_failure_keeps_raw_download(self):
        with patch('recorder.video_export.export_video', side_effect=OSError('encoder failed')):
            self.start(); feed(self.recorder); result = self.finish()
        self.assertEqual(result['video_status'], 'error')
        self.assertIn('encoder failed', result['warning'])
        self.assertEqual(len(self.recorder.data_files(result['id'])), 4)
        with self.assertRaises(ValueError): self.recorder.video(result['id'])

    def test_low_disk_prevents_start_and_stops_active_take(self):
        usage = shutil.disk_usage(self.root)
        with patch('recorder.recording.shutil.disk_usage', return_value=usage._replace(free=500_000_000)):
            with self.assertRaisesRegex(ValueError, 'free space'): self.start()
        self.start(); feed(self.recorder)
        with patch('recorder.recording.shutil.disk_usage', return_value=usage._replace(free=500_000_000)): self.recorder._monitor()
        result = self.finish()
        self.assertIn('Storage', result['error']); self.assertTrue(result['raw_data_saved'])

    def test_queue_overflow_and_bad_timing_stop_instead_of_silent_loss(self):
        self.start()
        with self.recorder.lock:
            with patch.object(self.recorder.queue, 'put_nowait', side_effect=queue.Full): feed(self.recorder)
        result = self.finish()
        self.assertEqual(result['dropped'], 1); self.assertIn('SD card', result['error'])
        self.start(); feed(self.recorder)
        self.recorder.put('frame', (jpeg(), dict(frame(), clock='unknown')))
        self.assertIn('timing changed', self.finish()['error'])

    def test_shutdown_and_interrupted_restart_retain_data(self):
        self.start(); feed(self.recorder); self.recorder.shutdown(); result = self.finish()
        self.assertTrue(result['raw_data_saved']); self.assertIn('shut down', result['error'])
        directory = self.root/result['id']
        meta = json.loads((directory/'metadata.json').read_text()); meta.update(error='')
        (directory/'metadata.json').write_text(json.dumps(meta))
        status = json.loads((directory/'status.json').read_text()); status.update(status='recording', raw_data_saved=False)
        (directory/'status.json').write_text(json.dumps(status))
        restarted = Recorder(self.root); recovered = restarted.list_saved()[0]
        self.assertEqual(recovered['status'], 'error'); self.assertIn('restarted', recovered['error'])
        self.assertEqual(len(restarted.data_files(result['id'])), 4)

    def test_archive_snapshot_and_length_even_if_metadata_changes(self):
        self.start(); feed(self.recorder); result = self.finish()
        archive = RecordingDownload(self.recorder.data_files(result['id']))
        path = self.root/result['id']/'metadata.json'; original = path.read_bytes(); path.write_text('{}')
        output = io.BytesIO(); archive.write(output)
        self.assertEqual(len(output.getvalue()), archive.size)
        with tarfile.open(fileobj=io.BytesIO(output.getvalue())) as data:
            self.assertEqual(set(data.getnames()), {'metadata.json', 'frames.csv', 'imu.csv', 'video.mjpg'})
            self.assertEqual(data.extractfile('metadata.json').read(), original)
            self.assertEqual(len(data.extractfile('video.mjpg').read()), result['bytes_written'])
        for method in (self.recorder.data_files, self.recorder.video, self.recorder.archive):
            with self.assertRaises(ValueError): method('../access-token')

    def test_write_failure_is_visible_and_retains_files(self):
        with patch('recorder.recording.os.fsync', side_effect=OSError('SD card removed')):
            self.start(); feed(self.recorder); result=self.finish()
        self.assertEqual(result['status'], 'error'); self.assertIn('SD card removed', result['error'])
        self.assertTrue((self.root/result['id']/'video.mjpg').exists())


class RecordingAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.state = State()
        self.state.recorder = Recorder(self.temp.name)
        self.server = Server(('127.0.0.1', 0), self.state, token='test-token')
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True); self.worker.start()

    def tearDown(self):
        self.state.recorder.shutdown()
        if self.state.recorder.thread: self.state.recorder.thread.join(10)
        self.server.shutdown(); self.server.server_close(); self.worker.join(2); self.temp.cleanup()

    def request(self, path, body=None, auth=True, header=True):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        headers = {'Cookie':'umi_access=test-token'} if auth else {}
        if body is not None:
            headers['Content-Type']='application/json'
            if header: headers['X-UMI-Action']='recording'
        connection.request('GET' if body is None else 'POST', path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = connection.getresponse(); status=response.status; data=response.read(); connection.close()
        return status,data

    def test_raw_capture_api_preserves_defaults_and_validates_flag(self):
        self.state.frame_received(jpeg(), dict(frame(), width=64, height=48))
        self.state.sample_received({'accel': [0, 0, 1], 'gyro': [0, 0, 0]})
        self.assertEqual(self.request('/api/recordings/start', {'export_video': 'false'})[0], 409)
        code, body = self.request('/api/recordings/start', {'label': 'hwcal ranges', 'export_video': False})
        self.assertEqual(code, 200, body)
        started = json.loads(body)
        self.assertFalse(started['video_export_requested'])
        feed(self.state.recorder)
        self.request('/api/recordings/stop', {'id': started['id']})
        self.state.recorder.thread.join(10)
        self.assertEqual(self.state.recorder.snapshot()['status'], 'ready')
        code, data = self.request('/api/recordings/download?kind=data&id='+started['id'])
        self.assertEqual(code, 200)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            self.assertEqual(len(archive.getnames()), 4)
        self.assertEqual(self.request('/api/recordings/download?kind=video&id='+started['id'])[0], 409)

    def test_auth_validation_start_stop_and_download(self):
        self.assertEqual(self.request('/api/recordings', auth=False)[0], 401)
        self.assertEqual(self.request('/api/recordings/start', {}, header=False)[0], 403)
        self.assertEqual(self.request('/api/recordings/start', [])[0], 409)
        self.assertEqual(self.request('/api/recordings/start', {})[0], 409)
        self.state.frame_received(jpeg(), dict(frame(),width=64,height=48))
        self.state.sample_received({'accel':[0,0,1],'gyro':[0,0,0]})
        self.assertEqual(self.request('/api/recordings/start', {'label':'x'*81})[0], 409)
        code, body = self.request('/api/recordings/start', {'label':'First take'})
        self.assertEqual(code, 200, body); started = json.loads(body)
        self.assertIsNone(started['duration_s'])
        self.assertEqual(self.request('/api/recordings/start', {})[0], 409)
        self.assertEqual(self.request('/api/recordings/stop', {})[0], 409)
        self.assertEqual(self.request('/api/recordings/stop', {'id':'wrong'})[0], 409)
        self.assertEqual(self.request('/api/recordings/download?kind=data&id='+started['id'])[0], 409)
        feed(self.state.recorder)
        self.assertEqual(self.request('/api/recordings/stop', {'id':started['id']})[0], 200)
        self.state.recorder.thread.join(10)
        for kind in ('video','data'):
            path=f"/api/recordings/download?kind={kind}&id={started['id']}"
            self.assertEqual(self.request(path, auth=False)[0], 401)
            code,data = self.request(path); self.assertEqual(code, 200, data[:200])
            if kind == 'video':
                with av.open(io.BytesIO(data)) as video: self.assertEqual(len(list(video.decode(video=0))), 3)
            else:
                with tarfile.open(fileobj=io.BytesIO(data)) as archive: self.assertEqual(len(archive.getnames()), 4)
        code,data=self.request('/api/recordings'); self.assertEqual(json.loads(data)[0]['label'], 'First take')
        self.assertEqual(self.request('/record')[0], 200)
        self.assertEqual(self.request('/api/recordings/download?kind=data&id=../../access-token')[0], 409)
