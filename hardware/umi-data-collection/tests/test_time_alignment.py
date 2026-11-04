import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import unittest

from recorder.recording import Recorder
from recorder.server import State
from recorder.time_alignment import SIGN, TimeAlignment


def profile(offset=.020234615732440238):
    return {'status': 'validated_software_offset', 'reasons': [], 'camera_to_imu_offset_s': offset,
            'recording_id': '20260927T021212Z-d9180d1d', 'recording_sha256': 'a'*64,
            'camera_intrinsics': {'image_size': [1280, 720]}, 'sign_convention': SIGN,
            'deployment': {'hostname': 'bucket', 'timestamp_source': 'start_of_exposure'}}


def frame(timestamp=1_000_000_000):
    return {'camera_timestamp_ns': timestamp, 'clock': 'CLOCK_MONOTONIC', 'timestamp_source': 'start_of_exposure',
            'buffer_error': False, 'width': 1280, 'height': 720, 'sequence': 42, 'flags': 0x12001,
            'dequeue_monotonic_ns': timestamp + 1_000_000}


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'time-calibration.json'

    def tearDown(self):
        self.directory.cleanup()

    def load(self, data=None, hostname='bucket'):
        self.path.write_text(json.dumps(data if data is not None else profile()))
        return TimeAlignment(self.path, hostname)

    def test_positive_and_negative_offsets_apply_once_and_preserve_raw_input(self):
        for offset in (.020234615732440238, -.043):
            alignment = self.load(profile(offset))
            original = frame()
            result = alignment.apply(original)
            self.assertEqual(result['camera_on_imu_host_clock_ns'], 1_000_000_000 + round(offset * 1e9))
            self.assertEqual(result['camera_timestamp_ns'], original['camera_timestamp_ns'])
            self.assertNotIn('camera_on_imu_host_clock_ns', original)
            self.assertEqual(alignment.apply(result), result)
            self.assertTrue(alignment.snapshot(result)['applied'])

    def test_missing_file_rejected_result_and_invalid_offsets_stay_raw(self):
        self.assertFalse(TimeAlignment(self.path).snapshot()['configured'])
        invalid = [dict(profile(), status='rejected'), dict(profile(), reasons=['Bad fit']),
                   dict(profile(), sign_convention='subtract')]
        invalid += [profile(v) for v in (float('nan'), float('inf'), 1, True)]
        for data in invalid:
            alignment = self.load(data)
            self.assertFalse(alignment.snapshot()['configured'])
            self.assertTrue(alignment.error)
            self.assertNotIn('camera_on_imu_host_clock_ns', alignment.apply(frame()))

    def test_different_pi_cannot_load_this_rigs_offset(self):
        alignment = self.load(hostname='another-pi')
        self.assertFalse(alignment.snapshot()['configured'])
        self.assertIn('different Pi', alignment.error)

    def test_incompatible_frames_do_not_keep_a_stale_correction(self):
        alignment = self.load()
        aligned = alignment.apply(frame())
        for change in ({'clock': 'unknown'}, {'width': 640}, {'height': 480},
                       {'timestamp_source': 'end_of_frame'}, {'buffer_error': True}):
            result = alignment.apply(dict(aligned, **change))
            self.assertNotIn('camera_on_imu_host_clock_ns', result)
            self.assertFalse(alignment.snapshot(result)['applied'])

    def test_invalid_file_does_not_stop_camera_service(self):
        self.path.write_text('{not-json')
        state = State(TimeAlignment(self.path, 'bucket'))
        state.frame_received(b'jpeg', frame())
        result = state.snapshot()
        self.assertTrue(result['camera']['live'])
        self.assertFalse(result['time_alignment']['configured'])
        self.assertNotIn('camera_on_imu_host_clock_ns', result['camera']['timing'])

    def test_live_status_and_recorded_archive_agree_and_keep_raw_samples(self):
        alignment = self.load()
        for incompatible_last_frame in (False, True):
            state = State(alignment)
            recorder = state.recorder = Recorder(Path(self.directory.name) / 'recordings')
            status = recorder.start({'time_calibration': alignment.profile}, duration=2)
            base = time.monotonic_ns()
            sample = {'read_monotonic_ns': base, 'read_duration_ns': 3_000_000,
                      'gyro': [1, 2, 3], 'accel': [4, 5, 6], 'clipped': False}
            state.sample_received(sample)
            for i in range(3):
                metadata = dict(frame(base + i * 50_000_000), sequence=i)
                if i == 2 and incompatible_last_frame:
                    metadata['width'] = 640
                state.frame_received(bytes([i]), metadata)
            recorder.stop(); recorder.thread.join(3)
            self.assertEqual(recorder.snapshot()['status'], 'ready')
            live = state.snapshot()
            self.assertEqual(live['time_alignment']['applied'], not incompatible_last_frame)
            self.assertEqual(live['imu']['sample'], sample)
            with tarfile.open(recorder.archive(status['id'])) as archive:
                rows = list(csv.DictReader(io.TextIOWrapper(archive.extractfile('frames.csv'))))
                imu = list(csv.DictReader(io.TextIOWrapper(archive.extractfile('imu.csv'))))
                meta = json.load(archive.extractfile('metadata.json'))
                self.assertEqual(len(rows), 3)
                for i, row in enumerate(rows):
                    self.assertEqual(int(row['camera_timestamp_ns']), base + i * 50_000_000)
                    if i == 2 and incompatible_last_frame:
                        self.assertEqual(row['camera_on_imu_host_clock_ns'], '')
                    else:
                        self.assertEqual(int(row['camera_on_imu_host_clock_ns']), int(row['camera_timestamp_ns']) + 20234616)
                        self.assertEqual(row['time_calibration_id'], profile()['recording_id'])
                self.assertEqual(int(imu[0]['read_monotonic_ns']), base)
                self.assertEqual(meta['format_version'], 2)
                self.assertEqual(meta['time_calibration']['offset_ns'], 20234616)
                self.assertEqual(set(meta['time_calibration']), {'recording_id', 'recording_sha256', 'offset_ns',
                                 'image_size', 'timestamp_source', 'sign_convention'})
