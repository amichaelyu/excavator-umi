import csv
import io
import json
import tarfile
import tempfile
import time
import unittest

from recorder.recording import Recorder
from recorder.capture import HEADER, TimedFrames


class TimedCaptureTests(unittest.TestCase):
    def test_timestamps_survive_arbitrary_chunking(self):
        jpeg = b'\xff\xd8test-image\xff\xd9'
        wire = HEADER.pack(b'UMI1',len(jpeg),83,0x12000,123456789000,123470000000,1280,720)+jpeg
        for size in (1,3,40,79,1000):
            parser=TimedFrames(); result=[]
            for i in range(0,len(wire),size): result.extend(parser.feed(wire[i:i+size]))
            self.assertEqual(len(result),1)
            self.assertEqual(result[0][0],jpeg)
            meta=result[0][1]
            self.assertEqual(meta['camera_timestamp_ns'],123456789000)
            self.assertEqual(meta['sequence'],83)
            self.assertEqual(meta['clock'],'CLOCK_MONOTONIC')
            self.assertEqual(meta['timestamp_source'],'start_of_exposure')

    def test_unknown_clock_is_not_called_monotonic(self):
        result=TimedFrames().feed(HEADER.pack(b'UMI1',1,1,0,4,5,1280,720)+b'a')
        self.assertEqual(result[0][1]['clock'],'unknown')

    def test_corrupt_length_is_rejected_before_allocation(self):
        with self.assertRaises(ValueError):
            TimedFrames().feed(HEADER.pack(b'UMI1',100_000_000,1,0,4,5,1280,720))


class RecordingTests(unittest.TestCase):
    def test_preserves_timestamped_data_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder=Recorder(directory)
            status=recorder.start({'test':True},duration=2)
            with self.assertRaises(ValueError): recorder.start({})
            base=time.monotonic_ns()
            for index in range(9):
                meta={'camera_timestamp_ns':base+index*16_666_667,'dequeue_monotonic_ns':base+index*16_666_667+1_000_000,
                      'receipt_monotonic_ns':base+index*16_666_667+2_000_000,'sequence':index,'flags':0x2000,'width':1280,'height':720}
                recorder.put('frame',(bytes([index]),meta))
                recorder.put('imu',{'read_monotonic_ns':base+index*5_000_000,'read_duration_ns':2_000_000,
                                    'gyro':[1,2,3],'accel':[4,5,6],'clipped':False})
            recorder.stop(); recorder.thread.join(3)
            self.assertEqual(recorder.snapshot()['status'],'ready')
            with tarfile.open(recorder.archive(status['id'])) as archive:
                frames=list(csv.DictReader(io.TextIOWrapper(archive.extractfile('frames.csv'))))
                imu=list(csv.DictReader(io.TextIOWrapper(archive.extractfile('imu.csv'))))
                self.assertEqual([int(r['sequence']) for r in frames],[0,3,6])
                self.assertEqual(int(frames[0]['camera_timestamp_ns']),base)
                self.assertEqual(archive.extractfile('video.mjpg').read(),bytes([0,3,6]))
                self.assertEqual(len(imu),9)
                metadata = json.load(archive.extractfile('metadata.json'))
                self.assertEqual(metadata['frames'], 3)
                self.assertEqual(set(metadata), {'time_calibration', 'imu_config', 'started_monotonic_s',
                                 'duration_s', 'format_version', 'id', 'label', 'frames', 'imu_samples', 'dropped',
                                 'clipped_samples', 'error'})
                self.assertTrue(all(r['camera_on_imu_host_clock_ns'] == '' for r in frames))
            with self.assertRaises(ValueError):recorder.archive('../access-token')

    def test_deadline_finishes_even_when_devices_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder=Recorder(directory); recorder.start({},duration=.05); recorder.thread.join(2)
            self.assertEqual(recorder.snapshot()['status'],'ready')
