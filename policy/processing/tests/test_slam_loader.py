"""Tests for the tolerant SLAM recording loader (synthetic tar, no dataset needed)."""
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from policy.processing.slam import recording  # noqa: E402


def make_tar(root: Path, eid: str, imu_stop_s=None, corrupt_frame=None, truncate_csv=False):
    rng = np.random.default_rng(0)
    img = (rng.random((720, 1280)) * 255).astype(np.uint8)
    jpg = cv2.imencode('.jpg', img)[1].tobytes()
    t0 = 100.0
    video, frows = b'', []
    for i in range(60):
        data = jpg if i != corrupt_frame else b'\xff\xd8garbage'
        frows.append(f'{int((t0 + 0.05 * i) * 1e9)},0,0,{i},73729,{len(video)},{len(data)},1280,720,0,0,x')
        video += data
    irows = []
    for k in range(int(3.4 * 208)):
        t = t0 - 0.2 + k / 208
        if imu_stop_s is not None and t > t0 + imu_stop_s:
            irows.append(f'{int(t * 1e9)},1,0,0,571,-0.0001,-0.0001,-0.0001,0')   # disconnect artefact
            break
        irows.append(f'{int(t * 1e9)},1,0.1,0.2,0.3,0.0,0.0,1.0,0')
    fcsv = ('camera_timestamp_ns,dequeue_monotonic_ns,receipt_monotonic_ns,sequence,flags,offset,size,width,height,'
            'camera_on_imu_host_clock_ns,camera_to_imu_offset_ns,time_calibration_id\n' + '\n'.join(frows) + '\n')
    icsv = 'read_monotonic_ns,read_duration_ns,gx_dps,gy_dps,gz_dps,ax_g,ay_g,az_g,clipped\n' + '\n'.join(irows) + '\n'
    if truncate_csv:
        fcsv = fcsv[:-25]; icsv = icsv[:-20]
    d = root / eid
    d.mkdir(parents=True)
    with tarfile.open(d / 'recording.tar', 'w') as tar:
        for name, data in (('metadata.json', json.dumps({'started_monotonic_s': t0}).encode()),
                           ('frames.csv', fcsv.encode()), ('imu.csv', icsv.encode()), ('video.mjpg', video)):
            ti = tarfile.TarInfo(name); ti.size = len(data); tar.addfile(ti, io.BytesIO(data))


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.old = recording.DATASET; recording.DATASET = self.root

    def tearDown(self):
        recording.DATASET = self.old; self.tmp.cleanup()

    def test_undecodable_frame_dropped_gap_kept(self):
        make_tar(self.root, 'e1', corrupt_frame=20)
        r = recording.load('e1')
        self.assertEqual(r.quality['frames_undecodable'], 1)
        self.assertFalse(r.keep[20])
        kept = r.cam_t[r.keep]
        self.assertAlmostEqual(float(np.max(np.diff(kept))), 0.10, places=3)   # gap preserved

    def test_imu_disconnect_truncates(self):
        make_tar(self.root, 'e2', imu_stop_s=1.5)
        r = recording.load('e2')
        self.assertTrue(r.quality['imu_ended_early'])
        self.assertEqual(r.quality['imu_invalid_samples_removed'], 1)
        self.assertLess(r.cam_t[r.keep][-1], 100 + 1.5)
        self.assertLess(np.max(np.linalg.norm(r.gyro, axis=1)), np.deg2rad(500))

    def test_truncated_csv(self):
        make_tar(self.root, 'e3', truncate_csv=True)
        r = recording.load('e3')
        self.assertTrue(r.quality['frames_csv_truncated'])
        self.assertTrue(r.quality['imu_csv_truncated'])
        self.assertGreater(r.keep.sum(), 20)


if __name__ == '__main__':
    unittest.main()
