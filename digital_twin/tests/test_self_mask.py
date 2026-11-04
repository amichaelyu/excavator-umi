"""Self-mask (digital_twin.sim.self_mask): lens inversion, mask geometry, fill, mount updates.

    .venv/bin/python -m unittest hub.tests.test_self_mask -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from policy.data import camera
from digital_twin.sim.self_mask import SelfMasker, camera_lens, distort_normalized, load_config, undistort_normalized


class LensTest(unittest.TestCase):
    def test_undistort_inverts_distort(self):
        cfg = load_config()['intrinsics']
        K, D = np.asarray(cfg['matrix']), cfg['distortion']
        x, y = np.meshgrid(np.linspace(-0.9, 0.9, 7), np.linspace(-0.5, 0.5, 5))
        for model in ('radtan', 'equidistant'):
            d = D if model == 'radtan' else D[:4]
            xd, yd = distort_normalized(x, y, d, model)
            xu, yu = undistort_normalized(K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2], K, d, model)
            np.testing.assert_allclose(xu, x, atol=1e-4)
            np.testing.assert_allclose(yu, y, atol=1e-4)


    def test_lens_comes_from_the_camera_config_and_stays_there(self):
        """configs/sim/self_mask.json has no lens of its own: it is configs/camera.json's, and save() keeps it out."""
        c = camera.intrinsics()
        cfg = load_config()
        self.assertEqual(cfg['intrinsics'], camera_lens())
        self.assertEqual((cfg['intrinsics']['model'], cfg['intrinsics']['matrix']), (c['distortion_model'], c['matrix']))
        m = SelfMasker()
        own = SelfMasker(config={**cfg, 'intrinsics': {**cfg['intrinsics'], 'source': 'measured here'}})
        try:
            self.assertEqual((m.lens, m.size0), ('equidistant', tuple(c['image_size'])))
            with tempfile.TemporaryDirectory() as d:
                self.assertNotIn('intrinsics', json.loads(m.save(Path(d) / 'a.json').read_text()))
                self.assertEqual(json.loads(own.save(Path(d) / 'b.json').read_text())['intrinsics']['source'], 'measured here')
        finally:
            m.close(); own.close()


class MaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = SelfMasker()
        cls.q = {'boom': 0.5, 'bucket': -0.7}           # homed pose: stick and cab in view of the default mount

    @classmethod
    def tearDownClass(cls):
        cls.m.close()

    def test_mask_covers_machine_and_moves_with_joints(self):
        a = self.m.mask(self.q, (360, 640, 3))
        b = self.m.mask({'boom': 0.0, 'bucket': 0.5}, (360, 640, 3))
        self.assertEqual(a.shape, (360, 640))
        self.assertGreater(a.mean(), 0.05)
        self.assertLess(a.mean(), 0.95)
        self.assertGreater((a ^ b).mean(), 0.05)

    def test_apply_fills_only_masked_pixels(self):
        rgb = np.full((720, 1280, 3), 200, np.uint8)
        out, mask = self.m.apply(rgb, self.q)
        self.assertTrue((out[mask] == 0).all())
        self.assertTrue((out[~mask] == 200).all())
        self.assertTrue((rgb == 200).all())             # input untouched

    def test_unknown_joints_are_ignored_and_omitted_ones_rest(self):
        a = self.m.mask(self.q, (180, 320, 3))
        b = self.m.mask({**self.q, 'nonsense': 3.0}, (180, 320, 3))
        np.testing.assert_array_equal(a, b)
        # slew / stick default to their rest values (where the hardware leaves them) ...
        c = self.m.mask({**self.q, **self.m._rest}, (180, 320, 3))
        np.testing.assert_array_equal(a, c)
        # ... and are joints of the twin: turning the base moves the machine in this bucket camera
        self.assertIn('slew', self.m.joints())
        d = self.m.mask({**self.q, 'stick': self.m._rest['stick'] - 0.6}, (180, 320, 3))
        self.assertGreater((a ^ d).mean(), 0.01)

    def test_mount_change_moves_mask(self):
        m = SelfMasker()
        try:
            a = m.mask(self.q, (180, 320, 3))
            m.set_mount(rpy_deg=[0, 0, 30])
            b = m.mask(self.q, (180, 320, 3))
            self.assertGreater((a ^ b).mean(), 0.02)
        finally:
            m.close()


if __name__ == '__main__':
    unittest.main()
