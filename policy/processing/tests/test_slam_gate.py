"""Tests for the SLAM-smoother plausibility gate and restart selection (no dataset needed)."""
import unittest
from unittest import mock

import numpy as np

from policy.processing.slam import pipeline  # noqa: E402
from policy.processing.slam.pipeline import Gate, plausibility  # noqa: E402


class PlausibilityTest(unittest.TestCase):
    def test_tabletop_motion_passes(self):
        t = np.linspace(0, 8, 160)
        c = np.column_stack([0.1 * np.sin(t), 0.05 * np.cos(t), 0.1 * t / 8])
        v = np.gradient(c, t, axis=0)
        ok, reasons, m = plausibility(c, v, 0.45)
        self.assertTrue(ok, reasons)
        self.assertLess(m['max_speed_mps'], 0.5)

    def test_each_limit_fires(self):
        c = np.zeros((10, 3)); v = np.zeros((10, 3))
        self.assertIn('max speed', plausibility(c, v + [0.6, 0, 0], 0.4)[1][0])
        c2 = c.copy(); c2[-1, 2] = 0.8
        self.assertIn('extent', plausibility(c2, v, 0.4)[1][0])
        self.assertIn('IMU whitened RMS', plausibility(c, v, 3.0)[1][0])
        c3 = c.copy(); c3[0, 0] = np.nan
        self.assertFalse(plausibility(c3, v, 0.4)[0])
        self.assertTrue(plausibility(c2, v, 0.4, Gate(max_extent_m=1.0))[0])


class FakeSmoother:
    """Stands in for fusion.Smoother: the standard init diverges (lower cost), restarts are plausible."""
    def __init__(self, P):
        self.speed_prior_sd = None; self.init_mode = 'dead_reckoning'; self.x = None

    def warm_start(self, other):
        return self

    def solve(self, verbose=0):
        big = self.speed_prior_sd is None
        self.cost = 100.0 if big else 120.0 + (self.speed_prior_sd or 0)
        self.v = np.full((5, 3), 1.2 if big else 0.1)
        self._c = np.linspace(0, 3.0 if big else 0.2, 15).reshape(5, 3)
        return self

    def camera_poses(self):
        return np.tile(np.eye(3), (5, 1, 1)), self._c

    def residuals(self, x, parts=False):
        return {'imu': np.full(9, 0.4)}


class GateSelectionTest(unittest.TestCase):
    def test_restart_replaces_implausible_even_if_cheaper(self):
        with mock.patch.object(pipeline, 'Smoother', FakeSmoother):
            sm = pipeline._solve_with_gate(P=None)
        self.assertTrue(sm.gate['fired'])
        self.assertTrue(sm.gate['passed'])
        self.assertNotEqual(sm.gate['selected_init'], 'standard')
        self.assertFalse(sm.gate['attempts'][0]['plausible'])
        self.assertLess(np.max(np.linalg.norm(sm.v, axis=1)), 0.5)

    def test_plausible_standard_does_not_fire(self):
        class Good(FakeSmoother):
            def solve(self, verbose=0):
                self.speed_prior_sd = 0.3
                return super().solve()
        with mock.patch.object(pipeline, 'Smoother', Good):
            sm = pipeline._solve_with_gate(P=None)
        self.assertFalse(sm.gate['fired'])
        self.assertEqual(len(sm.gate['attempts']), 1)


if __name__ == '__main__':
    unittest.main()
