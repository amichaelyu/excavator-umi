"""Link IMUs (boom, stick, bucket): estimator math, the hardware calibration (digital_twin.hardware.calibrate:
ranges, speeds, dead times, base turn, pin signs, the map merge, the CLI run), and the policy runner's measured
state, all on calibrate.SimArm (random IMU mounts, base tilt, accel noise, motor onset; no MuJoCo).

    .venv/bin/python -m unittest digital_twin.tests.test_arm_imus -v
"""
import json
import math
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from digital_twin.hardware import calibrate as cai
from digital_twin.paths import HARDWARE_MAP
from digital_twin.policy import runner as pr
from digital_twin.hardware import state_estimator as se
from digital_twin.tests.test_policy_runner import FakePolicy, make_args


def cal_args(**kw):
    return SimpleNamespace(**{'mode': 'sim', 'visits': 2, 'force': False, **kw})


def calibrate(seed=0, **kw):
    """A calibration on an arm whose ends are exact (no settling slop), for the estimator tests."""
    hw = pr.load_hw()
    arm = cai.SimArm(hw, seed=seed, **({'settle_deg': 0.0} | kw))
    return hw, arm, cai.run_calibration(arm, hw, cal_args(), log=lambda *_: None)


class EstimatorMathTests(unittest.TestCase):
    def test_link_rotation_sign_and_axis_fit(self):
        rng = np.random.default_rng(3)
        axis = se.unit(rng.normal(size=3))
        u0 = se.unit(np.cross(axis, rng.normal(size=3)) + 0.2 * axis)
        from scipy.spatial.transform import Rotation
        for theta in (0.3, -1.1, 2.0):
            u1 = Rotation.from_rotvec(-theta * axis).apply(u0)      # link turns +theta -> gravity turns -theta
            self.assertAlmostEqual(se.link_rotation(u0, u1, axis), theta, places=9)
        # one arc (the link turning +0.8 rad in steps) defines the plane; a gravity reading far from
        # normal to the axis (u0 has a large axis component) must not bias it
        arc = [Rotation.from_rotvec(-t * axis).apply(u0) for t in np.linspace(0, 0.8, 5)]
        a, resid, cond = se.axis_from_arcs([{'points': arc, 'sign': +1}])
        np.testing.assert_allclose(a, axis, atol=1e-9)
        self.assertLess(resid, 1e-9)
        self.assertLess(cond, 1e-4)
        a, _, _ = se.axis_from_arcs([{'points': arc[::-1], 'sign': -1}])      # same arc walked back: same axis
        np.testing.assert_allclose(a, axis, atol=1e-9)
        _, _, cond = se.axis_from_arcs([{'points': [arc[0], arc[-1], arc[0]], 'sign': +1}])
        self.assertGreater(cond, 0.3)                                        # two points: no plane

    def test_pitches_and_joints_round_trip(self):
        q = {'boom': 0.2, 'stick': 0.9, 'bucket': -0.4}
        back = se.joints_from_pitches(se.link_pitches(q))
        for j in q:
            self.assertAlmostEqual(back[j], q[j], places=12)
        self.assertEqual(set(se.joints_from_pitches({'boom': 0.1, 'bucket': 0.5})), {'boom'})   # no stick IMU


class CalibrationTests(unittest.TestCase):
    # the simulated ends settle up to 2 deg short of the true end (no hard stops): the reference pose carries
    # up to that much error per joint, a range up to twice that
    TOL = TOL_RANGE = math.radians(4.5)                            # a joint = difference of two link pitches

    def test_recovers_ranges_speeds_and_measures_any_pose(self):
        hw, arm, out = calibrate(seed=1, settle_deg=2.0)
        self.assertEqual(out['status'], 'done', out.get('calibration', {}).get('failures'))
        cal = out['calibration']
        self.assertEqual(cal['failures'], [])
        for j in se.ARM_LINKS:
            self.assertAlmostEqual(cal['ranges_rad'][j], np.ptp(arm.ends[j]), delta=self.TOL_RANGE)
            for k, i in (('positive', 0), ('negative', 1)):
                self.assertAlmostEqual(cal['speeds_rad_s'][j][k], arm.speeds[j][i], delta=0.05 * arm.speeds[j][i])
        est = se.ArmIMUEstimator(se.DeadReckoningEstimator(), cal)
        rng = np.random.default_rng(5)
        for _ in range(20):
            arm.q = {j: float(rng.uniform(*arm.ends[j])) for j in se.ARM_LINKS}
            m = cai.measure_joints(arm, est)
            for j in se.ARM_LINKS:
                self.assertAlmostEqual(m[j], arm.q[j], delta=self.TOL)

    def test_dead_pin_is_reported_and_the_joint_still_estimates(self):
        hw = pr.load_hw()
        arm = cai.SimArm(hw, seed=3, dead={('boom', -1)}, settle_deg=2.0)
        out = cai.run_calibration(arm, hw, cal_args(), log=lambda *_: None)
        cal = out['calibration']
        self.assertEqual(out['status'], 'done')                      # a dead pin is reported, not fatal
        pin = cai.joint_pins(hw, 'boom')['negative']['pin']
        self.assertTrue(any(f.startswith(f'GPIO {pin} ') and 'did not move the boom' in f for f in cal['failures']),
                        cal['failures'])
        self.assertNotIn('boom', cal['ranges_rad'])
        self.assertEqual(sum(1 for p in arm.pulses if p[:2] == ('boom', -1)), 9)   # 3 tries (lost-command retries) x 3 chunks
        est = se.ArmIMUEstimator(se.DeadReckoningEstimator(), cal)
        arm.q = {'boom': arm.ends['boom'][1] - 0.3, 'stick': 0.2, 'bucket': 1.0}
        m = cai.measure_joints(arm, est)
        for j in se.ARM_LINKS:
            self.assertAlmostEqual(m[j], arm.q[j], delta=self.TOL)

    def test_swapped_boom_and_stick_imus_fail_the_checks(self):
        hw = pr.load_hw()
        arm = cai.SimArm(hw, seed=2, swap=('boom', 'stick'))
        out = cai.run_calibration(arm, hw, cal_args(), log=lambda *_: None)
        self.assertEqual(out['status'], 'failed checks')
        self.assertTrue(any('not on the links' in f for f in out['calibration']['failures']), out['calibration']['failures'])

    def test_missing_imu_stops_before_any_pulse(self):
        hw = pr.load_hw()
        arm = cai.SimArm(hw)
        samples = arm.samples
        arm.samples = lambda n, t0, t1: samples(n, t0, t1) if n != 'boom' else samples(n, -2, -1)
        with self.assertRaisesRegex(cai.CalibrationError, 'boom IMU is not live'):
            cai.run_calibration(arm, hw, cal_args(), log=lambda *_: None)
        self.assertEqual(arm.pulses, [])

    def test_merge_writes_ranges_for_the_real_joint_config(self):
        hw, arm, out = calibrate(seed=4)
        merged = cai.merge_calibration(json.loads(json.dumps(hw)), out['calibration'], 'test')
        from digital_twin.sim.joint_config import real_joint_config
        real, _ = real_joint_config(merged, warn=lambda m: None)
        for j in se.ARM_LINKS:
            self.assertAlmostEqual(real[j]['hi'] - real[j]['lo'], np.ptp(arm.ends[j]), delta=self.TOL_RANGE)
        self.assertEqual(merged['calibration']['arm_imus']['session'], 'test')
        self.assertNotIn('slew', merged['calibration']['arm_imus'])
        se.ArmIMUEstimator.from_hardware_map(se.DeadReckoningEstimator(), merged)

    def test_dead_times_are_press_to_motion_and_reach_the_runner(self):
        hw, arm, out = calibrate(seed=5, onset_s=0.25)
        dt = out['calibration']['dead_time_s']
        for j in se.ARM_LINKS:
            for k in ('positive', 'negative'):
                self.assertAlmostEqual(dt[j][k], 0.25, delta=0.03)            # + one 10 ms sample
        h = json.loads(json.dumps(hw))
        h['calibration']['speeds']['boom']['dead_time_s'] = {'negative': -0.4, 'extra': 0.3}   # an old fit's values
        merged = cai.merge_calibration(h, out['calibration'], 'test')
        self.assertAlmostEqual(merged['calibration']['speeds']['boom']['dead_time_s']['negative'], dt['boom']['negative'])
        self.assertEqual(merged['calibration']['speeds']['boom']['dead_time_s']['extra'], 0.3)   # unmeasured keys kept
        onsets, src = pr.pulse_onsets(merged, ('boom', 'bucket'))
        self.assertIn('boom calibration', src)
        self.assertAlmostEqual(onsets['bucket'], max(dt['bucket'].values()))


class BaseTurnTests(unittest.TestCase):
    def test_turn_rates_onsets_and_sign_into_the_slew_entry(self):
        hw, arm, out = calibrate(seed=8, onset_s=0.2)
        s = out['calibration']['slew']
        self.assertEqual((s['sign'], s['sign_verified']), (1, True))
        for w, i, d in (('left', 0, 1), ('right', 1, -1)):
            self.assertAlmostEqual(s['slew'][w]['yaw_rate_deg_s'], d * math.degrees(arm.slew_speeds[i]), delta=1.5)
            self.assertAlmostEqual(s['slew'][w]['onset_s'], 0.2, delta=0.03)
            self.assertEqual(s['slew'][w]['n_pulses'], 4)
        self.assertEqual(sum(1 for p in arm.pulses if p[0] == 'slew'), 8)        # 2 lengths x (out + back) x 2 sides
        h = json.loads(json.dumps(hw))
        h['slew']['sign_verified'] = False                                     # a failed earlier check
        merged = cai.merge_calibration(h, out['calibration'], 'test')
        sl = merged['slew']
        self.assertNotIn('sign_verified', sl)
        self.assertNotIn('pin_positive', sl)                                   # the pins live in joints.slew only
        h['slew'].update(pin_positive=19, pin_negative=21)                     # an older map's duplicate pin copy
        self.assertFalse({'pin_positive', 'pin_negative'} & set(cai.merge_calibration(h, out['calibration'], 'test')['slew']))
        drives = merged['calibration']['arm_imus']['checks']['joints']['boom']['drives']
        self.assertEqual(set(drives), {'positive', 'negative'})                # checks per direction, no GPIO keys
        self.assertFalse({'direction', 'pin'} & set(drives['positive']))
        self.assertAlmostEqual(sl['yaw_deg_s']['left'], s['slew']['left']['yaw_rate_deg_s'])
        self.assertAlmostEqual(sl['onset_s']['right'], s['slew']['right']['onset_s'])
        self.assertEqual(merged['calibration']['slew']['session'], 'test')
        cfg, src = pr.slew_config(merged)                                       # the runner's slew speeds
        self.assertAlmostEqual(cfg['v_pos'], math.radians(abs(sl['yaw_deg_s']['left'])))
        self.assertTrue(pr.sign_verified(merged, 'slew'))

    def test_reversed_or_dead_base_pins_are_not_verified(self):
        hw = pr.load_hw()
        out = cai.run_calibration(cai.SimArm(hw, seed=2, flip={'slew'}), hw, cal_args(), log=lambda *_: None)
        s = out['calibration']['slew']
        self.assertEqual((s['sign'], s['sign_verified']), (-1, False))
        merged = cai.merge_calibration(json.loads(json.dumps(hw)), out['calibration'], 'test')
        self.assertFalse(merged['slew']['sign_verified'])
        self.assertLess(merged['slew']['yaw_deg_s']['left'], 0)
        out = cai.run_calibration(cai.SimArm(hw, seed=2, dead={('slew', 1), ('slew', -1)}), hw, cal_args(),
                                  log=lambda *_: None)
        self.assertEqual(out['calibration']['slew']['sign'], 0)
        h = json.loads(json.dumps(hw))
        merged = cai.merge_calibration(h, out['calibration'], 'test')
        self.assertFalse(merged['slew']['sign_verified'])
        self.assertEqual(merged['slew']['yaw_deg_s'], hw['slew']['yaw_deg_s'])      # no turn: the old rates stay
        out = cai.run_calibration(cai.SimArm(hw, seed=2), hw, cal_args(slew=False), log=lambda *_: None)
        self.assertIsNone(out['calibration']['slew'])


class PinSignTests(unittest.TestCase):
    def test_a_reversed_joint_is_found_by_its_neighbours(self):
        hw = pr.load_hw()
        for flipped in ('bucket', 'stick', 'boom'):
            with self.subTest(flipped=flipped):
                out = cai.run_calibration(cai.SimArm(hw, seed=1, flip={flipped}), hw, cal_args(), log=lambda *_: None)
                self.assertEqual(out['status'], 'failed checks')
                signs = out['calibration']['signs']
                self.assertEqual({j: s['sign'] for j, s in signs.items()},
                                 {j: -1 if j == flipped else 1 for j in se.ARM_LINKS})
                h = json.loads(json.dumps(hw))
                for j in se.ARM_LINKS:
                    h['joints'][j]['sign_verified'] = False
                merged = cai.merge_calibration(h, out['calibration'], 'test', signs_only=True)
                self.assertEqual(merged['calibration'], hw['calibration'])          # only the signs
                for j in se.ARM_LINKS:
                    self.assertEqual(pr.sign_verified(merged, j), j != flipped)

    def test_a_dead_pin_leaves_the_sign_unverified(self):
        hw = pr.load_hw()
        out = cai.run_calibration(cai.SimArm(hw, seed=3, dead={('bucket', 1)}), hw, cal_args(), log=lambda *_: None)
        signs = out['calibration']['signs']
        self.assertEqual((signs['bucket']['sign'], signs['bucket']['verified']), (None, False))
        h = json.loads(json.dumps(hw))
        h['joints']['bucket']['sign_verified'] = False
        merged = cai.merge_calibration(h, out['calibration'], 'test')
        self.assertFalse(merged['joints']['bucket']['sign_verified'])             # left as it was
        self.assertTrue(pr.sign_verified(merged, 'boom'))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='calibrate-'))
        self.hw = self.tmp / 'hardware_map.json'
        shutil.copy(HARDWARE_MAP, self.hw)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, *extra):
        return cai.parse_args(['--hardware-map', str(self.hw), '--artifacts', str(self.tmp / 'runs'), *extra])

    def test_sim_run_writes_the_artifact_and_never_the_map(self):
        before = self.hw.read_text()
        lines = []
        self.assertEqual(cai.run(self.args('--seed', '3'), log=lines.append), 0)
        (d,) = (self.tmp / 'runs').iterdir()
        self.assertTrue(d.name.endswith('-sim'))
        r = json.loads((d / 'calibration.json').read_text())
        self.assertEqual(r['status'], 'done')
        self.assertIn('slew', r['calibration'])
        self.assertIn('5/6 fit', (d / 'log.txt').read_text())
        self.assertTrue(any('left untouched' in line for line in lines))
        self.assertEqual(self.hw.read_text(), before)

    def test_stop_interrupts_before_any_press_and_live_needs_the_flag(self):
        stop = threading.Event(); stop.set()
        self.assertEqual(cai.run(self.args(), log=lambda *_: None, stop=stop), 1)
        (d,) = (self.tmp / 'runs').iterdir()
        r = json.loads((d / 'calibration.json').read_text())
        self.assertEqual((r['status'], r['sim_truth']['pulses']), ('interrupted', 0))
        lines = []
        self.assertEqual(cai.run(self.args('--live'), log=lines.append), 2)
        self.assertIn('--i-understand-open-loop', lines[0])

    def test_write_validates_and_merges_into_the_file(self):
        hw = json.loads(self.hw.read_text())
        _, _, out = calibrate(seed=4)
        cai.write_hardware_map(self.hw, out['calibration'], '20261104T000000Z')
        new = json.loads(self.hw.read_text())
        self.assertEqual(new['calibration']['arm_imus']['session'], '20261104T000000Z')
        self.assertEqual(new['coupling'], hw['coupling'])                          # not measured here: kept
        self.assertEqual(new['true_pose'], hw['true_pose'])


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.hw, self.arm, out = calibrate(seed=6, gyro_bias_dps=0.0)        # the tests feed bias-free gyro
        self.cal = out['calibration']
        self.arm.slew = 0.0                       # the base at its reference heading (the base turn ends a few deg off)

    def test_tilt_rereference_shifts_only_the_boom(self):
        est = se.ArmIMUEstimator(se.DeadReckoningEstimator(), self.cal)
        from scipy.spatial.transform import Rotation
        self.arm.tilt = Rotation.from_rotvec([0.0, -math.radians(4), 0.0]).as_matrix() @ self.arm.tilt   # base re-parked
        home = pr.home_values(self.hw)
        self.arm.q = dict(home)
        w = cai.window(self.arm, 0.5)
        est.set_static({n: w[n]['accel'] for n in se.ARM_LINKS}, self.arm.now())
        before = est.measured(self.arm.now())
        self.assertAlmostEqual(abs(before['boom'] - home['boom']), math.radians(4), delta=math.radians(0.6))
        est.rereference_boom(home['boom'], self.arm.now())
        after = est.measured(self.arm.now())
        for j in se.ARM_LINKS:
            self.assertAlmostEqual(after[j], home[j], delta=math.radians(0.5))
        self.assertAlmostEqual(after['stick'], before['stick'], places=9)

    def test_gyro_carries_the_pitch_through_dynamic_accel_and_stale_falls_back(self):
        base = se.DeadReckoningEstimator(q0={'boom': 0.1, 'stick': 1.0, 'bucket': 0.3})
        est = se.ArmIMUEstimator(base, self.cal)
        self.arm.q = {'boom': 0.1, 'stick': 1.0, 'bucket': 0.3}
        est.update({n: (self.arm.up_imu(n), np.zeros(3), 0.0) for n in se.ARM_LINKS})
        # boom turning at 0.5 rad/s for 0.2 s, readings 1.5 g (dynamic): only the gyro may move it
        for k in range(1, 11):
            self.arm.q['boom'] = 0.1 + 0.5 * 0.02 * k
            r = {n: (1.5 * self.arm.up_imu(n), np.degrees(0.5 * np.asarray(self.cal['imus'][n]['axis_imu'])), 0.02 * k)
                 for n in se.ARM_LINKS}
            est.update(r)
        m = est.measured(0.2)
        self.assertAlmostEqual(m['boom'], 0.2, delta=1e-3)   # gyro only (calibration error ~1e-4)
        self.assertAlmostEqual(m['stick'], 1.0, delta=math.radians(0.5))
        self.assertEqual(est.measured(5.0), {})                         # stale -> nothing measured
        self.assertAlmostEqual(est.get('boom', 5.0), est.base.get('boom', 5.0))


# ------------------------------------------------------------------ runner on the simulated arm
class SimArmExecutor:
    """Runner executor on a SimArm plant (virtual clock): pulses move the plant; the IMUs read it."""
    virtual_clock = True
    infer_latency_s = 0.05

    def __init__(self, plant, hw, drop_bucket_imu_after=None):
        self.plant, self.hw = plant, hw
        self.sent = []
        self.drop_after = drop_bucket_imu_after

    @property
    def time(self):
        return self.plant.t

    def execute(self, pulses, gap_ms=0.0, phase='policy', heartbeat=None):
        pr.check_pulse_pins(pulses, self.hw, phase, se.FULL4_JOINTS)
        t0 = self.plant.t
        for p in pulses:
            self.plant.t = max(self.plant.t, t0 + p['t_start_s'])
            self.plant.pulse(p['joint'], p['direction'], p['duration_ms'])
            self.plant.t += gap_ms / 1000.0
            self.sent.append({**p, 'phase': phase})
        return {'t_start': t0, 't_end': self.plant.t, 'sent': len(pulses)}

    def advance(self, dt):
        self.plant.t += max(0.0, dt)

    def release_all(self):
        pass

    def emergency_release(self):
        pass

    # sensors / camera
    def imus(self):
        r = self.plant.read()
        if self.drop_after is not None and self.plant.t > self.drop_after:
            r.pop('bucket')
        return r, {}

    def imu(self):
        r = self.plant.read()['bucket']
        return r, {}

    def grab(self, after=None):
        return np.zeros((360, 640, 3), np.uint8), self.plant.t, {}


class RunnerMeasuredStateTests(unittest.TestCase):
    def _runner(self, argv, drop=None, calibrated=True):
        hw, arm, out = calibrate(seed=7)
        hw = json.loads(json.dumps(hw))
        if calibrated:
            hw = cai.merge_calibration(hw, out['calibration'], 'test')
        else:                                          # the checked-in map may carry a live calibration
            (hw.get('calibration') or {}).pop('arm_imus', None)
        plant = cai.SimArm(hw, seed=7, settle_deg=0.0, run_on_s=0.0)  # same mounts and ends, a fresh pose
        plant.q = {'boom': plant.stops['boom'][0] + 0.1, 'stick': plant.stops['stick'][1], 'bucket': 1.2}
        ex = SimArmExecutor(plant, hw, drop)
        FakePolicy.instances.clear()
        args = make_args(['--gateway-url', 'wss://g', '--live', '--i-understand-open-loop', '--max-seconds', '6',
                          '--max-steps', '6', '--chunk', '3', '--no-home-stick', *argv])
        logs = []
        runner = pr.PolicyRunner(args, policy=FakePolicy(chunk=3), executor=ex, camera=ex, sensors=ex, hw=hw,
                                 log=logs.append)
        return runner, plant, logs

    def test_homing_and_start_pose_are_measured(self):
        runner, plant, logs = self._runner(['--imu', 'arm'])
        runner.run()
        self.assertEqual(runner.summary['estimate'], 'arm IMUs (measured)')
        self.assertIn('arm_imus_after_homing', runner.summary['homing'])
        align = runner.summary['start_pose_alignment']
        for j in se.ARM_LINKS:
            err = abs(align['joints'][j]['final_deg'] - align['target_real_deg'][se.FULL4_JOINTS.index(j)])
            self.assertLessEqual(err, max(1.5, 0.6 * math.degrees(max(plant.speeds[j])) * 0.06) + 0.3)
        self.assertEqual(len(FakePolicy.instances[-1].requests), 6)
        m = runner.arm_settle(0.3)
        for j in se.ARM_LINKS:
            self.assertAlmostEqual(m[j], plant.q[j], delta=math.radians(0.6))

    def test_each_state_matches_the_plant_at_its_observation(self):
        runner, plant, logs = self._runner(['--imu', 'arm'])
        seen = []
        orig = runner.step

        def step(k):
            # the plant pose at the start of the cycle is what the IMUs see before inference
            seen.append(np.array([plant.q[j] for j in se.ARM_LINKS]))
            return orig(k)
        runner.step = step
        runner.run()
        states = [r.state for r in FakePolicy.instances[-1].requests]
        self.assertEqual(len(states), 6)
        for s, truth in zip(states, seen):                            # real = policy joint space; slew mocked
            np.testing.assert_allclose(np.asarray(s, float)[1:], truth, atol=math.radians(1.0))   # gyro-fitted axes

    def test_auto_without_calibration_falls_back_to_dead_reckoning(self):
        runner, plant, logs = self._runner(['--imu', 'auto', '--no-quick-calibration'], calibrated=False)
        runner.setup()
        self.assertIsNone(runner.arm_est)
        self.assertIsInstance(runner.estimator, se.DeadReckoningEstimator)
        self.assertTrue(any('dead-reckoned' in line for line in logs))
        self.assertIsNone(runner.read_imu())
        with self.assertRaisesRegex(pr.RunnerError, 'no calibration.arm_imus'):
            r2, _, _ = self._runner(['--imu', 'arm'], calibrated=False)
            r2.setup()
        runner._start_pose(None, True)                                  # no bucket-IMU fallback: open loop, stays
        self.assertTrue(any('no arm IMUs' in line for line in logs))

    def test_stale_bucket_imu_falls_back_to_dead_reckoning(self):
        runner, plant, logs = self._runner(['--imu', 'arm'])
        runner.setup()
        runner.executor.drop_after = runner.now()                     # the bucket IMU disappears now
        runner.arm_est.pitch.pop('bucket', None)
        runner.read_imu()
        m = runner.arm_est.apply_correction(runner.now())
        self.assertIn('boom', m)
        self.assertNotIn('bucket', m)                                 # dead reckoning keeps the bucket


if __name__ == '__main__':
    unittest.main()
