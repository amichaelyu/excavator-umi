"""Closed-loop policy runner (digital_twin.policy.runner) with a fake policy injected into the runner.

Runs under .venv (MuJoCo, no policy server).

    .venv/bin/python -m unittest digital_twin.tests.test_policy_runner -v
"""
import json
import math
import threading
import time
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from digital_twin.policy import runner as pr
from digital_twin.hardware import excavator, pi, state_estimator as se
from digital_twin.sim.joint_config import real_joint_config

REPO = Path(__file__).resolve().parents[2]

# Every runner run needs live, calibrated arm IMUs (PolicyRunner._setup_arm_imus refuses otherwise). The tests here
# are about pulse planning, homing, latency and the CLOAK contract on the MuJoCo twin, which has no arm IMUs, so they
# skip that gate and the state is dead-reckoned (the runner's per-joint fallback for stale IMUs). The gate and the
# measured state are tested in test_arm_imus.
_SKIP_ARM_IMU_GATE = mock.patch.object(pr.PolicyRunner, '_setup_arm_imus', lambda self: None)


def setUpModule():
    _SKIP_ARM_IMU_GATE.start()


def tearDownModule():
    _SKIP_ARM_IMU_GATE.stop()


# ------------------------------------------------------------------ test-only MuJoCo executor
# The runner's CLI only drives the real machine (Pi camera + bridge). These classes give the tests
# an offline closed loop: a persistent MuJoCo twin with the command model of commands.SimPulseBackend,
# on a virtual clock (runner.now() = executor.time), and its bucket camera as the observation.
class SimCamera:
    def __init__(self, sim):
        self.sim = sim

    def grab(self):
        return self.sim.render(), self.sim.time, {'sim': True}




class SimExecutor:
    virtual_clock = True                               # the runner uses executor.time as its clock
    infer_latency_s = None                             # sim time charged per inference (None: measured)

    """Persistent MuJoCo twin with commands.SimPulseBackend's command model (a pulse moves the
    joint's position target at the calibrated speed; the finite-torque actuator follows; each pulse
    starts ``latency_s`` after it is sent, pulses are sequential with ``gap_ms``), but continuous
    across calls so the policy loop is closed. Targets are clipped at the true joint limits (the
    twin's hard stops) so homing into a stop behaves like the hardware, and gravity is compensated
    on the arm joints (non-back-drivable gearmotors; ``gravity_comp``) so the servo does not sag. Only boom and
    bucket have pulse channels of their own; stick pulses (which also yaw the base on the machine) are logged
    and skipped, and slew / stick hold their start values (default: rest). ``q0`` may be the hardware
    joints [boom, bucket] or the twin's four."""

    def __init__(self, arm=None, latency_s: float = 0.2, render_size=(640, 360), camera: str = 'bucket_camera',
                 q0=None, gravity_comp: bool = True, log=print):
        import mujoco

        from digital_twin.sim.kinematics import HARDWARE_JOINTS, JOINTS, Arm, locked_joints
        self.mj, self.JOINTS, self.welded = mujoco, list(JOINTS), locked_joints()
        self.pulsed = list(HARDWARE_JOINTS)
        self.arm = arm or Arm()
        self.m, self.d = self.arm.model, self.arm.data
        self.latency_s, self.camera, self.log = float(latency_s), camera, log
        self.gravity_comp = bool(gravity_comp)
        self.w, self.h = render_size
        self._renderer = None
        self.trace = []                                  # (t, qpos...) at 20 Hz
        self.reset(self.arm.q_rest if q0 is None else q0)

    @property
    def time(self) -> float:
        return float(self.d.time)

    def reset(self, q):
        from digital_twin.sim.kinematics import full_chain_q
        mj, a = self.mj, self.arm
        q = full_chain_q(q) if len(q) == len(self.pulsed) else np.asarray(q, float)
        mj.mj_resetData(self.m, self.d)
        self.d.qpos[a.qadr] = q; self.d.ctrl[a.act] = q
        mj.mj_forward(self.m, self.d)
        self.target = np.asarray(q, float).copy()

    def qpos_by_name(self) -> dict:
        return dict(zip(self.JOINTS, map(float, self.d.qpos[self.arm.qadr])))

    def _step(self, windows):
        h = self.m.opt.timestep
        t0, t1 = self.d.time, self.d.time + h
        for (a, b, j, rate) in windows:
            ov = min(b, t1) - max(a, t0)
            if ov > 0:
                self.target[j] += rate * ov
        self.target = np.clip(self.target, self.arm.lo, self.arm.hi)
        self.d.ctrl[self.arm.act] = self.target
        if self.gravity_comp:
            # the toy's gearmotors do not back-drive: cancel gravity/bias on the arm joints so the
            # finite-stiffness position servo does not sag (~4 mrad on the boom at its upper stop)
            self.d.qfrc_applied[self.arm.dadr] = self.d.qfrc_bias[self.arm.dadr]
        self.mj.mj_step(self.m, self.d)
        k = int(round(self.d.time / h))
        if k % 10 == 0:
            self.trace.append((self.d.time, *self.d.qpos[self.arm.qadr]))

    def advance(self, dt: float, windows=()):
        n = int(round(max(0.0, dt) / self.m.opt.timestep))
        for _ in range(n):
            self._step(windows)

    def execute(self, pulses, gap_ms: float = 0.0, phase: str = 'policy', heartbeat=None) -> dict:
        start = self.time
        windows, prev_end, skipped = [], -math.inf, []
        for p in pulses:
            send = max(start + p['t_start_s'], prev_end + gap_ms / 1000.0)
            a = send + self.latency_s
            b = a + p['duration_ms'] / 1000.0
            prev_end = b
            if p['joint'] not in self.pulsed:
                skipped.append(p['joint'])
                continue
            j = self.JOINTS.index(p['joint'])
            rate = p['direction'] * (self.arm.v_pos[j] if p['direction'] > 0 else self.arm.v_neg[j])
            windows.append((a, b, j, rate))
        if skipped:
            self.log(f'[sim] {phase}: {sorted(set(skipped))} has no pulse channel of its own in the twin, held '
                     f'(at {", ".join(f"{n}={self.welded.get(n, 0):.4f} rad" for n in sorted(set(skipped)))}); pulse skipped')
        end = max(prev_end, start)
        self.advance(end - start, windows)
        return {'t_start': start, 't_end': self.time, 'sent': len(windows), 'skipped': skipped}

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self.mj.Renderer(self.m, height=self.h, width=self.w)
        self._renderer.update_scene(self.d, camera=self.camera)
        return np.ascontiguousarray(self._renderer.render())

    def release_all(self):
        pass


    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None




def sim_executor(q0=None, calibrated=True, **kw):
    from digital_twin.sim.kinematics import Arm
    arm = Arm('full4_calibrated' if calibrated else 'full4')
    if calibrated:           # speeds as the runner's estimator uses them (the model file may lag the calibration)
        cfg, _ = real_joint_config(pr.load_hw(), None, warn=lambda m: None)
        for i, n in ((1, 'boom'), (3, 'bucket')):
            arm.v_pos[i], arm.v_neg[i] = cfg[n]['v_pos'], cfg[n]['v_neg']
    return SimExecutor(arm, q0=q0, log=kw.pop('log', lambda *_: None), **kw)


# ------------------------------------------------------------------ fake policy
class FakePolicy:
    """Deterministic chunks with the CLOAK client's interface (metadata / infer / reset / close): a baseline
    server (no mask) in [slew, boom, stick, bucket] order. Moves boom and bucket toward GOAL by <= STEP rad per
    action and asks to hold slew and stick (the sim twin has no stick channel)."""
    GOAL = np.array([0.0, 0.05, 0.0, 1.8])
    MOVE = np.array([False, True, False, True])
    STEP = 0.04
    instances = []

    def __init__(self, chunk=3, prompt='scoop the dirt', policy_space='calibrated'):
        self.chunk, self.prompt, self.policy_space = int(chunk), prompt, policy_space
        self.requests, self.resets, self.closed = [], 0, False
        FakePolicy.instances.append(self)

    def metadata(self):
        return {'camera_names': ['rgb'], 'action_keys': ['actions'], 'action_horizon': 10, 'action_dim': 4,
                'joints': list(se.FULL4_JOINTS), 'policy_backend': 'cloak', 'policy_space': self.policy_space,
                'mode': 'baseline', 'needs_mask': False}

    def chunk_for(self, state):
        rows, q = [], np.asarray(state, float).copy()
        for _ in range(self.chunk):
            q = q + np.where(self.MOVE, np.clip(self.GOAL - q, -self.STEP, self.STEP), 0.0); rows.append(q.copy())
        return np.array(rows)

    def infer(self, image, state, mask=None):
        state = np.asarray(state, np.float32)
        self.requests.append(SimpleNamespace(image=image, state=state, mask=mask, prompt=self.prompt))
        return self.chunk_for(state), 1e-3, 1.0

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


class FakeCloakClient(FakePolicy):
    """Stands in for digital_twin.policy.cloak.CloakPolicy (same constructor, 10 actions per chunk)."""

    def __init__(self, gateway_url, prompt, **kw):
        super().__init__(10, prompt)
        self.url = gateway_url


def make_args(argv):
    """Runner args; a policy run gets the required --max-seconds bound (60 s) unless the test sets one."""
    if '--home-only' not in argv and '--max-seconds' not in argv:
        argv = [*argv, '--max-seconds', '60']
    return pr.build_parser().parse_args(argv)


class NoIMU:
    """Sensor stand-in for homing runs that read nothing (no Pi): the bucket IMU is never live."""

    def imu(self):
        return None, {}


def fake_policy(chunk=3):
    return FakePolicy(chunk)


class FakePolicyMixin:
    def setUp(self):
        FakePolicy.instances.clear()


# ------------------------------------------------------------------ estimator
class EstimatorTests(unittest.TestCase):
    def test_dead_reckoning_integrates_and_clips(self):
        cfg = se.load_joint_config()
        est = se.DeadReckoningEstimator({'boom': 0.0, 'bucket': 0.0, 'stick': 0.0}, cfg)
        est.update_from_pulses([{'t_start_s': 0.0, 'joint': 'boom', 'direction': 1, 'duration_ms': 500},
                                {'t_start_s': 0.6, 'joint': 'bucket', 'direction': -1, 'duration_ms': 100000}], 10.0)
        self.assertAlmostEqual(est.get('boom', 10.25), 0.25 * cfg['boom']['v_pos'])
        self.assertAlmostEqual(est.get('boom', 11.0), 0.5 * cfg['boom']['v_pos'])
        self.assertAlmostEqual(est.get('bucket', 200.0), cfg['bucket']['lo'])      # saturates at the stop
        # external correction mid-pulse: the running pulse continues from the corrected value
        est2 = se.DeadReckoningEstimator({}, cfg)
        est2.update_from_pulses([{'t_start_s': 0.0, 'joint': 'bucket', 'direction': 1, 'duration_ms': 1000}], 0.0)
        est2.set_joint('bucket', 0.3, t=0.5)
        self.assertAlmostEqual(est2.get('bucket', 1.0), 0.3 + 0.5 * cfg['bucket']['v_pos'])


class HomingTests(unittest.TestCase):
    def setUp(self):
        self.hw = pr.load_hw()
        self.cfg = se.load_joint_config()

    def test_homing_config_matches_model_and_formula(self):
        from digital_twin.sim.kinematics import load_model, locked_joints
        m = load_model('full4')
        home = self.hw['homing']['joints']
        for n, stop in (('stick', 'hi'), ('boom', 'hi'), ('bucket', 'lo')):
            lo, hi = m.jnt_range[m.joint(n).id]
            self.assertAlmostEqual(home[n]['home_value_rad'], hi if stop == 'hi' else lo, places=6)
            c = self.cfg[n]; v = c['v_pos'] if home[n]['home_direction'] > 0 else c['v_neg']
            self.assertAlmostEqual(home[n]['home_duration_s'], (c['hi'] - c['lo']) / v * 1.5 + 1.0, places=2)
        self.assertAlmostEqual(locked_joints()['stick'], home['stick']['home_value_rad'])

    def test_homing_pulses_order_pins_and_timing(self):
        pulses = pr.homing_pulses(self.hw, 150, self.cfg)
        self.assertEqual([p['joint'] for p in pulses], ['stick', 'boom', 'bucket'])
        self.assertEqual([p['gpio'] for p in pulses], [23, 33, 26])
        self.assertEqual([p['direction'] for p in pulses], [1, 1, -1])
        for a, b in zip(pulses, pulses[1:]):
            self.assertGreaterEqual(b['t_start_s'], a['t_start_s'] + a['duration_ms'] / 1000 + 0.15 - 1e-9)
        self.assertEqual([p['joint'] for p in pr.homing_pulses(self.hw, 150, self.cfg, ('boom', 'bucket'))],
                         ['boom', 'bucket'])

    def test_sim_homing_reaches_the_stops(self):
        args = make_args(['--home-only'])
        logs = []
        runner = pr.PolicyRunner(args, executor=sim_executor(log=logs.append), sensors=NoIMU(), log=logs.append)
        runner.run()
        rep = runner.summary['homing']
        home = pr.home_values(self.hw)
        for n in ('boom', 'bucket'):
            self.assertAlmostEqual(rep['sim_qpos_rad'][n], home[n], delta=1e-3)
            self.assertAlmostEqual(rep['estimator_rad'][n], home[n], delta=1e-6)
        self.assertAlmostEqual(rep['estimator_rad']['stick'], home['stick'], delta=1e-6)
        self.assertEqual(rep['directions'], {'stick': 1, 'boom': 1, 'bucket': -1})   # the stick is always homed
        self.assertEqual(rep['joints_homed'], ['stick', 'boom', 'bucket'])
        # homing from anywhere: start the twin at the opposite stops
        from digital_twin.sim.kinematics import Arm
        arm = Arm('full4_calibrated')
        sim = SimExecutor(arm, q0=np.array([arm.lo[1], arm.hi[3]]), log=logs.append)
        runner = pr.PolicyRunner(args, executor=sim, sensors=NoIMU(), log=logs.append); runner.run()
        q = sim.qpos_by_name()
        self.assertAlmostEqual(q['boom'], home['boom'], delta=1e-3); self.assertAlmostEqual(q['bucket'], home['bucket'], delta=1e-3)

    def test_bridge_homing_sends_stick_boom_bucket_on_their_up_pins(self):
        fake = FakeBridge()
        self.addCleanup(fake.close)
        logs = []
        ex = pr.BridgeExecutor(self.hw, fake.host, dry_run=False, pace=False, log=logs.append)
        runner = pr.PolicyRunner(make_args(['--home-only']), executor=ex, sensors=NoIMU(), hw=self.hw, log=logs.append)
        runner.run()
        self.assertEqual(runner.summary['homing']['joints_homed'], ['stick', 'boom', 'bucket'])
        trig = [l for l in logs if 'trigger GPIO' in l]
        self.assertTrue(trig and all(l.startswith('[LIVE]') for l in trig))                # always sent: no dry run
        self.assertEqual([m['pin'] for m in fake.moves if m['cmd'] == 'trigger'], [23, 33, 26])
        self.assertIn('release', [m['cmd'] for m in fake.moves])                       # released on exit

    def test_homing_refuses_without_stick_pins(self):
        hw = json.loads(json.dumps(self.hw))
        hw['joints'].pop('stick', None)
        hw.get('locked_joints', {}).pop('stick', None)
        self.assertIsNotNone(pr.stick_blocked(hw))
        sim = sim_executor()
        runner = pr.PolicyRunner(make_args(['--home-only']), executor=sim, sensors=NoIMU(), hw=hw, log=lambda *_: None)
        with self.assertRaisesRegex(pr.RunnerError, 'homing refused: no stick pins'):
            runner.run()
        self.assertNotIn('homing', runner.summary)                                     # no pulse went out


# ------------------------------------------------------------------ safety
class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.hw = pr.load_hw()

    def test_a_policy_run_needs_only_max_seconds(self):
        # no pin-sign gate (the arm IMU gate is test_arm_imus'); a policy run needs its time bound, homing does not
        base = ['--gateway-url', 'wss://g']
        unbounded = pr.build_parser().parse_args(base)
        with self.assertRaisesRegex(pr.RunnerError, 'max-seconds'):
            pr.check_run_allowed(unbounded)
        pr.check_run_allowed(make_args(base + ['--max-seconds', '5']))
        pr.check_run_allowed(make_args(['--home-only']))
        # the removed flags are gone from the CLI: every run moves the machine
        opts = {o for a_ in pr.build_parser()._actions for o in a_.option_strings}
        for gone in ('--allow-unverified-signs', '--home-check', '--home-tol-deg', '--live', '--dry-run',
                     '--i-understand-open-loop', '--no-pace', '--probe-latency', '--imu', '--home-stick',
                     '--no-home-stick'):
            self.assertNotIn(gone, opts)
        # the runner refuses before connecting anything
        ex = mock.Mock(spec=['execute', 'release_all'])
        with self.assertRaisesRegex(pr.RunnerError, 'max-seconds'):
            pr.PolicyRunner(unbounded, policy=fake_policy(), executor=ex, log=lambda *_: None).run()
        ex.execute.assert_not_called()

    def test_pin_rules(self):
        # pins follow the hardware map's pins: boom 33/32, bucket 26/25, stick 23/22, base 19/21
        ok = [{'joint': 'boom', 'gpio': 33}, {'joint': 'boom', 'gpio': 32},
              {'joint': 'bucket', 'gpio': 26}, {'joint': 'bucket', 'gpio': 25}]
        pr.check_pulse_pins(ok, self.hw, 'policy')
        pr.check_pulse_pins(ok, self.hw, 'homing')
        # a pulse's GPIO must belong to its joint; the stick only in homing; the base never
        for bad in ({'joint': 'stick', 'gpio': 23}, {'joint': 'stick', 'gpio': 22}, {'joint': 'boom', 'gpio': 26},
                    {'joint': 'bucket', 'gpio': 32}, {'joint': 'slew', 'gpio': 19}, {'joint': 'slew', 'gpio': 21}):
            with self.assertRaises(pr.RunnerError):
                pr.check_pulse_pins([bad], self.hw, 'policy')
        self.assertIsNone(pr.stick_blocked(self.hw))
        pr.check_pulse_pins([{'joint': 'stick', 'gpio': 23}, {'joint': 'stick', 'gpio': 22}], self.hw, 'homing')
        for bad in ({'joint': 'boom', 'gpio': 23}, {'joint': 'slew', 'gpio': 19}):
            with self.assertRaises(pr.RunnerError):
                pr.check_pulse_pins([bad], self.hw, 'homing')
        self.assertEqual(set(excavator.allowed_pins(self.hw, 'homing')), {33, 32, 26, 25, 23, 22})

    def test_latency_skip_logic(self):
        keep, times, skipped = pr.select_actions(0.0, 0.15, 3, 10)
        self.assertEqual(keep.tolist(), [1, 2]); self.assertEqual(skipped, 1)
        np.testing.assert_allclose(times, [0.2, 0.3])
        keep, _, skipped = pr.select_actions(0.0, 0.05, 3, 10, exec_latency_s=0.2)
        self.assertEqual(keep.tolist(), [2]); self.assertEqual(skipped, 2)
        keep, _, skipped = pr.select_actions(0.0, 0.5, 3, 10)                    # all stale: keep the last
        self.assertEqual(keep.tolist(), [2]); self.assertEqual(skipped, 2)
        keep, _, skipped = pr.select_actions(0.0, 0.0, 3, 10)
        self.assertEqual(keep.tolist(), [0, 1, 2]); self.assertEqual(skipped, 0)
        rel = pr.schedule_times([0.3], 0.5, 10)
        self.assertAlmostEqual(rel[0], 0.1)                                       # stale fallback -> one step

    def test_watchdog_releases_on_stall(self):
        now = [0.0]; released = []
        wd = pr.Watchdog(lambda: released.append(1), stall_s=2.0, clock=lambda: now[0])
        wd._armed = True; wd.beat()
        now[0] = 1.5; self.assertFalse(wd.check())
        now[0] = 2.6; self.assertTrue(wd.check()); self.assertEqual(released, [1])
        self.assertFalse(wd.check()); self.assertEqual(released, [1])            # trips once


# ------------------------------------------------------------------ real joint config / calibrated twin
class JointConfigTests(unittest.TestCase):
    """The policy is trained in the calibrated twin's joint space = real space: the runner drives the machine with
    the measured ranges (anchored at the homing stops) and speeds, no transform."""
    def setUp(self):
        self.hw = pr.load_hw()
        self.cal = self.hw['calibration']

    def test_runner_uses_the_measured_real_config(self):
        runner = pr.PolicyRunner(make_args(['--home-only']), log=lambda *_: None)
        real, _ = real_joint_config(self.hw, None, warn=lambda m: None)
        for j in ('boom', 'stick', 'bucket'):
            self.assertEqual(runner.cfg[j], real[j])
        home = pr.home_values(self.hw)
        rb, rk = (self.cal['ranges'][n]['range_measured_rad'] for n in ('boom', 'bucket'))
        self.assertAlmostEqual(runner.cfg['boom']['hi'], home['boom']); self.assertAlmostEqual(runner.cfg['boom']['lo'], home['boom'] - rb)
        self.assertAlmostEqual(runner.cfg['bucket']['lo'], home['bucket']); self.assertAlmostEqual(runner.cfg['bucket']['hi'], home['bucket'] + rk)
        self.assertEqual(runner.policy_space, 'calibrated')

    def test_measured_speeds_drive_pulse_durations(self):
        from digital_twin.hardware.commands import compile_schedule
        hw = json.loads(json.dumps(self.hw))             # fixed calibration, independent of the live map
        sp = {'boom': {'positive': 0.55, 'negative': 0.84}, 'bucket': {'positive': 1.75, 'negative': 4.03}}
        hw['calibration']['speeds'] = {n: {'speed_measured_rad_s': v} for n, v in sp.items()}
        cfg, _ = real_joint_config(hw, None, warn=lambda m: None)
        self.assertEqual(cfg['boom']['v_pos'], sp['boom']['positive']); self.assertEqual(cfg['bucket']['v_neg'], sp['bucket']['negative'])
        q0 = np.array([0.3, 0.2]); q1 = np.array([0.1, 0.9])
        vp = np.array([cfg[n]['v_pos'] for n in ('boom', 'bucket')]); vn = np.array([cfg[n]['v_neg'] for n in ('boom', 'bucket')])
        sched = compile_schedule([0, 2.0], np.vstack([q0, q1]), v_pos=vp, v_neg=vn, hw_map=self.hw)
        d = {p['joint']: p['duration_ms'] for p in sched['pulses']}
        self.assertEqual(d['boom'], round(0.2 / sp['boom']['negative'] * 1000))
        self.assertEqual(d['bucket'], round(0.7 / sp['bucket']['positive'] * 1000))
        # calibration with trusted directions only (boom down, bucket out): the missing direction
        # falls back to the same joint's measured one, with a warning
        hw['calibration']['speeds'] = {'boom': {'speed_measured_rad_s': {'negative': 0.874}},
                                       'bucket': {'speed_measured_rad_s': {'positive': 1.836}}}
        warns = []
        cfg2, _ = real_joint_config(hw, None, warns.append)
        self.assertEqual((cfg2['boom']['v_pos'], cfg2['boom']['v_neg']), (0.874, 0.874))
        self.assertEqual((cfg2['bucket']['v_pos'], cfg2['bucket']['v_neg']), (1.836, 1.836))
        self.assertEqual(len([w for w in warns if 'using its measured' in w]), 2)

    def test_homing_duration_uses_measured_range_and_slow_speed(self):
        cfg, _ = real_joint_config(self.hw, None, warn=lambda m: None)
        pulses = pr.homing_pulses(self.hw, 150, cfg, compute=True)
        for p in pulses:
            c = cfg[p['joint']]
            self.assertEqual(p['duration_ms'], round(((c['hi'] - c['lo']) / min(c['v_pos'], c['v_neg']) * 1.5 + 1.0) * 1000))
        rb = self.cal['ranges']['boom']['range_measured_rad']
        v = min(self.cal['speeds']['boom']['speed_measured_rad_s'].values())
        self.assertEqual(pulses[1]['duration_ms'], round((rb / v * 1.5 + 1) * 1000))


class CalibratedTwinTests(unittest.TestCase):
    def test_calibrated_model_ranges_speeds_and_replay(self):
        from digital_twin.sim.kinematics import Arm
        from digital_twin.sim.replay import run, synthetic_trajectory
        from digital_twin.sim.build import calibrated_config
        from digital_twin.sim.kinematics import load_config
        hw = pr.load_hw()
        arm = Arm('full4_calibrated')
        ccfg = load_config('full4_calibrated')                               # configs/hardware/robot_full4_calibrated.json
        home = pr.home_values(hw)
        for i, n in ((1, 'boom'), (3, 'bucket')):
            j = ccfg['joints'][n]
            np.testing.assert_allclose(np.degrees([arm.lo[i], arm.hi[i]]), j['range_deg'], atol=1e-4)   # model == config
            self.assertAlmostEqual(arm.v_pos[i], j['speed_positive_rad_s']); self.assertAlmostEqual(arm.v_neg[i], j['speed_negative_rad_s'])
        self.assertAlmostEqual(arm.hi[1], home['boom'], places=6)              # anchored at the homing stops
        self.assertAlmostEqual(arm.lo[3], home['bucket'], places=6)
        self.assertFalse(arm.model.jnt_limited[arm.model.joint('slew').id])      # the base turns freely
        # the config is what build --calibrated derives from the calibration it was built from
        cal = ccfg['joints']
        fresh = calibrated_config(json.loads((REPO / 'configs/hardware/robot.json').read_text()), hw)['joints']
        stale = [n for n in ('boom', 'bucket') if not np.allclose(fresh[n]['range_deg'], cal[n]['range_deg'], atol=1e-4)
                 or not np.isclose(fresh[n]['speed_positive_rad_s'], cal[n]['speed_positive_rad_s'])
                 or not np.isclose(fresh[n]['speed_negative_rad_s'], cal[n]['speed_negative_rad_s'])]
        if stale:                          # calibration updated since the last build: warn, do not fail
            print(f'\nNOTE: full4_calibrated is stale for {stale}; rerun python -m digital_twin.sim.build --calibrated')
        v1 = Arm('full4')                                                     # prior ranges unchanged
        np.testing.assert_allclose(np.degrees([v1.lo[[1, 3]], v1.hi[[1, 3]]]), [[-48, -40], [30, 140]], atol=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'traj.csv'
            synthetic_trajectory(duration=3.0).to_csv(path, index=False)
            m = run(path, Path(tmp) / 'out', profile='full4_calibrated', render=False)
            self.assertEqual(m['profile'], 'full4_calibrated')
            self.assertFalse(m['dynamic']['unstable'])
            self.assertTrue((Path(tmp) / 'out' / 'actions.csv').is_file())


# ------------------------------------------------------------------ closed loop
class ClosedLoopTests(FakePolicyMixin, unittest.TestCase):
    def test_sim_closed_loop(self):
        tmp = tempfile.mkdtemp()
        args = make_args(['--gateway-url', 'wss://g', '--max-steps', '12', '--record', tmp,
                          '--exec-latency-s', '0'])
        logs = []
        policy = fake_policy(3)
        home = pr.home_values(pr.load_hw())                  # a policy run does not home: start the twin homed
        sim = sim_executor(q0=[home['boom'], home['bucket']], log=logs.append); sim.infer_latency_s = 0.05
        runner = pr.PolicyRunner(args, policy=policy, executor=sim, camera=SimCamera(sim), log=logs.append)
        summary = runner.run()
        self.assertEqual(summary['steps'], 12)
        client = FakePolicy.instances[-1]
        self.assertEqual(client.resets, 1); self.assertTrue(client.closed)
        steps = runner.recorder.steps
        for req, rec in zip(client.requests, steps):
            self.assertEqual((req.image.shape, req.image.dtype), ((360, 640, 3), np.uint8))   # native render
            self.assertIsNone(req.mask)                                                       # baseline server
            self.assertEqual(req.prompt, 'scoop the dirt')
            self.assertEqual((req.state.shape, req.state.dtype), ((4,), np.float32))
            np.testing.assert_allclose(req.state, rec['state'], atol=1e-6)
        # start pose 'none': the first state is the standard pose, slew mocked at 0
        np.testing.assert_allclose(steps[0]['state'], [0.0, home['boom'], home['stick'], home['bucket']], atol=1e-3)
        # the twin actually moved toward the fake goal, and dead reckoning tracks its qpos
        self.assertTrue(all(s['estimate'] == 'dead reckoning' for s in steps))
        sim.advance(1.5)
        q_sim = np.array([sim.qpos_by_name()[n] for n in ('boom', 'bucket')])
        q_est = runner.estimator.policy_state(sim.time, ('boom', 'bucket'))
        np.testing.assert_allclose(q_est, q_sim, atol=0.02)
        self.assertGreater(q_est[1], home['bucket'] + 0.1)
        for s in steps:
            self.assertAlmostEqual(s['est_boom'], s['sim_boom'], delta=0.08)
            self.assertAlmostEqual(s['est_bucket'], s['sim_bucket'], delta=0.08)
        # policy pulses: arm pins only (never the slew), one at a time
        pol = [p for p in runner.recorder.pulses if p['phase'] == 'policy']
        self.assertTrue(pol)
        self.assertTrue({p['gpio'] for p in pol} <= {33, 32, 26, 25, 23, 22})
        gap = runner.hw['defaults']['gap_ms'] / 1000
        for step in {p['step'] for p in pol}:
            ps = sorted((p for p in pol if p['step'] == step), key=lambda p: p['t_start_s'])
            for x, y in zip(ps, ps[1:]):
                self.assertGreaterEqual(y['t_start_s'], x['t_start_s'] + x['duration_ms'] / 1000 + gap - 1e-6)
        out = Path(tmp)
        for f in ('steps.csv', 'pulses.csv', 'run.json', 'observations.mp4', 'frames/00000.jpg'):
            self.assertTrue((out / f).is_file(), f)
        self.assertIn('latency_ms', json.loads((out / 'run.json').read_text()))


class FakeBucketIMU:
    """IMU on the sim twin's bucket (no per-link ``imus``): accel = gravity 'up' in the IMU frame (``R_bucket_imu``:
    its mount), optionally tilted by ``tilt_deg`` about IMU x once ``tilt_when()`` is true."""

    def __init__(self, sim, R_bucket_imu=np.eye(3), tilt_deg=0.0, tilt_when=lambda: False):
        self.sim, self.R = sim, np.asarray(R_bucket_imu, float)
        self.tilt, self.when = math.radians(tilt_deg), tilt_when

    def imu(self):
        arm = self.sim.arm
        up_bucket = arm.data.xmat[arm.bucket].reshape(3, 3).T @ np.array([0, 0, 1.0])
        a = self.R.T @ up_bucket
        if self.when():
            c, s_ = math.cos(self.tilt), math.sin(self.tilt)
            a = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]) @ a
        return (a, np.zeros(3), time.monotonic()), {}


class HomingImuLoggedOnlyTests(unittest.TestCase):
    """Homing pulses boom / bucket into their top stops and continues; the IMU is only logged
    (no repeatability re-home, no pitch comparison, never aborts)."""
    HOME = ['--home-only']

    def _runner(self, argv, tilt_deg=0.0, hw=None):
        hw = hw or pr.load_hw()
        sim = sim_executor()
        logs, phases = [], []
        orig = sim.execute
        def counting(pulses, gap_ms=0.0, phase='policy', heartbeat=None):
            phases.append(phase)
            return orig(pulses, gap_ms, phase, heartbeat)
        sim.execute = counting
        imu = FakeBucketIMU(sim, np.eye(3), tilt_deg, lambda: True)
        runner = pr.PolicyRunner(make_args(argv), hw=hw, executor=sim, camera=SimCamera(sim), sensors=imu, log=logs.append)
        return runner, logs, phases

    def test_homing_logs_the_imu_and_continues(self):
        runner, logs, phases = self._runner(self.HOME, tilt_deg=40.0)      # a reading far from calibration
        runner.run()
        rep = runner.summary['homing']
        self.assertEqual(phases, ['homing'])                               # no re-home
        self.assertEqual(rep['joints_homed'], ['stick', 'boom', 'bucket'])
        self.assertIn('imu_after_homing', rep)
        self.assertNotIn('vs_calibration_home_deg', rep)
        self.assertIsNone(runner.arm_est)                                  # arm IMU gate skipped here: nothing corrects it
        self.assertTrue(any('logged only' in l for l in logs))
        home = pr.home_values(runner.hw)
        for n in ('boom', 'bucket'):
            self.assertAlmostEqual(rep['estimator_rad'][n], home[n], delta=1e-6)

    def test_unverified_signs_still_home(self):
        hw = json.loads(json.dumps(pr.load_hw()))
        for n in ('boom', 'bucket'):
            hw['joints'][n]['sign_verified'] = False
        runner, logs, phases = self._runner(self.HOME, hw=hw)
        runner.run()
        self.assertEqual(phases, ['homing'])
        self.assertFalse(any('WARNING' in l and 'calibrat' in l and 'pin direction' not in l for l in logs), logs)
        self.assertEqual(runner.summary['unverified_signs'], ['boom', 'bucket'])      # warned, never blocked


# ------------------------------------------------------------------ coupled channels / pulse planner
class ChannelModelTests(unittest.TestCase):
    def setUp(self):
        from digital_twin.hardware.commands import ChannelModel
        self.hw = json.loads(json.dumps(pr.load_hw()))
        self.hw.pop('slew', None)                       # the old wiring: A / B yaw and move the stick
        self.hw['coupling'] = {'channels': {'positive': {'yaw_deg_s': -43.0, 'stick_deg_s': 25.0},
                                            'negative': {'yaw_deg_s': 62.0, 'stick_deg_s': -28.0}}}
        lim = {'slew': (-np.inf, np.inf), 'boom': (-0.3, 0.52), 'stick': (-0.08, 1.01), 'bucket': (-0.7, 1.69)}
        self.cm = ChannelModel.from_config(self.hw, {n: dict(zip(('lo', 'hi'), v)) for n, v in lim.items()})

    def test_yaw_and_stick_move_together_then_pure_yaw_at_the_stop(self):
        q0 = np.array([0.0, 0.2, 0.5, 0.3])
        q = self.cm.step(q0, 'A', 0.5)
        np.testing.assert_allclose(q - q0, np.radians([-43.0 * 0.5, 0, 25.0 * 0.5, 0]), atol=1e-12)
        q = self.cm.step(q0, 'A', 10.0)                                    # stick saturates at its top stop
        self.assertAlmostEqual(q[2], 1.01); self.assertAlmostEqual(q[0], math.radians(-430.0))   # yaw continues
        q2 = self.cm.step(q, 'A', 1.0)
        self.assertEqual(q2[2], q[2]); self.assertAlmostEqual(q2[0] - q[0], math.radians(-43.0))  # pure yaw
        qb = self.cm.step(q0, 'B', 0.2)
        np.testing.assert_allclose(qb - q0, np.radians([62 * 0.2, 0, -28 * 0.2, 0]), atol=1e-12)
        self.assertEqual(self.cm.gpio['A'], 23); self.assertEqual(self.cm.gpio['B'], 22)
        self.assertEqual(self.cm.channel_from_duties([0.2, -0.6, 0.1, 0.0]), ('bucket-', 0.6))
        self.assertEqual(self.cm.channel_from_duties([0.1, 0.2, 0.25, 0.0]), (None, 0.0))       # < 30 ms
        self.assertEqual(self.cm.duty_row('B', 0.7), {'duty_boom': 0.0, 'duty_bucket': 0.0, 'duty_A': 0.0, 'duty_B': 0.7})

    def test_placeholder_rates_without_coupling(self):
        from digital_twin.hardware.commands import ChannelModel
        hw = json.loads(json.dumps(self.hw)); hw.pop('coupling')
        warns = []
        cm = ChannelModel.from_config(hw, {n: {'lo': -1, 'hi': 1} for n in ('slew', 'boom', 'stick', 'bucket')}, warn=warns.append)
        self.assertIn('placeholder', cm.source); self.assertTrue(warns)
        self.assertAlmostEqual(math.degrees(cm.rates['A'][0]), 63.0)

    def test_coupled_dead_reckoning(self):
        est = se.CoupledDeadReckoningEstimator.from_channel_model(self.cm, [0, 0.2, 0.9, 0.3])
        est.update_from_pulses([{'t_start_s': 0.0, 'channel': 'A', 'duration_ms': 1000},
                                {'t_start_s': 1.0, 'joint': 'boom', 'direction': -1, 'duration_ms': 100}], 0.0)
        q = est.joint_state(5.0)
        self.assertAlmostEqual(q[2], 1.01)                                   # stick at its stop
        self.assertAlmostEqual(q[0], math.radians(-43.0))
        self.assertAlmostEqual(q[1], 0.2 + 0.1 * self.cm.rates['boom-'][1])      # boom- rate (< 0; per direction)

    def test_yaw_estimator_integrates_about_gravity_with_rest_bias(self):
        y = se.YawEstimator(rest_s=0.2, bias_tau_s=0.5)
        up = np.array([0.6, 0.8, 0.0])                                      # gravity along a tilted IMU axis
        t = 0.0
        for _ in range(40):                                                  # at rest with a 0.5 dps bias
            y.update(up, 0.5 * up + np.array([0.0, 0.0, 0.3]), t); t += 0.05
        self.assertTrue(y.at_rest); self.assertAlmostEqual(math.degrees(y.bias), 0.5, delta=0.2)
        y0 = y.yaw
        for _ in range(20):                                                  # 1 s at +30 dps about up
            y.update(up, 30.5 * up, t); t += 0.05
        self.assertAlmostEqual(math.degrees(y.yaw - y0), 30.0, delta=2.0)


class PulsePlannerTests(unittest.TestCase):
    def test_planner_tracks_a_synthetic_arc_with_yaw_decoupled(self):
        from digital_twin.hardware.commands import ChannelModel, load_hardware_map
        from digital_twin.sim.coupled import joint_cfg, plan_pulses
        from digital_twin.sim.kinematics import Arm
        arm = Arm('full4_calibrated')
        hw = load_hardware_map()
        hw['slew'] = {'sign_verified': True,
                      'yaw_deg_s': {'left': 58.0, 'right': -58.0}, 'onset_s': {'left': 0.17, 'right': 0.17}}
        cm = ChannelModel.from_config(hw, joint_cfg=joint_cfg(arm), warn=lambda m: None)
        self.assertEqual(cm.kind, 'decoupled')
        q0 = np.array([0.0, 0.1, arm.hi[2], 0.4])
        seq = ['boom-'] * 5 + ['slew-'] * 6 + ['bucket+'] * 5 + ['stick-'] * 4 + ['slew+'] * 6 + ['boom+'] * 5
        q, T = q0.copy(), [arm.fk(q0)]
        for c in seq:
            q = cm.step(q, c, 0.07); T.append(arm.fk(q))
        T = np.array(T)
        self.assertGreater(np.ptp(T[:, 1]), 0.01)                           # the arc leaves the arm plane
        plan = plan_pulses(T, q0, cm, arm)
        err = np.linalg.norm(plan['tip'] - T, axis=1)
        self.assertLess(float(np.sqrt(np.mean(err ** 2))), 0.006)
        self.assertTrue({'slew+', 'slew-'} & set(plan['channel']))         # it yaws through the base pins
        d = plan['duties']
        self.assertEqual(list(d.columns), ['duty_boom', 'duty_bucket', 'duty_stick', 'duty_slew'])
        self.assertTrue(((d.abs() > 0).sum(1) <= 1).all())
        self.assertTrue({p['gpio'] for p in plan['pulses']} <= {33, 32, 26, 25, 23, 22, 19, 21})
        self.assertEqual(ChannelModel.channel_from_duties([0, 0, 0, -0.6], kind='decoupled'), ('slew-', 0.6))

    def test_planner_tracks_a_synthetic_arc_with_yaw(self):
        from digital_twin.hardware.commands import ChannelModel, load_hardware_map
        from digital_twin.sim.coupled import joint_cfg, plan_pulses
        from digital_twin.sim.kinematics import Arm
        arm = Arm('full4_calibrated')
        hw = load_hardware_map(); hw.pop('slew', None); hw.pop('coupling', None)    # the old wiring: A / B placeholders
        cm = ChannelModel.from_config(hw, joint_cfg=joint_cfg(arm), warn=lambda m: None)
        q0 = np.array([0.0, 0.1, arm.hi[2], 0.4])
        # target: the tip path of a known feasible channel sequence (a yawing, dipping arc)
        seq = ['boom-'] * 5 + ['B'] * 6 + ['bucket+'] * 5 + ['A'] * 6 + ['boom+'] * 5
        q, T = q0.copy(), [arm.fk(q0)]
        for c in seq:
            q = cm.step(q, c, 0.07); T.append(arm.fk(q))
        T = np.array(T)
        self.assertGreater(np.ptp(T[:, 1]), 0.01)                           # the arc leaves the arm plane
        plan = plan_pulses(T, q0, cm, arm)
        err = np.linalg.norm(plan['tip'] - T, axis=1)
        self.assertLess(float(np.sqrt(np.mean(err ** 2))), 0.006)
        self.assertTrue({'A', 'B'} & set(plan['channel']))                  # it yaws through the coupled channels
        d = plan['duties']
        self.assertEqual(list(d.columns), ['duty_boom', 'duty_bucket', 'duty_A', 'duty_B'])
        self.assertTrue(((d.abs() > 0).sum(1) <= 1).all())                  # one channel per tick
        self.assertTrue(all(p['duration_ms'] >= 30 for p in plan['pulses']))
        self.assertTrue({p['gpio'] for p in plan['pulses']} <= {33, 32, 26, 25, 23, 22})

    def test_replay_coupled_planner_runs(self):
        from digital_twin.sim.replay import run, synthetic_trajectory
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'traj.csv'
            synthetic_trajectory(duration=3.0).to_csv(path, index=False)
            m = run(path, Path(tmp) / 'out', profile='full4_calibrated', render=False, planner='coupled')
            self.assertEqual((m['profile'], m['planner']), ('full4_calibrated', 'coupled'))
            self.assertLess(m['tip_tracking']['rmse_m'], 0.05)          # moves with the live calibration
            cols = pd_read(Path(tmp) / 'out' / 'actions_pulse.csv').columns
            duty = ('duty_stick', 'duty_slew') if 'duty_slew' in cols else ('duty_A', 'duty_B')
            for c in ('t_s', 'duty_boom', 'duty_bucket', *duty, 'q_est_slew', 'q_est_stick',
                      'tip_target_x', 'tip_achieved_z'):
                self.assertIn(c, cols)


def pd_read(path):
    import pandas as pd
    return pd.read_csv(path)



class FakePiDashboard:
    """Local stand-in for umi-data-collection: /stream.mjpg (multipart JPEG, 20 fps), /preview.jpg
    (204 on alternate requests), /api/status (IMU sample). ``stream_ok=False`` answers 503 on the stream."""

    def __init__(self, stream_ok=True):
        import http.server
        import io as _io
        from PIL import Image
        buf = _io.BytesIO(); Image.fromarray(np.full((720, 1280, 3), 120, np.uint8)).save(buf, 'JPEG'); self.jpeg = buf.getvalue()
        dash = self
        self.preview_calls = 0
        self.dead = False

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def _reply(self, code, body=b'', ctype='application/json', extra=None):
                self.send_response(code); self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers(); self.wfile.write(body)

            def do_GET(self):
                if dash.dead:                               # link dropped: kill keep-alive connections too
                    self.close_connection = True
                    self.connection.close()
                    return
                if 'umi_access=TOKEN' not in (self.headers.get('Cookie') or ''):
                    return self._reply(401)
                if self.path == '/api/status':
                    st = {'camera': {'live': True, 'age_s': 0.01},
                          'imu': {'live': True, 'sample': {'accel': [0.0, 0.98, -0.2], 'gyro': [0.1, 0.0, 0.0]}}}
                    return self._reply(200, json.dumps(st).encode())
                if self.path.startswith('/preview.jpg'):
                    dash.preview_calls += 1
                    if dash.preview_calls % 2:
                        return self._reply(204)
                    return self._reply(200, dash.jpeg, 'image/jpeg', {'X-Frame-Sequence': str(dash.preview_calls),
                                                                      'X-Frame-Age-Ms': '5'})
                if self.path == '/stream.mjpg':
                    if not stream_ok:
                        return self._reply(503, b'{}')
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Connection', 'close'); self.end_headers()
                    try:
                        for _ in range(200):
                            if dash.dead:
                                break
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                             + str(len(dash.jpeg)).encode() + b'\r\n\r\n' + dash.jpeg + b'\r\n')
                            self.wfile.flush(); time.sleep(0.05)
                    except OSError:
                        pass
                    self.close_connection = True
                    return
                self._reply(404)

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.host = f'127.0.0.1:{self.server.server_address[1]}'

    def close(self):
        self.dead = True
        self.server.shutdown(); self.server.server_close()


class PiSensorSourceTests(unittest.TestCase):
    def _source(self, stream_ok):
        dash = FakePiDashboard(stream_ok)
        tok = Path(tempfile.mkdtemp()) / 'token'; tok.write_text('TOKEN\n')
        src = pr.PiSensorSource(dash.host, tok)
        self.addCleanup(dash.close); self.addCleanup(src.close)
        return src, dash

    def test_stream_gives_new_full_res_frames_with_low_wait(self):
        src, dash = self._source(True)
        img, t, meta = src.grab()
        self.assertEqual(img.shape, (720, 1280, 3)); self.assertEqual(meta['source'], 'stream.mjpg')
        waits = []
        for _ in range(5):
            img2, t2, m2 = src.grab()
            self.assertGreater(t2, t); t = t2                                  # never the same frame twice
            waits.append(m2['fetch_wait_ms'])
        self.assertLess(np.median(waits), 80)                                  # ~one frame interval (50 ms)
        time.sleep(0.15)
        got, st = src.imu()
        np.testing.assert_allclose(got[0], [0.0, 0.98, -0.2])

    def test_host_preference_skips_a_refusing_host(self):
        dash = FakePiDashboard(True); self.addCleanup(dash.close)
        tok = Path(tempfile.mkdtemp()) / 'token'; tok.write_text('TOKEN\n')
        logs = []
        src = pr.PiSensorSource(f'127.0.0.1:1,{dash.host}', tok, log=logs.append); self.addCleanup(src.close)
        img, t, meta = src.grab()
        self.assertEqual(src.base, f'http://{dash.host}')
        self.assertTrue(any('tried http://127.0.0.1:1 failed' in l for l in logs))
        self.assertEqual(pr.link_kind('http://169.254.10.2:8000'), 'wired (link-local)')
        self.assertEqual(pi.parse_pi_hosts('169.254.10.2:8000, 192.168.0.20:8000'),
                         ['http://169.254.10.2:8000', 'http://192.168.0.20:8000'])
        with self.assertRaisesRegex(pr.RunnerError, 'unreachable on every host'):
            pr.PiSensorSource('127.0.0.1:1', tok).start()

    def test_fails_over_mid_run_without_aborting(self):
        a, b = FakePiDashboard(True), FakePiDashboard(True)
        self.addCleanup(b.close)
        tok = Path(tempfile.mkdtemp()) / 'token'; tok.write_text('TOKEN\n')
        logs = []
        src = pr.PiSensorSource(f'{a.host},{b.host}', tok, log=logs.append); self.addCleanup(src.close)
        src.grab()
        self.assertEqual(src.base, f'http://{a.host}')
        a.close()                                                               # preferred link drops
        deadline = time.monotonic() + 8
        while src.base != f'http://{b.host}' and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(src.base, f'http://{b.host}')
        img, t, meta = src.grab()                                               # frames keep coming
        self.assertEqual(img.shape, (720, 1280, 3))
        self.assertTrue(any('re-probing' in l for l in logs))
        self.assertIn('failover', src.host_events[-1][2])

    def test_falls_back_to_the_preview_when_the_stream_is_refused(self):
        src, dash = self._source(False)
        img, t, meta = src.grab()
        self.assertEqual(img.shape, (720, 1280, 3))
        self.assertIn('preview.jpg', meta['source'])
        self.assertGreater(dash.preview_calls, 1)                              # retried through the 204s



class SlowFakePolicy(FakePolicy):
    STEP = 0.004                                    # rad per action: far below one minimum pulse


class ExecLeadTests(FakePolicyMixin, unittest.TestCase):
    def test_sub_minimum_targets_are_logged_and_the_state_is_the_estimate(self):
        args = make_args(['--gateway-url', 'wss://g', '--max-steps', '4', '--exec-latency-s', '0'])
        sim = sim_executor(); sim.infer_latency_s = 0.05
        logs = []
        runner = pr.PolicyRunner(args, policy=SlowFakePolicy(5), executor=sim, camera=SimCamera(sim), log=logs.append)
        runner.run()
        self.assertFalse([p for p in runner.pulse_log if p['phase'] == 'policy'])   # every step below one pulse
        self.assertTrue(any('below the minimum pulse' in l for l in logs))
        states = [r.state for r in FakePolicy.instances[-1].requests]
        np.testing.assert_allclose(states[-1], states[0], atol=1e-6)        # nothing sent: the estimate stays

    def test_auto_lead_from_probe_and_onset(self):
        hw = json.loads(json.dumps(pr.load_hw()))           # independent of the live calibration's fitted values
        for j in ('boom', 'bucket'):
            hw.setdefault('calibration', {}).setdefault('speeds', {}).setdefault(j, {})['dead_time_s'] = {
                'positive': -0.1, 'negative': -0.05}
        onsets, _ = pr.pulse_onsets(hw)
        self.assertEqual(onsets['boom'], pr.DEFAULT_ONSET_S)                  # negative fitted dead times: unusable
        hw['calibration']['speeds']['boom']['dead_time_s'] = {'positive': 0.04, 'negative': 0.06}
        self.assertAlmostEqual(pr.pulse_onsets(hw)[0]['boom'], 0.06)
        hw['calibration']['speeds']['boom']['dead_time_s'] = {'positive': -0.1, 'negative': -0.05}
        st = pr.latency_stats([0.05, 0.06, 0.07, 0.3])
        self.assertAlmostEqual(st['p50'], 0.065); self.assertGreater(st['p90'], 0.2)
        fake = FakeBridge(delay_s=0.03)
        self.addCleanup(fake.close)
        ex = pr.BridgeExecutor(hw, fake.host, dry_run=False, pace=False, log=lambda *_: None)   # not connected:
        rtts = ex.probe_latency(4)                                                               # a short session
        self.assertEqual(len(rtts), 4)
        self.assertTrue(all(0.025 < r < 0.5 for r in rtts))
        quiet = pr.PolicyRunner(make_args(['--home-only']), executor=pr.BridgeExecutor(hw, fake.host, dry_run=False),
                                hw=hw, log=lambda *_: None)
        self.assertAlmostEqual(quiet.update_exec_lead(probe=False), pr.PRIOR_ACK_S + pr.DEFAULT_ONSET_S)   # not measured
        runner = pr.PolicyRunner(make_args(['--home-only']), executor=ex, hw=hw, log=lambda *_: None)
        lead = runner.update_exec_lead(probe=True)
        self.assertAlmostEqual(lead, runner.latency_info['ack_p90_s'] + pr.DEFAULT_ONSET_S)
        self.assertTrue(0.17 < lead < 0.7)
        self.assertEqual(fake.moves, [])                                     # only 'state' was sent
        fixed = pr.PolicyRunner(make_args(['--home-only', '--exec-latency-s', '0.1']), executor=ex,
                                hw=hw, log=lambda *_: None)
        self.assertEqual(fixed.update_exec_lead(), 0.1)


class FakeBridge:
    """Minimal GPIO-bridge websocket stand-in: answers ``state`` after ``delay_s``; records anything else and acks
    triggers."""

    def __init__(self, delay_s=0.02):
        from websockets.sync.server import serve
        self.moves = []

        def handler(ws):
            ws.send(json.dumps({'type': 'bridge', 'esp32': True}))
            for raw in ws:
                msg = json.loads(raw)
                if msg.get('cmd') == 'state':
                    time.sleep(delay_s); ws.send(json.dumps({'type': 'state', 'pins': []}))
                else:
                    self.moves.append(msg)
                    if msg.get('cmd') == 'trigger':
                        ws.send(json.dumps({'type': 'ack', 'id': msg.get('id'), 'cmd': 'trigger', 'pin': msg.get('pin')}))
        self.server = serve(handler, '127.0.0.1', 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.host = f'127.0.0.1:{self.server.socket.getsockname()[1]}'

    def close(self):
        self.server.shutdown()



class PolicySpaceTests(FakePolicyMixin, unittest.TestCase):
    def test_only_the_calibrated_joint_space_runs(self):
        args = make_args(['--gateway-url', 'ws://127.0.0.1:8040',
                          '--max-steps', '1'])
        with self.assertRaisesRegex(pr.RunnerError, "joint space 'handheld_reconstructed'"):
            pr.PolicyRunner(args, policy=FakePolicy(policy_space='handheld_reconstructed'), executor=sim_executor(),
                            camera=NativeCamera(), log=lambda *_: None).run()


class StartPoseTests(FakePolicyMixin, unittest.TestCase):
    def test_explicit_start_pose_in_real_joint_space(self):
        hw = pr.load_hw()
        home = pr.home_values(hw)
        target = np.array([0.0, home['boom'] - 0.15, home['stick'], home['bucket'] + 0.5])
        args = make_args(['--gateway-url', 'ws://127.0.0.1:8040', '--max-steps', '1',
                          '--exec-latency-s', '0', '--start-pose', ','.join(f'{v:.4f}' for v in np.degrees(target))])
        sim = sim_executor(); sim.infer_latency_s = 0.01
        runner = pr.PolicyRunner(args, policy=fake_policy(), executor=sim, camera=SimCamera(sim), log=lambda *_: None)
        runner.run()
        np.testing.assert_allclose(runner.summary['start_pose']['deg'], np.degrees(target), atol=0.01)
        self.assertTrue(any(p['phase'] == 'start_pose' for p in runner.pulse_log))
        np.testing.assert_allclose(FakePolicy.instances[-1].requests[0].state, target, atol=0.02)
        self.assertEqual(make_args([]).start_pose, 'none')                          # default: the standard pose


class CancellationAndCliTests(FakePolicyMixin, unittest.TestCase):
    def _runner(self, argv, **kw):
        args = make_args(['--gateway-url', 'wss://g', '--exec-latency-s', '0'] + argv)
        sim = sim_executor(); sim.infer_latency_s = 0.05
        return pr.PolicyRunner(args, policy=fake_policy(3), executor=sim,
                               camera=SimCamera(sim), log=kw.get('log', lambda *_: None)), sim

    def test_stop_ends_the_run_and_releases(self):
        runner, sim = self._runner(['--max-steps', '50'])
        released = []
        sim.release_all = lambda: released.append(1)
        runner.external_hook = lambda r, t: r.stop('test') if r.summary['steps'] >= 2 else None
        summary = runner.run()
        self.assertEqual(summary['stopped'], 'test')
        self.assertLessEqual(summary['steps'], 3)
        self.assertTrue(released)

    def test_stop_during_homing_between_pulses(self):
        runner, sim = self._runner(['--home-only'])
        orig = sim.execute
        def one_pulse_then_stop(pulses, gap_ms=0.0, phase='policy', heartbeat=None):
            res = orig(pulses[:1], gap_ms, phase, heartbeat); runner.stop('dashboard'); return res
        sim.execute = one_pulse_then_stop
        summary = runner.run()
        self.assertEqual(summary['stopped'], 'dashboard')
        self.assertNotIn('homing', summary)

    def test_infer_hard_timeout(self):
        runner, sim = self._runner(['--max-steps', '3', '--infer-timeout-s', '0.2', '--start-pose', 'none'])
        slow = runner.policy
        real = slow.infer
        slow.infer = lambda image, state, mask=None: (time.sleep(1.0), real(image, state, mask))[1]
        runner.camera.grab()                 # first render (GL context) can take ~1 s on a headless node: not timed
        t0 = time.monotonic()
        with self.assertRaisesRegex(pr.RunnerError, 'inference timed out'):
            runner.run()
        self.assertLess(time.monotonic() - t0, 0.9)

    def test_stop_interrupts_a_slow_inference(self):
        runner, sim = self._runner(['--max-steps', '3', '--infer-timeout-s', '10', '--start-pose', 'none'])
        slow = runner.policy
        real = slow.infer
        slow.infer = lambda image, state, mask=None: (time.sleep(2.0), real(image, state, mask))[1]
        runner.camera.grab()                 # warm the renderer before timing
        threading.Timer(0.3, lambda: runner.stop('dashboard')).start()
        t0 = time.monotonic()
        summary = runner.run()
        self.assertEqual(summary['stopped'], 'dashboard'); self.assertLess(time.monotonic() - t0, 1.5)

    def test_camera_on_another_clock_is_refused(self):
        runner, sim = self._runner(['--max-steps', '2', '--start-pose', 'none'])
        cam = runner.camera
        runner.camera = SimpleNamespace(grab=lambda: (cam.grab()[0], time.monotonic(), {}))   # wall clock vs sim clock
        with self.assertRaisesRegex(pr.RunnerError, 'runner clock'):
            runner.run()

    def test_cli_has_no_sim_or_file_paths(self):
        opts = {a for act in pr.build_parser()._actions for a in act.option_strings}
        for gone in ('--camera', '--camera-file', '--backend', '--sim-latency-s', '--estimator', '--action-space',
                     '--policy-space', '--dataset-version', '--retarget-scale', '--self-mask',
                     '--no-joint-transform'):
            self.assertNotIn(gone, opts)
        self.assertIn('--pi-host', opts); self.assertIn('--token-file', opts)
        self.assertFalse(hasattr(pr, 'SimExecutor')); self.assertFalse(hasattr(pr, 'FileCamera'))

    def test_one_direction_speed_falls_back_to_the_other_measured(self):
        hw = json.loads(json.dumps(pr.load_hw()))
        hw['calibration']['speeds'] = {'bucket': {'speed_measured_rad_s': {'positive': 1.8}},
                                       'boom': {'speed_measured_rad_s': {'negative': 0.9, 'positive': 0.5}}}
        warns = []
        cfg, src = real_joint_config(hw, None, warns.append)
        self.assertEqual((cfg['bucket']['v_pos'], cfg['bucket']['v_neg']), (1.8, 1.8))
        self.assertEqual((cfg['boom']['v_pos'], cfg['boom']['v_neg']), (0.5, 0.9))
        self.assertTrue(any('bucket' in w and 'measured positive' in w for w in warns))


# ------------------------------------------------------------------ pi05_tabletop_excavator (4-joint CLOAK client)
PIN_MAP = {19, 21, 22, 23, 25, 26, 32, 33}                     # the hardware map's pins
JOINT_PINS = {'slew': {19, 21}, 'stick': {22, 23}, 'bucket': {25, 26}, 'boom': {32, 33}}


class FakeCloak:
    """Stands in for digital_twin.policy.cloak.CloakPolicy (same constructor and interface): 4 joints in
    [slew, boom, stick, bucket] order, 10 actions; alternates between two goals so every joint moves both ways."""
    GOALS = (np.array([0.4, -0.15, 0.55, 1.0]), np.array([-0.4, 0.45, 0.95, -0.5]))
    STEP = 0.08

    def __init__(self, gateway_url='ws://g', prompt='scoop the dirt', needs_mask=True, **kw):
        self.url, self.prompt, self.needs_mask = gateway_url, prompt, needs_mask
        self.requests, self.resets, self.closed = [], 0, False

    def metadata(self):
        return {'camera_names': ['rgb'], 'action_keys': ['actions'], 'action_horizon': 10, 'action_dim': 4,
                'joints': list(se.FULL4_JOINTS), 'policy_backend': 'cloak', 'policy_space': 'calibrated',
                'mode': 'cloak_oov' if self.needs_mask else 'baseline', 'needs_mask': self.needs_mask}

    def infer(self, rgb, state, mask=None):
        state = np.asarray(state, float)
        self.requests.append(SimpleNamespace(rgb=rgb, state=state.copy(), mask=mask))
        goal, rows, q = self.GOALS[(len(self.requests) // 2) % 2], [], state.copy()
        for _ in range(10):
            q = q + np.clip(goal - q, -self.STEP, self.STEP); rows.append(q.copy())
        return np.array(rows, np.float32), 1e-3, 0.5

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


class FakeMasker:
    def __init__(self):
        self.calls = []

    def mask(self, q, shape):
        self.calls.append((dict(q), tuple(shape)))
        m = np.zeros(shape[:2], bool); m[-200:, 400:900] = True            # "robot" at the bottom
        return m

    def describe(self):
        return 'self-mask: fake'

    def close(self):
        pass


class NativeCamera:
    """1280x720 frames with a gradient (any padding / resize / blanking would change them)."""
    def __init__(self):
        y, x = np.mgrid[0:720, 0:1280]
        self.frame = np.stack([x % 256, y % 256, (x + y) % 256], -1).astype(np.uint8)

    def grab(self, after=None):
        return self.frame.copy(), time.monotonic(), {'frame_shape': [720, 1280, 3]}


class Cloak4Tests(unittest.TestCase):
    def _executor(self, hw=None, log=lambda *_: None):
        """The runner's bridge executor (live: every run sends) against a fake GPIO bridge that acks."""
        self.fake = FakeBridge(delay_s=0.001)
        self.addCleanup(self.fake.close)
        return pr.BridgeExecutor(hw or pr.load_hw(), self.fake.host, dry_run=False, pace=False, log=log,
                                 joints=se.FULL4_JOINTS)

    def _run(self, *extra, needs_mask=True, hw=None, sensors=None, steps=6):
        args = make_args(['--profile', 'pi05_tabletop_excavator', '--gateway-url', 'ws://g', '--max-steps', str(steps),
                          '--exec-latency-s', '0', '--start-pose', 'none'] + list(extra))
        logs, pol, masker, cam = [], FakeCloak(needs_mask=needs_mask), FakeMasker(), NativeCamera()
        runner = pr.PolicyRunner(args, policy=pol, executor=self._executor(hw, logs.append), camera=cam,
                                 sensors=sensors, hw=hw, log=logs.append)
        runner.masker = masker
        summary = runner.run()
        return runner, summary, pol, masker, cam, logs

    def test_drives_the_arm_with_native_frames_and_masks_and_never_the_slew(self):
        runner, summary, pol, masker, cam, logs = self._run()
        self.assertEqual(summary['steps'], 6); self.assertEqual(summary['joints'], list(se.FULL4_JOINTS))
        sent = [c for c in runner.executor.sent if c['cmd'] == 'trigger']
        self.assertTrue(sent)                                                     # every run sends: no dry run
        self.assertEqual([m['pin'] for m in self.fake.moves if m['cmd'] == 'trigger'], [c['pin'] for c in sent])
        self.assertLessEqual({int(c['pin']) for c in sent}, PIN_MAP)
        pol_pulses = [p for p in runner.pulse_log if p['phase'] == 'policy']
        self.assertEqual({p['joint'] for p in pol_pulses}, {'boom', 'stick', 'bucket'})   # the slew is locked
        self.assertFalse({19, 21} & {int(c['pin']) for c in sent})
        for p in runner.pulse_log:
            self.assertIn(int(p['gpio']), JOINT_PINS[p['joint']])
        self.assertLessEqual({int(p['gpio']) for p in pol_pulses}, PIN_MAP)
        self.assertEqual(len(pol.requests), 6)
        for r, (q, shape) in zip(pol.requests, masker.calls):
            self.assertEqual(r.rgb.shape, (720, 1280, 3))                         # native: no padding / resize
            np.testing.assert_array_equal(r.rgb, cam.frame)                       # and nothing blanked
            self.assertEqual(r.mask.dtype, bool); self.assertEqual(r.mask.shape, (720, 1280))
            self.assertEqual(shape, (720, 1280)); self.assertEqual(list(q), list(se.FULL4_JOINTS))
            self.assertEqual(q['slew'], 0.0)                                      # the mask sees the real, locked base
            np.testing.assert_allclose(r.state[1:], [q[j] for j in se.FULL4_JOINTS[1:]])
        self.assertEqual(len(masker.calls), 6)
        hw = pr.load_hw()
        np.testing.assert_allclose(pol.requests[0].state[1:], [pr.home_values(hw)[j] for j in ('boom', 'stick', 'bucket')])
        self.assertEqual(pol.requests[0].state[0], 0.0)                          # slew 0 = facing forward
        self.assertEqual(runner.estimator.as_dict(time.monotonic() + 600)['slew'], 0.0)   # not estimated: never moved
        self.assertEqual(summary['unverified_signs'], [])                         # the bucket Pi's calibration verified the arm
        self.assertFalse(any('pin direction not verified' in l for l in logs))
        self.assertIn('mocked', summary['state_sources']['slew'])
        self.assertEqual(summary['state_source_steps']['slew'], {'mocked': 6})
        self.assertEqual(summary['cloak'], {'mode': 'cloak_oov', 'needs_mask': True, 'policy_space': 'calibrated'})
        self.assertNotIn('homing', summary)                                      # a policy run never homes
        self.assertEqual(pol.resets, 1); self.assertTrue(pol.closed)

    def test_the_policy_is_fed_back_its_own_desired_slew(self):
        recs, step = [], pr.PolicyRunner.step
        with mock.patch.object(pr.PolicyRunner, 'step', lambda self, k: recs.append(step(self, k)) or recs[-1]):
            runner, summary, pol, masker, cam, logs = self._run(steps=4)
        self.assertEqual(len(recs), 4)
        self.assertEqual(recs[0]['state'][0], 0.0)
        for prev, rec in zip(recs, recs[1:]):                                     # state slew = last desired slew
            self.assertAlmostEqual(rec['state'][0], prev['slew_desired'], places=6)
            self.assertAlmostEqual(prev['slew_desired'], prev['targets_policy'][-1][0], places=5)
        self.assertTrue(any(abs(r['slew_desired']) > 1e-3 for r in recs))      # the policy did ask to turn
        for rec in recs:
            self.assertTrue(all(t[0] == 0.0 for t in rec['targets']))             # ... and the real target stays put

    def test_baseline_gets_no_mask_and_the_robot_stays_visible(self):
        runner, summary, pol, masker, cam, logs = self._run(needs_mask=False, steps=3)
        self.assertEqual(masker.calls, [])
        for r in pol.requests:
            self.assertIsNone(r.mask); np.testing.assert_array_equal(r.rgb, cam.frame)
        self.assertNotIn('self_mask', summary)

    def test_unverified_joints_warn_but_never_block(self):
        hw = json.loads(json.dumps(pr.load_hw()))
        hw['slew']['sign_verified'] = False; hw['joints']['stick']['sign_verified'] = False
        runner, summary, pol, *_, logs = self._run(hw=hw, steps=2)
        self.assertEqual(summary['unverified_signs'], ['stick'])                  # the locked slew is never pulsed
        self.assertTrue(any(l.startswith('WARNING: pin direction not verified') and "['stick']" in l for l in logs))
        self.assertEqual(summary['steps'], 2)
        self.assertIn('stick', {p['joint'] for p in runner.pulse_log if p['phase'] == 'policy'})

    def test_the_bucket_imu_yaw_does_not_feed_the_slew(self):
        class Sensors:                                   # bucket IMU turning at +20 deg/s about gravity
            def __init__(self):
                self.n = 0

            def _st(self):
                self.n += 1
                return {'imus': {'bucket': {'live': True, 'age_s': 0.0, 'sample': {
                    'accel': [0.0, 0.0, 1.0], 'gyro': [0.0, 0.0, 20.0], 'read_monotonic_ns': self.n}}}}

            def imus(self):
                return {}, self._st()

            def imu(self):
                return None, self._st()
        runner, summary, pol, *_ = self._run(sensors=Sensors(), steps=4)   # arm IMU gate skipped: dead reckoning
        self.assertEqual(summary['state_source_steps']['slew'], {'mocked': 4})
        self.assertEqual(summary['state_source_steps']['boom'], {'dead_reckoning': 4})
        self.assertEqual(runner.estimator.as_dict(time.monotonic())['slew'], 0.0)

    def test_cli_builds_the_cloak_client_and_pi05_tabletop_excavator_is_the_only_profile(self):
        made = []
        def factory(url, prompt, **kw):
            made.append((url, prompt, kw)); return FakeCloak(url, prompt, needs_mask=False)
        args = make_args(['--gateway-url', 'ws://127.0.0.1:8002', '--prompt', 'dig',
                          '--max-steps', '1'])
        self.assertEqual(args.profile, 'pi05_tabletop_excavator')
        self.assertEqual(list(pr.PROFILES), ['pi05_tabletop_excavator'])
        with mock.patch('digital_twin.policy.cloak.CloakPolicy', factory):
            pr.PolicyRunner(args, executor=self._executor(), camera=NativeCamera(), log=lambda *_: None).run()
        self.assertEqual(made[0][:2], ('ws://127.0.0.1:8002', 'dig'))
        with self.assertRaisesRegex(pr.RunnerError, 'chunk'):
            pr.PolicyRunner(make_args(['--gateway-url', 'ws://g', '--chunk', '12']))
        with self.assertRaises(SystemExit), mock.patch('sys.stderr'):
            make_args(['--profile', 'excavator_joint_targets'])
        with self.assertRaisesRegex(pr.RunnerError, '4 angles'):
            self._run('--start-pose', '0,10', steps=1)
        hw = pr.load_hw()
        self.assertEqual(set(excavator.allowed_pins(hw, 'policy', se.FULL4_JOINTS)), PIN_MAP)

    def test_pin_map_validation(self):
        from digital_twin.hardware.commands import validate_hardware_map
        from digital_twin.sim.pins import labels
        hw = pr.load_hw()
        self.assertEqual(set(labels(hw)), PIN_MAP)
        validate_hardware_map(hw)
        for mutate, why in ((lambda h: h['joints']['slew']['positive'].update(pin=5, label='drive forward'), 'drive pin'),
                            (lambda h: h['joints']['slew']['positive'].update(label=' '), 'no label'),
                            (lambda h: h['joints']['slew']['positive'].update(pin='19'), 'must be'),
                            (lambda h: h['joints']['slew']['positive'].update(pin=22), 'GPIO 22 is mapped to both'),
                            (lambda h: h['joints']['stick'].pop('negative'), 'missing'),
                            (lambda h: h['joints'].update(tracks={'status': 'locked'}), 'unknown joint')):
            bad = json.loads(json.dumps(hw)); mutate(bad)
            with self.assertRaisesRegex(ValueError, why):
                validate_hardware_map(bad)


if __name__ == '__main__':
    unittest.main()
