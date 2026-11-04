import asyncio
import json
import threading
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from digital_twin.sim import kinematics as K
from digital_twin.sim.pins import pin
from digital_twin.hardware.commands import (BridgePulseBackend, SimPulseBackend, compile_schedule,
                                   load_hardware_map, validate_hardware_map)
from digital_twin.plan.goto_pose import drive_to_pose, plan_joint_trajectory


def reachable_q(rng, arm, n):
    out = []
    while len(out) < n:
        q = rng.uniform(arm.lo * .8, arm.hi * .8)
        if arm.fk(q)[2] > .03:
            out.append(q)
    return out


H = K.full_chain_q          # hardware [boom, bucket] -> twin q (slew / stick at rest)


class KinematicsTests(unittest.TestCase):
    """The twin's one chain: slew + boom (shoulder) + stick (arm) + bucket (wrist)."""

    @classmethod
    def setUpClass(cls):
        cls.arm = K.Arm()
        cls.model, cls.data = cls.arm.model, mujoco.MjData(cls.arm.model)

    def test_fk_matches_mujoco_and_analytic(self):
        self.assertEqual(K.JOINTS, ['slew', 'boom', 'stick', 'bucket'])
        self.assertEqual(self.arm.joints, K.JOINTS)
        for q in np.random.default_rng(1).uniform(self.arm.lo, self.arm.hi, (25, K.NQ)):
            f = K.fk(self.model, self.data, q)
            self.data.qpos[:] = q; mujoco.mj_forward(self.model, self.data)
            np.testing.assert_allclose(f['tip_pos'], self.data.site('tooth_tip').xpos, atol=1e-12)
            np.testing.assert_allclose(f['bucket_rot'], self.data.body('bucket').xmat.reshape(3, 3), atol=1e-12)
            a = K.analytic_fk(q)
            np.testing.assert_allclose(a['tip_pos'], f['tip_pos'], atol=1e-8)
            np.testing.assert_allclose(a['bucket_rot'], f['bucket_rot'], atol=1e-8)
            # orientation is exactly Rz(slew) R_-y(pitch) R_rest
            np.testing.assert_allclose(self.arm.rotation_for(self.arm.pitch(q), q[0]), f['bucket_rot'], atol=1e-9)
            self.assertAlmostEqual(f['pitch'], self.arm.pitch(q), places=12)
            self.assertAlmostEqual(f['slew'], q[0], places=12)
        # pitch is 0 at the rest pose, where R_rest is taken
        np.testing.assert_allclose(self.arm.pose(K.rest_q())['bucket_rot'], self.arm.R_rest, atol=1e-12)
        self.assertAlmostEqual(self.arm.pitch(K.rest_q()), 0.0, places=12)

    def test_slew_and_stick_held_match_the_welded_rl_model(self):
        """The two-joint model (arm2.xml) is the twin with slew / stick welded at rest."""
        arm2 = mujoco.MjModel.from_xml_path(str(K.SIM / 'models' / 'arm2.xml')); d2 = mujoco.MjData(arm2)
        adr = arm2.jnt_qposadr[[arm2.joint(n).id for n in K.HARDWARE_JOINTS]]
        for qh in np.random.default_rng(3).uniform([-0.8, -0.6], [0.5, 2.4], (10, 2)):
            d2.qpos[adr] = qh; mujoco.mj_kinematics(arm2, d2)
            np.testing.assert_allclose(self.arm.fk(H(qh)), d2.site('tooth_tip').xpos, atol=1e-12)
            np.testing.assert_allclose(self.arm.bucket_rot(), d2.body('bucket').xmat.reshape(3, 3), atol=1e-12)
            self.assertAlmostEqual(self.arm.pitch(H(qh)), qh.sum(), places=12)
            np.testing.assert_allclose(K.hardware_q(H(qh)), qh)

    def test_planar_chain_matches_mujoco(self):
        chain = K.PlanarChain(self.arm)
        for q in np.random.default_rng(4).uniform(self.arm.lo, self.arm.hi, (20, 4)):
            q[0] = 0.0
            np.testing.assert_allclose(chain.fk(q[1:])[0], self.arm.fk(q)[[0, 2]], atol=1e-12)
            q3, ok = chain.ik(*self.arm.fk(q)[[0, 2]], self.arm.pitch(q))
            self.assertTrue(ok.any())
            self.assertTrue(any(np.allclose(q3[b], q[1:], atol=1e-9) for b in range(2) if ok[b]))

    def test_ik_fk_round_trip(self):
        rng = np.random.default_rng(5)
        for point in K.POINTS:
            arm = K.Arm(point=point)
            for q in reachable_q(rng, arm, 15):
                p = arm.fk(q)
                r = K.ik(p, np.degrees(arm.pitch(q)), arm=arm)
                self.assertTrue(r.reachable, (point, q))
                self.assertEqual(r.joints, K.JOINTS)
                f = K.fk(self.model, self.data, r.q)
                got = f['tip_pos'] if point == 'tooth_tip' else f['bucket_pos']
                self.assertLess(np.linalg.norm(got - p), 1e-3)
                self.assertLess(abs(np.degrees(f['pitch']) - np.degrees(arm.pitch(q))), 0.5)

    def test_base_spins_to_targets_off_the_rest_plane(self):
        qh = np.array([-.3, .4]); p = self.arm.fk(H(qh))
        for off in ([0, 0.03, 0], [-0.25, 0.25, 0.0]):
            target = p + off
            # with the slew / stick held (as on the hardware) the offset is out of reach ...
            held = K.ik(target, np.degrees(qh.sum()), arm=self.arm, locked=K.locked_joints())
            self.assertFalse(held.reachable)
            # ... the free chain turns the base to it
            r = K.ik(target, np.degrees(qh.sum()), arm=self.arm)
            self.assertTrue(r.reachable, off)
            self.assertEqual(r.lateral_error_m, 0.0)
            self.assertNotAlmostEqual(r.q[0], 0.0, places=3)                  # the slew did the work
            self.assertLess(np.linalg.norm(r.achieved_pos - target), 1e-3)

    def test_locked_joints_stay_put(self):
        qh = np.array([0.1, 0.8]); p = self.arm.fk(H(qh))
        r = K.ik(p, np.degrees(qh.sum()), arm=self.arm, locked=K.locked_joints())
        self.assertTrue(r.reachable)
        np.testing.assert_allclose(r.q, H(qh), atol=1e-4)
        # held stick: the position fixes the pitch, an impossible one is flagged, not faked
        r = K.ik(p, 170.0, arm=self.arm, locked=K.locked_joints())
        self.assertFalse(r.reachable)
        self.assertGreater(abs(r.pitch_residual_deg), 30)
        self.assertEqual((r.q[0], r.q[2]), (K.locked_joints()['slew'], K.locked_joints()['stick']))

    def test_locked_joints_follow_a_rewritten_map_after_reload(self):
        import os
        import tempfile
        from pathlib import Path
        from digital_twin import paths
        from digital_twin.paths import HARDWARE_MAP
        prev = os.environ.get(paths.HARDWARE_MAP_ENV)
        with tempfile.TemporaryDirectory() as d:
            hwp = Path(d) / 'hardware_map.json'
            hw = json.loads(HARDWARE_MAP.read_text())
            hwp.write_text(json.dumps(hw))
            try:
                self.assertEqual(paths.set_hardware_map(hwp), hwp.resolve())
                old = hw['locked_joints']['stick']['value_rad']
                self.assertEqual(K.locked_joints()['stick'], old)
                hw['locked_joints']['stick']['value_rad'] = old - 0.25
                hwp.write_text(json.dumps(hw))                       # rewritten in place: still the cached read
                self.assertEqual(K.locked_joints()['stick'], old)
                K.reload()
                self.assertAlmostEqual(K.locked_joints()['stick'], old - 0.25)
                self.assertAlmostEqual(K.rest_q()[K.JOINTS.index('stick')], old - 0.25)
                self.assertAlmostEqual(K.pitch_of(K.rest_q()), 0.0)
            finally:
                paths.set_hardware_map(prev)
                K.reload()
        self.assertEqual(K.locked_joints()['stick'], json.loads(paths.hardware_map().read_text())['locked_joints']['stick']['value_rad'])

    def test_pitch_is_independent_of_position(self):
        p = self.arm.fk([0.0, -0.2, 0.6, 0.4])
        lo, hi = K.reachable_envelope(self.arm).pitch_range(p)
        self.assertGreater(np.degrees(hi - lo), 60)
        for phi in np.linspace(lo + 0.1, hi - 0.1, 5):
            r = K.ik(p, np.degrees(phi), arm=self.arm)
            self.assertTrue(r.reachable, np.degrees(phi))
            self.assertLess(abs(r.pitch_residual_deg), 0.5)
        # position only: still solved, the redundant pitch picked by the solver
        self.assertTrue(K.ik(p, arm=self.arm).reachable)

    def test_unreachable_target_flagged(self):
        r = K.ik([0.9, 0.0, 0.1], 0.0, arm=self.arm)
        self.assertFalse(r.reachable)
        self.assertGreater(r.position_residual_m, 0.3)
        self.assertTrue(K.within_limits(r.q, self.arm.lo, self.arm.hi))

    def test_rotation_input_variants_agree(self):
        q = np.array([0.5, -0.2, 0.4, 0.3]); f = K.fk(self.model, self.data, q)
        R = f['bucket_rot']
        variants = [K.ik(f['tip_pos'], np.degrees(f['pitch']), arm=self.arm),
                    K.ik(f['tip_pos'], f['bucket_quat'], arm=self.arm),
                    K.ik(f['tip_pos'], R, arm=self.arm),
                    K.ik(f['tip_pos'], rotvec=Rotation.from_matrix(R).as_rotvec(), arm=self.arm),
                    K.ik(f['tip_pos'], Rotation.from_matrix(R), arm=self.arm)]
        for r in variants:
            self.assertTrue(r.reachable)
            np.testing.assert_allclose(r.q, variants[0].q, atol=1e-4)
        for r in variants[1:]:
            self.assertLess(r.orientation_error_deg, 0.05)
            self.assertLess(r.unachievable_orientation_deg, 0.05)
        # yaw is set by the slew facing the point, so an extra roll cannot be realised: reported, not hidden
        r = K.ik(f['tip_pos'], R @ Rotation.from_euler('x', 20, degrees=True).as_matrix(), arm=self.arm)
        self.assertTrue(r.reachable)
        self.assertAlmostEqual(r.unachievable_orientation_deg, 20.0, delta=0.1)
        self.assertAlmostEqual(r.orientation_error_deg, 20.0, delta=0.2)

    def test_envelope_sweeps_about_the_slew(self):
        env = K.reachable_envelope(self.arm)
        q = np.array([0.0, -0.2, 0.6, 0.4]); p = self.arm.fk(q)
        for s in (-1.5, 0.0, 1.0):
            q[0] = s
            self.assertTrue(env.contains(self.arm.fk(q))[0], s)
        self.assertEqual(env.lateral(p)[0], 0.0)
        self.assertFalse(env.contains(np.array([[0.9, 0.0, 0.1]]))[0])
        # slew / stick held: the planar boom + bucket region only
        held = K.reachable_envelope(self.arm, locked=K.locked_joints())
        self.assertTrue(held.planar)
        self.assertTrue(held.contains(self.arm.fk(H([0.1, 0.8])))[0])
        self.assertLess(held.grid.sum(), env.grid.sum())

    def test_track_ik_keeps_bucket_angle_within_speed_limits(self):
        # the boom and stick sweep faster than they can move while the bucket angle stays constant
        dt = 1 / K.CONTROL_HZ
        n = 12
        boom, stick = np.linspace(-0.5, 0.2, n), np.linspace(0.2, 1.0, n)
        qs = np.c_[np.zeros(n), boom, stick, 0.4 - boom - stick + K.rest_q()[2]]
        T = np.array([self.arm.fk(q) for q in qs]); phi = K.pitch_of(qs)
        self.assertGreater(K.speed_ratio(qs, dt, self.arm.v_pos, self.arm.v_neg).max(), 1.5)
        Qw = K.track_ik(self.arm, T, phi, qs[0], dt)                       # bucket angle weighted
        Q0 = K.track_ik(self.arm, T, phi, qs[0], dt, w_pitch=0.0)          # tip position only
        for Q in (Qw, Q0):
            self.assertLessEqual(K.speed_ratio(Q, dt, self.arm.v_pos, self.arm.v_neg).max(), 1 + 1e-6)
        err = lambda Q: np.abs(K.pitch_of(Q) - phi).mean()
        self.assertLess(err(Qw), 0.5 * err(Q0))                            # the angle is kept ...
        tip = lambda Q: np.mean([np.linalg.norm(self.arm.fk(q) - p) for q, p in zip(Q, T)])
        self.assertGreater(tip(Qw), tip(Q0))                               # ... at the cost of the tip
        # within the speed limits both are met
        slow = np.linspace(qs[0], qs[0] + [0.02, 0.02, 0.02, 0.02], 6)
        Qs = K.track_ik(self.arm, [self.arm.fk(q) for q in slow], K.pitch_of(slow), slow[0], dt)
        np.testing.assert_allclose(K.pitch_of(Qs), K.pitch_of(slow), atol=1e-3)
        self.assertLess(max(np.linalg.norm(self.arm.fk(q) - self.arm.fk(r)) for q, r in zip(Qs, slow)), 1e-3)

    def test_planned_trajectory_respects_speed_limits(self):
        rng = np.random.default_rng(9)
        qs = reachable_q(rng, self.arm, 3)
        q0 = self.arm.q_rest
        plan = plan_joint_trajectory(q0, qs, self.arm.v_pos, self.arm.v_neg, [0.3, 0.0, 0.2])
        ratio = K.speed_ratio(plan['q'], 1 / K.CONTROL_HZ, self.arm.v_pos, self.arm.v_neg)
        self.assertLessEqual(ratio.max(), 1.0)
        self.assertGreater(ratio.max(), 0.8)                      # not needlessly slow
        np.testing.assert_allclose(plan['q'][0], q0)
        np.testing.assert_allclose(plan['q'][-1], qs[-1])
        fast = plan_joint_trajectory(q0, qs, self.arm.v_pos, self.arm.v_neg, duration=1.0)
        self.assertGreater(max(fast['max_speed_ratio'].values()), 1.0)   # forced duration: violation reported

    def test_goto_pose_drives_all_four_joints(self):
        wps = [{'pos': self.arm.fk(H([-0.3, 0.3])).tolist(), 'pitch': np.degrees(0.0), 'dwell_s': 0.2},
               {'pos': [0.20, 0.18, 0.08], 'pitch': -30.0}]
        o = drive_to_pose(wps, settle_s=1.5, arm=self.arm)
        m = o['metrics']
        self.assertTrue(m['all_reachable']); self.assertTrue(m['speed_limits_respected'])
        self.assertEqual(m['joints'], K.JOINTS)
        self.assertLess(m['kinematic_final']['tip_error_m'], 1e-3)
        self.assertLess(m['dynamic_final']['tip_error_m'], 0.01)
        self.assertLess(abs(m['dynamic_final']['pitch_error_deg']), 2.0)
        self.assertFalse(m['dynamic_tracking']['unstable'])
        # the hardware cannot slew: with hardware=True slew / stick are held and that target is out of reach
        o = drive_to_pose(wps[1:], arm=self.arm, hardware=True)
        self.assertFalse(o['metrics']['all_reachable'])
        self.assertEqual(set(o['locked']), {'slew', 'stick'})


class PulseScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm = K.Arm()
        cls.hw = load_hardware_map()

    def _ramp(self, q1, T=2.0):
        """Twin trajectory moving boom / bucket by q1 from rest (slew / stick held)."""
        t = np.arange(0, T + 1e-9, 0.05)
        Q = H(np.outer(np.clip(t / (T * .8), 0, 1), q1))
        return t, Q

    def test_durations_sequence_and_pins(self):
        t, Q = self._ramp(np.array([-0.3, 0.6]))
        s = compile_schedule(t, Q, self.arm, min_pulse_ms=50, gap_ms=100)
        ps = s['pulses']
        self.assertEqual([p['joint'] for p in ps], ['boom', 'bucket'])
        self.assertEqual(ps[0]['gpio'], 32); self.assertEqual(ps[0]['direction'], -1)   # boom down (boom down)
        self.assertEqual(ps[1]['gpio'], 25); self.assertEqual(ps[1]['direction'], 1)    # bucket out (bucket up)
        self.assertEqual(ps[0]['duration_ms'], round(0.3 / self.arm.v_neg[1] * 1000))
        self.assertEqual(ps[1]['duration_ms'], round(0.6 / self.arm.v_pos[3] * 1000))
        self.assertEqual(s['held_joints'], ['slew', 'stick'])
        self.assertEqual(s['warnings'][:0], [])
        # hardware-only columns [boom, bucket] compile to the same pulses
        s2 = compile_schedule(t, K.hardware_q(Q), self.arm, min_pulse_ms=50, gap_ms=100)
        self.assertEqual(s2['pulses'], ps)
        for a, b in zip(ps, ps[1:]):                           # one pin at a time, gap respected
            self.assertGreaterEqual(b['t_start_s'] + 1e-9, a['t_start_s'] + a['duration_ms'] / 1000 + 0.1)
        # the unverified-sign warning appears exactly when the map still has an unverified pin joint
        unverified = any(not self.hw['joints'][p['joint']].get('sign_verified', True) for p in ps)
        self.assertEqual(any(w.startswith('pin direction') for w in s['warnings']), unverified)

    def test_min_pulse_merges_small_deltas(self):
        t = np.arange(0, 3.0001, 0.05)
        Q = H(np.zeros((len(t), 2))); Q[:, 3] = 0.005 * t / 0.5   # bucket: 5 mrad per 0.5 s window << 50 ms
        s = compile_schedule(t, Q, self.arm, chunk_s=0.5, min_pulse_ms=50)
        self.assertTrue(all(p['duration_ms'] >= 50 for p in s['pulses']))
        self.assertEqual(len(s['pulses']), 0)                     # 25 ms total: merged, never emitted
        Q[:, 3] = 0.03 * t                                         # 90 mrad total -> 75 ms total
        s = compile_schedule(t, Q, self.arm, chunk_s=0.5, min_pulse_ms=50)
        self.assertEqual(len(s['pulses']), 1)
        self.assertGreaterEqual(s['pulses'][0]['duration_ms'], 50)
        self.assertLess(abs(s['residual_deg']['bucket']), np.degrees(0.05 * self.arm.v_pos[3]) + 1e-6)

    def test_only_boom_bucket_pins_emitted_and_welded_joints_refused(self):
        t, Q = self._ramp(np.array([-0.3, 0.6]))
        s = compile_schedule(t, Q, self.arm)
        mapped = {pin(self.hw, j, d) for j in ('boom', 'bucket') for d in (1, -1)}
        self.assertLessEqual({p['gpio'] for p in s['pulses']}, mapped)
        self.assertEqual({p['joint'] for p in s['pulses']}, {'boom', 'bucket'})
        self.assertEqual(s['held_joints'], ['slew', 'stick'])       # no pins of their own: held
        self.assertEqual(set(self.hw['joints']), {'slew', 'boom', 'stick', 'bucket'})   # all four pin-mapped
        s4 = compile_schedule(t, Q + np.array([0.2, 0, -0.2, 0]) * (t / t[-1])[:, None], self.arm, joints=K.JOINTS)
        self.assertEqual({p['joint'] for p in s4['pulses']}, {'slew', 'boom', 'stick', 'bucket'})   # opt-in 4-joint emit
        self.assertEqual(s4['held_joints'], [])
        self.assertLessEqual({p['gpio'] for p in s4['pulses']}, {19, 21, 22, 23, 25, 26, 32, 33})
        bad = json.loads(json.dumps(self.hw))
        bad['joints']['boom']['positive']['pin'] = 22           # the stick's down pin: one pin, two joints
        with self.assertRaises(ValueError):
            validate_hardware_map(bad)
        with self.assertRaises(ValueError):
            compile_schedule(t, Q, self.arm, hw_map=bad)
        bad = json.loads(json.dumps(self.hw))
        bad['joints']['slew'] = {'status': 'available', 'positive': {'pin': 26, 'label': 'x'},
                                 'negative': {'pin': 21, 'label': 'base right'}}
        with self.assertRaises(ValueError):
            validate_hardware_map(bad)
        # a plan that moves the stick is held there, with a warning
        Qs = Q.copy(); Qs[:, 2] -= 0.3 * t / t[-1]
        self.assertTrue(any(w.startswith('stick') for w in compile_schedule(t, Qs, self.arm)['warnings']))
        for joint, gpio in (('slew', 27), ('stick', 26)):           # locked pin / unlocked pin, held joint
            forged = {'pulses': [{'t_start_s': 0, 'joint': joint, 'direction': 1, 'duration_ms': 100, 'gpio': gpio,
                                  'label': '', 'sign_verified': False}], 'q_start': [0, 0], 'gap_ms': 0}
            with self.assertRaises(PermissionError):
                SimPulseBackend(self.arm).execute(forged)
            with self.assertRaises(PermissionError):
                BridgePulseBackend(dry_run=True, log=lambda *_: None).execute(forged)

    def test_sim_backend_reaches_boom_bucket_target(self):
        q_goal = np.array([-0.4, 0.9])                             # boom, bucket
        target = self.arm.fk(H(q_goal))
        o = drive_to_pose([{'pos': target.tolist(), 'pitch': np.degrees(q_goal.sum())}], arm=self.arm,
                          hardware=True, pulse_backend=SimPulseBackend(K.Arm(), latency_s=0.2))
        m = o['metrics']
        self.assertTrue(m['all_reachable'])
        self.assertEqual(m['pulses']['n_pulses'], 2)
        self.assertLess(m['pulses']['final']['tip_error_m'], 0.01)     # held slew / stick are servoed, not welded
        self.assertLess(abs(m['pulses']['final']['pitch_error_deg']), 2.0)


class FakeBridge:
    """Tiny in-process stand-in for the hub's GPIO bridge websocket (/api/gpio/ws)."""

    def __init__(self):
        self.frames, self.port = [], None
        self._ready = threading.Event()

    def __enter__(self):
        self.thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self.thread.start(); self._ready.wait(5)
        return self

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.stop.set_result, None); self.thread.join(5)

    async def _main(self):
        from websockets.asyncio.server import serve
        self.loop = asyncio.get_running_loop(); self.stop = self.loop.create_future()

        async def handler(ws):
            await ws.send(json.dumps({'type': 'bridge', 'esp32': True, 'target': 'fake'}))
            async for raw in ws:
                msg = json.loads(raw); self.frames.append(msg)
                if msg.get('cmd') in ('trigger', 'press', 'release'):
                    await ws.send(json.dumps({'type': 'ack', 'id': msg.get('id'), 'cmd': msg['cmd'],
                                              'pin': msg.get('pin')}))

        async with serve(handler, 'localhost', 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self.stop


class BridgeBackendTests(unittest.TestCase):
    schedule = {'pulses': [
        {'t_start_s': 0.0, 'joint': 'boom', 'direction': 1, 'duration_ms': 60, 'gpio': 33, 'label': 'Big arm up',
         'sign_verified': False},
        {'t_start_s': 0.1, 'joint': 'bucket', 'direction': -1, 'duration_ms': 50, 'gpio': 26, 'label': 'Gripper in',
         'sign_verified': False}], 'q_start': [0, 0], 'gap_ms': 20}

    def test_dry_run_sends_nothing(self):
        with FakeBridge() as fb:
            res = BridgePulseBackend(f'localhost:{fb.port}', log=lambda *_: None).execute(self.schedule)
        self.assertTrue(res['dry_run']); self.assertEqual(res['sent'], 0); self.assertEqual(fb.frames, [])

    def test_live_requires_acknowledgement_flag(self):
        with self.assertRaises(PermissionError):
            BridgePulseBackend('localhost:1', dry_run=False, log=lambda *_: None).execute(self.schedule)

    def test_live_against_fake_bridge_sends_triggers_then_release_all(self):
        with FakeBridge() as fb:
            res = BridgePulseBackend(f'localhost:{fb.port}', dry_run=False,
                                     i_understand_open_loop=True).execute(self.schedule)
        cmds = [(f['cmd'], f.get('pin'), f.get('ms')) for f in fb.frames]
        self.assertEqual(cmds, [('trigger', 33, 60), ('trigger', 26, 50), ('release', 'all', None)])
        self.assertEqual(res['sent'], 2)
        ids = [f['id'] for f in fb.frames]
        self.assertEqual(len(set(ids)), len(ids))



class SolveActionsTests(unittest.TestCase):
    """Replays made with the earlier two-joint twin are re-solved from their targets on the twin."""

    def test_two_joint_targets_are_reproduced_and_lateral_restored(self):
        from digital_twin.sim.replay import solve_actions
        arm = K.Arm()
        qh = np.array([(0.0, 0.5), (0.2, 1.0), (-0.3, 0.3)])
        T = np.array([arm.fk(H(q)) for q in qh])
        acts = {'t_s': np.arange(3) * 0.05, 'tip_target_x': T[:, 0], 'tip_target_y': T[:, 1],
                'tip_target_z': T[:, 2], 'pitch_target': qh.sum(1)}
        r = solve_actions(acts, arm=arm)
        self.assertEqual(r['q'].shape, (3, 4))
        self.assertLess(r['tip_err'].max(), 1e-4)
        np.testing.assert_allclose(K.pitch_of(r['q']), qh.sum(1), atol=1e-3)
        # a recorded lateral offset (dropped by the two-joint replay) is put back: the slew reaches it
        r = solve_actions({**acts, 'tip_target_lateral_m': np.full(3, 0.04)}, arm=arm)
        self.assertLess(r['tip_err'].max(), 1e-4)
        self.assertGreater(np.abs(r['q'][:, 0]).min(), 0.05)


if __name__ == '__main__':
    unittest.main()
