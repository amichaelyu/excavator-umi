import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from digital_twin.paths import MODELS
from digital_twin.sim.kinematics import JOINTS as NQ_NAMES, full_chain_q, locked_joints, speed_ratio
from digital_twin.sim.replay import (
    NQ, Arm, PlanarChain, apply_mount, apply_transform, fit_retarget, heading_info, load_trajectory, parse_mount,
    pitch_track, reachable_envelope, resample, resolve_mount, run, solve_ik, synthetic_trajectory)
from scipy.spatial.transform import Rotation


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm = Arm()
        cls.env = reachable_envelope(cls.arm)

    def test_planar_chain_matches_mujoco(self):
        chain = PlanarChain(self.arm)
        for q in np.random.default_rng(3).uniform(self.arm.lo, self.arm.hi, (30, NQ)):
            q[0] = 0.0                                          # the chain lives in the slew-0 plane
            np.testing.assert_allclose(chain.fk(q[1:])[0], self.arm.fk(q)[[0, 2]], atol=1e-12)
            self.arm.fk(q)
            self.assertAlmostEqual(self.arm.body_pitch(), self.arm.pitch(q), places=9)
            # closed-form IK (point + pitch) recovers the joints (one of the two arm branches)
            qs, ok = chain.ik(*chain.fk(q[1:])[0], self.arm.pitch(q))
            self.assertTrue(any(ok[b] and np.allclose(qs[b], q[1:], atol=1e-9) for b in range(2)))

    def test_ik_solution_reaches_target_under_fk(self):
        rng = np.random.default_rng(11)
        model = mujoco.MjModel.from_xml_path(str(MODELS / 'full4.xml'))
        data = mujoco.MjData(model)
        checked = 0
        while checked < 12:
            q_true = rng.uniform(self.arm.lo * .8, self.arm.hi * .8)
            tip = self.arm.fk(q_true)
            if tip[2] < .03:
                continue
            q, pos_res, phi_res = solve_ik(self.arm, tip[None], np.array([self.arm.pitch(q_true)]))
            self.assertTrue(np.all(q[0] >= self.arm.lo - 1e-9) and np.all(q[0] <= self.arm.hi + 1e-9))
            data.qpos[:] = q[0]
            mujoco.mj_forward(model, data)          # independent FK of the IK answer
            self.assertLess(np.linalg.norm(data.site('tooth_tip').xpos - tip), 1e-3)
            self.assertLess(abs(phi_res[0]), np.radians(0.5))
            checked += 1

    def test_retarget_keeps_path_inside_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'traj.csv'
            synthetic_trajectory(duration=4.0).to_csv(path, index=False)
            df, _ = load_trajectory(path)
        res = resample(df)
        P, R = apply_mount(res, parse_mount(None))
        tf = fit_retarget(P, self.env)
        targets = apply_transform(tf, P)
        self.assertTrue(self.env.contains(targets).all())
        self.assertTrue((targets[:, 2] > 0.01).all())
        A = np.array(tf['matrix_world_to_model'])[:3, :3] / tf['scale']
        np.testing.assert_allclose(A @ A.T, np.eye(3), atol=1e-9)       # similarity, no shear
        np.testing.assert_allclose(A[2], [0, 0, 1], atol=1e-9)           # world +z stays model +z
        # Every target is solvable exactly in 3D: the slew turns the arm plane to it.
        _, pos_res, _ = solve_ik(self.arm, targets[::4], None)
        self.assertLess(pos_res.max(), 2e-3)
        self.assertEqual(np.abs(self.env.lateral(targets)).max(), 0)
        # A pitch profile is achievable along the whole path with three pitch joints.
        theta = pitch_track(R, np.array(tf['dig_direction_world']))
        tf2 = fit_retarget(P, self.env, tf['yaw_rad'], None, 0.9, theta)
        self.assertGreater(tf2['scale'], 0)
        self.assertTrue(tf2['pitch_feasible'])
        self.assertTrue(self.env.contains(apply_transform(tf2, P)).all())
        # With slew / stick held (the hardware) the region is the boom + bucket plane, and the
        # sideways part of the path is what it drops.
        held = reachable_envelope(self.arm, locked=locked_joints())
        tf3 = fit_retarget(P, held)
        t3 = apply_transform(tf3, P)
        self.assertGreater(np.abs(held.lateral(t3)).max(), 0)
        t3[:, 1] = held.y0
        _, pos_res, _ = solve_ik(self.arm, t3[::4], None, locked=locked_joints())
        self.assertLess(pos_res.max(), 2e-3)

    def test_loader_handles_status_nan_and_time_glitches(self):
        rows = [
            [0.00, 0, 0, 0, 1, 0, 0, 0, 'tracked'],
            [0.05, .01, 0, 0, 1, 0, 0, 0, 'imu_only'],
            [0.10, np.nan, 0, 0, 1, 0, 0, 0, 'tracked'],
            [0.04, .02, 0, 0, 1, 0, 0, 0, 'tracked'],          # backwards timestamp
            [0.15, .03, 0, 0, -1, 0, 0, 0, ' Interpolated '],  # sign flip + messy status
            [0.60, .04, 0, 0, 1, 0, 0, 0, 'tracked'],          # 0.45 s gap
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'traj.csv'
            pd.DataFrame(rows, columns=['t_s', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'status']).to_csv(path, index=False)
            df, rep = load_trajectory(path)
            self.assertEqual(rep['dropped_nan_or_bad_quat'], 1)
            self.assertEqual(rep['dropped_non_monotonic'], 1)
            self.assertEqual(list(df['status']), ['tracked', 'imu_only', 'interpolated', 'tracked'])
            self.assertTrue((np.diff(df['t_s']) > 0).all())
            self.assertGreater(df['qw'].iloc[2], 0)
            res = resample(df)
            self.assertEqual(len(res.t), 13)
            self.assertEqual(res.status[1], 'imu_only')
            self.assertIn('interpolated', set(res.status[5:10]))
            pd.DataFrame(rows, columns=['t_s', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'status']).drop(
                columns='status').to_csv(path, index=False)
            df2, _ = load_trajectory(path)
            self.assertEqual(set(df2['status']), {'tracked'})

    @staticmethod
    def _scoop_along_y(reverse=False, n=120):
        """Camera looks along +y (35 deg down, small pitch/yaw wobble); the path moves mostly along
        +y (reach out and dig), or with ``reverse`` its shape is mirrored so the deepest point
        lies behind the start. Orientation is identical in both cases."""
        rng = np.random.default_rng(3)
        s = np.linspace(0, 1, n)
        fwd = 0.25 * np.sin(np.pi * s) + 0.05 * s
        P = np.c_[0.03 * np.sin(2 * np.pi * s), -fwd if reverse else fwd, -0.3 * np.sin(np.pi * s) ** 2]
        c, sn = np.cos(np.radians(35)), np.sin(np.radians(35))
        R0 = np.array([[0, -sn, c], [-1, 0, 0], [0, -c, -sn]], float)   # looks along +x, 35 deg down
        rot = (Rotation.from_euler('z', 90 + 5 * np.sin(4 * np.pi * s), degrees=True)
               * Rotation.from_rotvec(np.c_[np.zeros(n), -np.radians(30) * np.sin(np.pi * s), np.zeros(n)])
               * Rotation.from_matrix(R0) * Rotation.from_rotvec(rng.normal(0, .01, (n, 3))))
        status = np.array(['tracked'] * n, dtype=object); status[40:50] = 'imu_only'
        return P, rot, status

    def test_heading_follows_camera_not_path_shape(self):
        P, rot, status = self._scoop_along_y()
        self.assertGreater((P[np.argmin(P[:, 2])] - P[0])[1], 0)    # deepest point ahead (+y)
        h = heading_info(P, rot, status)                             # default: backhoe
        self.assertEqual(h['heading_mode'], 'backhoe')
        self.assertTrue(h['heading_source'].startswith('camera_optical_axis'))
        self.assertAlmostEqual(np.degrees(h['yaw_rad']), 90, delta=3)    # camera-forward (+y) -> model -x
        P2, rot2, status2 = self._scoop_along_y(reverse=True)
        self.assertLess((P2[np.argmin(P2[:, 2])] - P2[0])[1], 0)    # deepest point now behind the start
        h2 = heading_info(P2, rot2, status2)
        self.assertAlmostEqual(h2['yaw_rad'], h['yaw_rad'], places=9)
        cam = heading_info(P, rot, status, 'camera')
        self.assertAlmostEqual(np.degrees(cam['yaw_rad']), -90, delta=3)  # camera-forward (+y) -> model +x
        h = cam
        # PCA fallback is signed by camera forward, so it does not flip either
        hp = heading_info(P2, rot2, status2, 'pca')['heading_world']
        self.assertGreater(np.dot(hp, h['heading_world']), 0)
        back = heading_info(P, rot, status, 'backhoe')['yaw_rad']
        self.assertAlmostEqual(abs((np.degrees(back - h['yaw_rad']) + 180) % 360 - 180), 180, delta=1e-6)

    def test_retarget_pitch_stays_feasible_after_base_rotation(self):
        # Reduced pose sequence from recording 091140. This checks the
        # retargeting math, not the physical accuracy of the VIO reconstruction.
        rows = np.array([
            [2.464364, 0, 0, 0, .384466, -.548145, .673699, -.312815],
            [2.964447, .25597, .31892, .12339, .397893, -.540841, .63185, -.387219],
            [3.468544, 1.23542, .83843, .44336, .496476, -.504966, .583369, -.397745],
            [3.972584, 2.81811, 1.4502, .82437, .55366, -.507933, .526339, -.398036],
            [4.472677, 5.13155, 2.29299, 1.27672, .65091, -.398249, .434164, -.478764],
            [4.976733, 7.707, 3.76463, 2.05653, .749071, -.209203, .343653, -.526336],
            [5.480822, 10.63026, 5.19967, 3.02777, .767924, -.024868, .236192, -.594884],
            [5.98095, 14.87598, 5.8678, 3.50212, .765857, -.001029, .238785, -.59703],
        ])
        rows[:, 1:4] /= 15
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trajectory.csv'
            df = pd.DataFrame(rows, columns=['t_s', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz'])
            df['status'] = 'tracked'
            df.to_csv(path, index=False)
            m = run(path, Path(tmp) / 'out', render=False)
        self.assertTrue(m['retarget']['pitch_feasible'])
        self.assertLess(m['ik']['position_residual_m']['max'], 1e-5)
        # Previously reported feasible while missing bucket pitch by 65 degrees.
        self.assertLess(m['ik']['pitch_residual_deg']['max'], .1)

    def test_physical_curl_curls_the_twin_bucket(self):
        # Camera looks along +y (along the teeth), 35 deg down; the scoop is curled closed by
        # tilting the teeth/camera up by 20 deg (rotation about world +x takes +y toward +z).
        c, sn = np.cos(np.radians(35)), np.sin(np.radians(35))
        R0 = Rotation.from_euler('z', 90, degrees=True) * Rotation.from_matrix(
            np.array([[0, -sn, c], [-1, 0, 0], [0, -c, -sn]], float))
        rot = Rotation.concatenate([R0, Rotation.from_euler('x', 20, degrees=True) * R0])
        P = np.zeros((2, 3))
        arm = Arm()
        pivot = lambda q: (arm.fk(full_chain_q(q)), arm.data.xanchor[arm.model.joint('bucket').id].copy())
        for mode, sign in (('backhoe', -1), ('camera', 1)):
            h = heading_info(P, rot, None, mode)
            th = pitch_track(rot, np.asarray(h['heading_world']), sigma=0)
            self.assertAlmostEqual(np.degrees(th[-1]), sign * 20, delta=0.5, msg=mode)
        # backhoe: theta < 0 -> negative bucket joint -> the teeth curl toward the cab (-x).
        # Boom / bucket with the stick at its top stop (kinematics.locked_joints(), where homing
        # leaves it); the curl sense does not depend on the stick angle.
        ang = lambda v: np.degrees(np.arctan2(v[2], v[0]))            # tooth direction in the arm plane
        for b0 in (0, -20):                                             # successive curl-ups
            tip0, a0 = pivot(np.array([0, np.radians(b0)]))
            tip1, a1 = pivot(np.array([0, np.radians(b0 - 20)]))
            self.assertAlmostEqual(ang(tip1 - a1) - ang(tip0 - a0), -20, delta=1e-6)   # curl sense
            self.assertLess(tip1[0], tip0[0])                          # teeth move toward the cab (-x)
            self.assertLess((tip1 - a1)[0], (tip0 - a0)[0])            # ... and point more toward it
        tipc, ac = pivot(np.array([0, arm.lo[3]]))                     # fully curled (bucket-in stop)
        self.assertLess((tipc - ac)[0], 0)                             # teeth point toward the cab

    def test_heading_falls_back_to_image_up_when_looking_down(self):
        n = 40
        # camera straight down (z_cam = -z_world), image-up (-y_cam) along world +y
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
        rot = Rotation.from_matrix(np.repeat(R[None], n, 0)) * Rotation.from_rotvec(
            np.random.default_rng(0).normal(0, .02, (n, 3)))
        P = np.c_[np.zeros(n), np.linspace(0, .2, n), -np.sin(np.linspace(0, np.pi, n)) * .1]
        h = heading_info(P, rot, np.array(['tracked'] * n), 'camera')
        self.assertTrue(h['heading_source'].startswith('camera_image_up'))
        self.assertAlmostEqual(np.degrees(h['yaw_rad']), -90, delta=5)
        hb = heading_info(P, rot, np.array(['tracked'] * n))
        self.assertAlmostEqual(np.degrees(hb['yaw_rad']), 90, delta=5)

    def test_mount_file_is_applied(self):
        t_cb = [0.04, 0.06, -0.12]                                  # right, below, behind the camera
        q_cb = Rotation.from_euler('y', 30, degrees=True).as_quat()  # xyzw
        with tempfile.TemporaryDirectory() as tmp:
            mf = Path(tmp) / 'mount.json'
            mf.write_text(json.dumps({'description': 't', 'measured': True, 'camera_side': 'right',
                                      'camera_faces': 'backwards_toward_cab', 'bucket_point_in_camera_m': t_cb,
                                      'rotation_quat_wxyz': [q_cb[3], *q_cb[:3]], 'notes': ''}))
            vals, info = resolve_mount(None, mf)
            self.assertTrue(info['mount_source'].startswith('file'))
            self.assertTrue(info['measured'])
            path = Path(tmp) / 'traj.csv'
            synthetic_trajectory(duration=2.0).to_csv(path, index=False)
            df, _ = load_trajectory(path)
            res = resample(df)
            P, R = apply_mount(res, parse_mount(vals))
            # tooth-tip target = camera position + R_wc @ offset (differs from the camera position)
            np.testing.assert_allclose(P - res.pos, res.rot.apply(t_cb), atol=1e-12)
            self.assertGreater(np.linalg.norm(P - res.pos, axis=1).min(), 0.1)
            np.testing.assert_allclose((res.rot.inv() * R).as_quat() * np.sign((res.rot.inv() * R).as_quat()[:, 3:]),
                                       np.tile(q_cb * np.sign(q_cb[3]), (len(P), 1)), atol=1e-9)
            m = run(path, Path(tmp) / 'out', render=False, mount_file=mf)
            self.assertTrue(m['mount_source'].startswith('file'))
            np.testing.assert_allclose(m['mount']['bucket_point_in_camera_m'], t_cb)
            vals_cli, info_cli = resolve_mount([0.01, 0.02, 0.03], mf)          # CLI overrides the file
            self.assertEqual(vals_cli, [0.01, 0.02, 0.03, 1.0, 0.0, 0.0, 0.0])
            self.assertTrue(info_cli['mount_source'].startswith('cli'))

    def test_end_to_end_without_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'traj.csv'
            synthetic_trajectory(duration=3.0).to_csv(path, index=False)
            m = run(path, Path(tmp) / 'out', render=False)
            for f in ['actions.csv', 'fk_kinematic.csv', 'fk_dynamic.csv', 'metrics.json', 'joints.png']:
                self.assertTrue((Path(tmp) / 'out' / f).exists(), f)
            self.assertFalse(m['dynamic']['unstable'])
            self.assertEqual(m['ik']['unreachable_count'], 0)
            self.assertLess(m['dynamic']['tip_rmse_vs_commanded_fk_m'], 0.02)
            self.assertTrue(m['heading_source'].startswith('camera_optical_axis'))
            self.assertIn('backhoe', m['heading_source'])
            # synthetic camera looks along +x (turns 40 deg during the swing) -> model -x
            self.assertGreater(abs(m['yaw_deg']), 165)
            self.assertEqual(len(m['heading_world']), 3)
            self.assertEqual(m['joints'], ['slew', 'boom', 'stick', 'bucket'])
            self.assertEqual(m['active_joints'], NQ_NAMES)
            self.assertLess(m['ik']['pitch_residual_deg']['p95'], 0.5)      # pitch tracked, not compromised
            # the commanded joints stay within the speed limits (the synthetic scoop is far too fast for them)
            q = pd.read_csv(Path(tmp) / 'out' / 'actions.csv')[[f'q_{n}' for n in NQ_NAMES]].to_numpy()
            arm = Arm()
            self.assertLessEqual(speed_ratio(q, 0.05, arm.v_pos, arm.v_neg).max(), 1 + 1e-6)
            self.assertIn('commanded', m['bucket_angle_error_deg'])
            self.assertGreater(np.ptp(m['slew_used_deg']), 1)                 # the base turned with the swing
            json.loads((Path(tmp) / 'out' / 'metrics.json').read_text())
            # boom / bucket only (the hardware and pi05 contract): slew and stick never move
            m = run(path, Path(tmp) / 'held', render=False, hold_joints=True)
            act = pd.read_csv(Path(tmp) / 'held' / 'actions.csv')
            self.assertEqual(m['held_joints'], locked_joints())
            self.assertEqual(m['active_joints'], ['boom', 'bucket'])
            self.assertEqual(m['joint_motion_deg']['slew'], 0.)
            self.assertEqual(m['joint_motion_deg']['stick'], 0.)
            self.assertTrue(np.allclose(act['q_slew'], locked_joints()['slew']))
            self.assertTrue(np.allclose(act['q_stick'], locked_joints()['stick']))
            self.assertEqual(m['ik']['unreachable_count'], 0)

    def test_lookahead_exports_execution_and_recording_clocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'traj.csv'
            trajectory = synthetic_trajectory(duration=2.)
            trajectory['t_s'] += 7.
            trajectory.to_csv(path, index=False)
            m = run(path, Path(tmp) / 'lookahead', render=False, planner='lookahead', time_scale=1.3)
            act = pd.read_csv(Path(tmp) / 'lookahead/actions.csv')
            self.assertEqual(m['planner'], 'lookahead')
            self.assertIsNone(m['held_joints'])
            self.assertEqual(m['active_joints'], NQ_NAMES)
            q = act[[f'q_{name}' for name in NQ_NAMES]].to_numpy()
            dynamic = pd.read_csv(Path(tmp) / 'lookahead/fk_dynamic.csv')
            self.assertTrue(np.all(np.ptp(q, axis=0) > np.radians(1.)))
            for i, name in enumerate(NQ_NAMES):
                self.assertAlmostEqual(m['joint_motion_deg'][name], np.degrees(np.ptp(q[:, i])), places=3)
                self.assertGreater(np.ptp(dynamic[f'qpos_{name}']), np.radians(1.))
            self.assertTrue(m['source_time']['nonuniform'])
            self.assertAlmostEqual(act.source_t_s.iloc[0], 7.)
            self.assertTrue(np.all(np.diff(act.source_t_s) > 0))
            self.assertTrue(np.all(np.diff(act.source_t_s) <= .05 / 1.3 + 2e-6))
            self.assertLessEqual(m['timing']['continuous_velocity_limit_ratio'], 1.)
            self.assertLessEqual(m['timing']['continuous_acceleration_limit_ratio'], 1.)
            self.assertFalse(m['dynamic']['unstable'])
            self.assertLess(m['kinematic']['tip_error_vs_target_m']['p95'], .003)


if __name__ == '__main__':
    unittest.main()
