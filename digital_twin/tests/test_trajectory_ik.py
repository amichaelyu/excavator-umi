import numpy as np
import pytest

from digital_twin.sim.kinematics import Arm, full_chain_q, locked_joints, pitch_of, propagate_dynamic
from digital_twin.sim.trajectory_ik import fit_joint_path, plan_ik_trajectory, retime_joint_path


def limits(q, t, **options):
    return retime_joint_path(q, t, lower=-2., upper=2., v_pos=.5, v_neg=.8, acceleration=1., **options)


def check_sampled_limits(plan, v_pos, v_neg, acceleration):
    dt = np.diff(plan.time)
    np.testing.assert_allclose(dt, dt[0], atol=1e-12)
    velocity = np.diff(plan.q, axis=0) / dt[:, None]
    assert np.all(velocity <= np.asarray(v_pos) + 1e-7)
    assert np.all(velocity >= -np.asarray(v_neg) - 1e-7)
    # Include the initial/final rests. Averages across adjacent control ticks
    # must obey the same acceleration bounds as the continuous reference.
    acc = np.diff(np.vstack([np.zeros(plan.q.shape[1]), velocity, np.zeros(plan.q.shape[1])]), axis=0) / dt[0]
    assert np.all(np.abs(acc) <= np.asarray(acceleration) + 1e-6)
    assert plan.metrics['continuous_velocity_limit_ratio'] <= 1
    assert plan.metrics['continuous_acceleration_limit_ratio'] <= 1


def test_local_slowdown_preserves_waypoints_and_brakes_before_reversal():
    source = np.array([0., .1, 1.1, 2.1, 2.2, 3.2])
    q = np.array([[0.], [.8], [1.], [.9], [.2], [0.]])
    plan = limits(q, source)
    check_sampled_limits(plan, .5, .8, 1.)
    np.testing.assert_allclose(plan.path(source), q, atol=1e-12)
    np.testing.assert_allclose(plan.q[[0, -1]], q[[0, -1]], atol=1e-12)
    assert plan.source_time[0] == source[0] and plan.source_time[-1] == source[-1]
    assert np.all(np.diff(plan.source_time) > 0)
    assert np.all(np.diff(plan.source_time) <= np.diff(plan.time) + 1e-8)
    uniform_duration = source[-1] * max(np.max(np.diff(q[:, 0]) / np.diff(source) / .5),
                                       np.max(-np.diff(q[:, 0]) / np.diff(source) / .8))
    assert source[-1] < plan.time[-1] < .5 * uniform_duration


def test_position_limits_and_stationary_axes_remain_exact():
    q = np.array([[-1., .3], [.9, .3], [1., .3], [1., .3]])
    plan = retime_joint_path(q, [5., 5.1, 5.2, 6.], lower=-1., upper=1.,
                            v_pos=[.8, .3], v_neg=[.4, .3], acceleration=[.7, .3])
    check_sampled_limits(plan, [.8, .3], [.4, .3], [.7, .3])
    assert np.all(plan.q >= -1.) and np.all(plan.q <= 1.)
    np.testing.assert_array_equal(plan.q[:, 1], np.full(len(plan.q), .3))


def test_continuous_certificate_matches_independent_dense_derivatives():
    from scipy.interpolate import PchipInterpolator
    from digital_twin.sim.trajectory_ik import _continuous_limits
    source = np.array([0., .13, .8, 1., 1.5])
    q = np.array([[0., -.2], [.5, -.4], [-.3, .6], [-.1, .7], [0., .1]])
    curve = PchipInterpolator(source, q, axis=0)
    speed2 = np.array([0., .6, .3, .8, 0.])
    vp, vn, a = np.array([.7, .4]), np.array([.3, .8]), np.array([2., 3.])
    vr, ar = _continuous_limits(curve, source, speed2, vp, vn, a)
    dense_v, dense_a = [], []
    for i, h in enumerate(np.diff(source)):
        s = np.linspace(source[i], np.nextafter(source[i + 1], source[i]), 10001)
        x = np.interp(s, source[i:i + 2], speed2[i:i + 2])
        u = (speed2[i + 1] - speed2[i]) / (2*h)
        v = curve(s, 1) * np.sqrt(x[:, None])
        acc = curve(s, 2) * x[:, None] + curve(s, 1) * u
        dense_v.append(np.max(np.abs(v) / np.where(v >= 0, vp, vn)))
        dense_a.append(np.max(np.abs(acc) / a))
    np.testing.assert_allclose([vr, ar], [max(dense_v), max(dense_a)], rtol=1e-7)


def test_stationary_recording_and_two_waypoint_path():
    for q in (np.array([[.2], [.2]]), np.array([[.2], [-.6]])):
        plan = limits(q, [0., .3])
        check_sampled_limits(plan, .5, .8, 1.)
        np.testing.assert_allclose(plan.q[[0, -1]], q, atol=1e-12)


@pytest.mark.parametrize('q,t,options', [
    ([[0.], [np.nan]], [0., 1.], {}),
    ([[0.], [1.]], [0., 0.], {}),
    ([[0.], [3.]], [0., 1.], {}),
    ([[0.], [1.]], [0., 1.], {'dt': 0}),
])
def test_invalid_paths_are_rejected(q, t, options):
    with pytest.raises(ValueError):
        limits(q, t, **options)


def test_reachable_excavator_path_keeps_tip_and_pitch():
    arm = Arm()
    source = np.linspace(0, 1, 12)
    q = np.array([[-.1 + .2*s, -.4 + .5*s, .5 - .3*s, .7 + .1*s] for s in source])
    targets = np.array([arm.fk(v) for v in q])
    plan = plan_ik_trajectory(arm, targets, pitch_of(q), source, reference_q=q)
    got = plan.path(source)
    np.testing.assert_allclose([arm.fk(v) for v in got], targets, atol=1e-8)
    np.testing.assert_allclose(pitch_of(got), pitch_of(q), atol=1e-8)
    check_sampled_limits(plan, arm.v_pos, arm.v_neg, np.maximum(arm.v_pos, arm.v_neg) / .1)


def test_joint_interpolation_keeps_a_straight_tooth_path():
    arm = Arm()
    source = np.array([0., 1.])
    q = np.array([[-.5, -.3, .5, .7], [.5, -.3, .5, .7]])
    targets = np.array([arm.fk(v) for v in q])
    # Two exact endpoint IK solutions alone make the tooth swing in an arc.
    midpoint = arm.fk(q.mean(0))
    assert np.linalg.norm(midpoint - targets.mean(0)) > .005
    plan = plan_ik_trajectory(arm, targets, pitch_of(q), source, reference_q=q)
    dense = np.linspace(0, 1, 1001)
    wanted = targets[0] + dense[:, None] * (targets[1] - targets[0])
    actual = np.array([arm.fk(v) for v in plan.path(dense)])
    assert np.linalg.norm(actual - wanted, axis=1).max() < .0006
    assert plan.metrics['cartesian_refinement']['converged']
    assert plan.metrics['cartesian_refinement']['refined_waypoints'] > 2
    check_sampled_limits(plan, arm.v_pos, arm.v_neg, np.maximum(arm.v_pos, arm.v_neg) / .1)


@pytest.mark.parametrize('profile', ['full4', 'full4_calibrated'])
def test_default_ik_solves_base_swing_and_moves_every_joint(profile):
    arm = Arm(profile)
    source = np.linspace(0., 1., 19)
    known = np.c_[-1.2 + 2.4 * source, .06 + .2 * np.sin(np.pi * source),
                  .15 + .6 * source, .75 + .3 * np.cos(np.pi * source)]
    targets = np.array([arm.fk(q) for q in known])
    # Give IK only the bucket poses, not the known joint solution or a seed.
    plan = plan_ik_trajectory(arm, targets, pitch_of(known), source)
    solved = plan.path(source)
    np.testing.assert_allclose([arm.fk(q) for q in solved], targets, atol=1e-5)
    np.testing.assert_allclose(pitch_of(solved), pitch_of(known), atol=1e-4)
    assert np.ptp(plan.q[:, 0]) > np.radians(130.)
    assert np.all(np.ptp(plan.q, axis=0) > np.radians(10.))
    assert np.all(plan.q >= arm.lo) and np.all(plan.q <= arm.hi)
    check_sampled_limits(plan, arm.v_pos, arm.v_neg, np.maximum(arm.v_pos, arm.v_neg) / .1)
    dynamic = propagate_dynamic(arm, plan.q)
    assert not dynamic['unstable']
    assert np.all(np.ptp(dynamic['qpos'], axis=0) > np.radians(10.))


def test_held_joints_and_unreachable_pitch_do_not_change_hardware_contract():
    arm = Arm()
    q = full_chain_q(np.array([[-.3, .4], [-.2, .5], [-.1, .6]]))
    targets = np.array([arm.fk(v) for v in q])
    desired_pitch = pitch_of(q) + .2
    fitted = fit_joint_path(arm, targets, desired_pitch, reference_q=q, locked=locked_joints())
    np.testing.assert_array_equal(fitted[:, [0, 2]], q[:, [0, 2]])
    def objective(values):
        return np.sum((np.array([arm.fk(v) for v in values]) - targets)**2, axis=1) + .2**2 * (pitch_of(values) - desired_pitch)**2
    assert np.all(objective(fitted) <= objective(q))
    assert np.any(np.abs(pitch_of(fitted) - desired_pitch) > 1e-5)
