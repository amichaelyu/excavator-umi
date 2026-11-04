"""Drive the excavator twin from a start configuration to a bucket pose (optionally via waypoints).

IK (see kinematics.ik) -> smooth quintic joint-space plan whose duration respects the joint
speed limits in configs/sim/robot.json -> forward propagation in MuJoCo:

* ``kinematic``: qpos = planned action, mj_forward;
* ``dynamic``: planned action -> position actuators, 20 Hz ticks of 10 x 5 ms physics steps;
* ``pulses`` (``--mode pulses``): the plan is compiled into the sequential button-pulse schedule
  the real machine accepts (commands.py) and executed either in the twin's ternary / delayed
  command model (``--backend sim``) or through the GPIO bridge (``--backend bridge``, dry run
  unless ``--live --i-understand-open-loop``).

The twin's joints are ``[slew, boom, stick, bucket]``: the base spins to face the target and
the bucket pitch is solved independently of the position. With ``--mode pulses`` (and
``hardware=True``) only boom and bucket move - the machine has no other pins - so slew and
stick are held at their start values and targets off that arm plane are reported unreachable.

    python -m digital_twin.plan.goto_pose --pose 0.20 0.18 0.08 --pitch -30 --output /tmp/goto
    python -m digital_twin.plan.goto_pose --waypoints wp.json --output /tmp/goto --mode pulses --backend sim

Waypoint JSON: ``[{"pos": [x, y, z], "pitch": deg | "quat": [w, x, y, z] | "rotvec": [..],
"dwell_s": 0.5}, ...]``. Positions are metres in the model/world frame for the controlled
point (``--point tooth_tip`` default, or ``bucket`` body origin). Joint angles on the CLI are
degrees. Simulator-first: check a plan in the sim before using the bridge backend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from digital_twin.sim.kinematics import (CONTROL_HZ, JOINTS, Arm, IKResult, ik, parse_rotation, propagate_dynamic,
                         propagate_kinematic, rotation_angle, speed_ratio)  # noqa: F401
from digital_twin.sim.viz import add_frame, add_segment, hud, make_camera, open_writer

QUINTIC_PEAK = 1.875  # peak / mean speed of the 10-15-6 quintic


# ----------------------------------------------------------------- planning
def quintic(tau):
    tau = np.clip(tau, 0, 1)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


def segment_time(q0, q1, v_pos, v_neg, margin=1.05, dt=1 / CONTROL_HZ) -> float:
    """Slowest joint's limit-respecting quintic duration, rounded up to the control tick."""
    d = np.asarray(q1) - np.asarray(q0)
    T = margin * QUINTIC_PEAK * np.max(np.abs(d) / np.where(d >= 0, v_pos, v_neg))
    return float(max(2 * dt, np.ceil(T / dt - 1e-9) * dt))


def plan_joint_trajectory(q_start, q_goals, v_pos, v_neg, dwell_s=None, duration=None, settle_s=1.0,
                          dt=1 / CONTROL_HZ, margin=1.05, joints=None) -> dict:
    """Quintic segments q_start -> q_goals[0] -> ... with dwell after each goal and a final settle.

    ``duration``: None/'auto' = per-segment auto times; a float = total motion time, split in
    proportion to the auto times (shorter than auto => speed-limit violations, reported).
    ``joints``: names for the report (default ``kinematics.JOINTS``).
    """
    q_goals = [np.asarray(q, float) for q in q_goals]
    dwell_s = list(dwell_s or [0.0] * len(q_goals))
    starts = [np.asarray(q_start, float)] + q_goals[:-1]
    T_auto = [segment_time(a, b, v_pos, v_neg, margin, dt) for a, b in zip(starts, q_goals)]
    if duration in (None, 'auto'):
        T = T_auto
    else:
        total = float(duration)
        T = [max(2 * dt, np.round(total * ta / sum(T_auto) / dt) * dt) for ta in T_auto]
    t_list, q_list, seg_list = [np.array([0.0])], [starts[0][None]], [np.array([0])]
    t0 = 0.0
    segments = []
    for i, (a, b, Ti) in enumerate(zip(starts, q_goals, T)):
        n = int(round(Ti / dt))
        tau = np.arange(1, n + 1) / n
        q_list.append(a + (b - a) * quintic(tau)[:, None])
        t_list.append(t0 + tau * Ti); seg_list.append(np.full(n, i))
        nd = int(round((dwell_s[i] + (settle_s if i == len(q_goals) - 1 else 0)) / dt))
        if nd:
            q_list.append(np.repeat(b[None], nd, 0))
            t_list.append(t0 + Ti + np.arange(1, nd + 1) * dt); seg_list.append(np.full(nd, i))
        segments.append({'index': i, 't_start_s': t0, 't_end_s': t0 + Ti, 'auto_duration_s': T_auto[i],
                         'duration_s': Ti, 'dwell_s': dwell_s[i], 'q_start': a.tolist(), 'q_goal': b.tolist()})
        t0 += Ti + nd * dt
    Q = np.concatenate(q_list)
    ratio = speed_ratio(Q, dt, v_pos, v_neg)
    return {'t': np.concatenate(t_list), 'q': Q, 'segment': np.concatenate(seg_list), 'segments': segments,
            'motion_duration_s': float(sum(T)), 'auto_motion_duration_s': float(sum(T_auto)),
            'total_duration_s': float(np.concatenate(t_list)[-1]),
            'max_speed_ratio': {n: float(ratio[:, j].max()) for j, n in enumerate(joints or JOINTS)}}


# ------------------------------------------------------------------ targets
def normalise_target(tp) -> dict:
    """Accept {'pos', 'pitch'|'pitch_deg'|'quat'|'rotvec'|'rotmat', 'dwell_s'}."""
    t = {'pos': np.asarray(tp['pos'], float), 'dwell_s': float(tp.get('dwell_s', 0.0))}
    if 'pitch' in tp or 'pitch_deg' in tp:
        t['rot'] = ('pitch', float(tp.get('pitch', tp.get('pitch_deg'))))
    elif 'quat' in tp:
        t['rot'] = ('quat', list(map(float, tp['quat'])))
    elif 'rotvec' in tp:
        t['rot'] = ('rotvec', list(map(float, tp['rotvec'])))
    elif 'rotmat' in tp:
        t['rot'] = ('rotmat', np.asarray(tp['rotmat'], float).tolist())
    else:
        t['rot'] = ('none', None)
    return t


def _rot_kwargs(rot):
    kind, v = rot
    return {'pitch': {'pitch_deg': v}, 'quat': {'quat_wxyz': v}, 'rotvec': {'rotvec': v},
            'rotmat': {'rotmat': v}, 'none': {}}[kind]


def requested_rotation(arm: Arm, target: dict, res: IKResult) -> np.ndarray:
    """Full requested bucket rotation; for a pitch-only request it is the rotation the IK
    was asked for, in the arm plane at the solved slew (fixed plane for arm2)."""
    kind, val = parse_rotation(**_rot_kwargs(target['rot']))
    if kind == 'matrix':
        return val
    phi = np.radians(res.pitch_target_deg) if res.pitch_target_deg is not None else arm.pitch(res.q)
    return arm.rotation_for(phi, float(arm.slew(res.q)))


# ------------------------------------------------------------------ driving
def drive_to_pose(targets, q_start=None, *, start_pose=None, profile='full4', point='tooth_tip',
                  duration=None, settle_s=1.0, hardware=False, pulse_backend=None, pulse_options=None,
                  arm: Arm | None = None) -> dict:
    """Plan and propagate a drive through ``targets`` (list of target dicts, see module doc).

    ``q_start``: start joints [rad] (default rest pose); or ``start_pose`` = target dict solved by IK.
    ``hardware``: hold joints the real machine cannot move (per configs/hardware_map.json)
    fixed at their start values, as the pulse compiler requires. Slew and stick are already
    welded in the model, so with the shipped map nothing is held.
    ``pulse_backend``: None, or an object with ``execute(schedule)`` (commands.SimPulseBackend /
    BridgePulseBackend); the plan is compiled into pulses and executed.
    Returns a dict with plan, IK results, kinematic / dynamic (/ pulses) traces and metrics.
    """
    arm = arm or Arm(profile, point=point)
    joints = arm.joints
    hardware = hardware or pulse_backend is not None
    targets = [normalise_target(t) for t in targets]
    warnings = []
    if start_pose is not None:
        sp = normalise_target(start_pose)
        r0 = ik(sp['pos'], arm=arm, **_rot_kwargs(sp['rot']))
        if not r0.reachable:
            warnings.append('start pose not reachable; using nearest solution')
        q_start = r0.q
    q_start = arm.q_rest.copy() if q_start is None else np.asarray(q_start, float)
    locked = None
    if hardware:
        from ..hardware.commands import load_hardware_map, held_joints
        locked = {n: float(q_start[joints.index(n)]) for n in held_joints(load_hardware_map())}

    results, q_prev = [], q_start
    for i, tg in enumerate(targets):
        r = ik(tg['pos'], q0=q_prev, arm=arm, locked=locked, **_rot_kwargs(tg['rot']))
        if locked:
            free = ik(tg['pos'], q0=q_prev, arm=arm, **_rot_kwargs(tg['rot']))
            need = {n: float(np.degrees(free.q[joints.index(n)] - q_start[joints.index(n)])) for n in locked}
            if not r.reachable:
                warnings.append(f'target {i}: not reachable with {sorted(locked)} held; unconstrained IK would '
                                f'move them by {", ".join(f"{k} {v:+.1f} deg" for k, v in need.items())}')
        if not r.reachable:
            warnings.append(f'target {i}: unreachable, residual {1000 * r.position_residual_m:.1f} mm'
                            + (f', unreachable sideways {1000 * r.lateral_error_m:+.1f} mm' if r.lateral_error_m else '')
                            + (f', pitch {r.pitch_residual_deg:+.1f} deg' if r.pitch_residual_deg is not None else ''))
        results.append(r); q_prev = r.q

    plan = plan_joint_trajectory(q_start, [r.q for r in results], arm.v_pos, arm.v_neg,
                                 [t['dwell_s'] for t in targets], duration, settle_s, joints=joints)
    t, Q = plan['t'], plan['q']
    R_req = [requested_rotation(arm, tg, r) for tg, r in zip(targets, results)]
    tgt_pos = np.array([tg['pos'] for tg in targets])

    tip_kin, pitch_kin, _ = propagate_kinematic(arm, Q)
    rot_kin = []
    for q in Q:
        arm.fk(q); rot_kin.append(arm.bucket_rot())
    rot_kin = np.array(rot_kin)
    dyn = propagate_dynamic(arm, Q)
    out = {'arm': arm, 'targets': targets, 'ik': results, 'plan': plan, 'R_req': R_req, 'target_pos': tgt_pos,
           'kinematic': {'t': t, 'q': Q, 'tip': tip_kin, 'pitch': pitch_kin, 'rot': rot_kin},
           'dynamic': {**dyn, 't': t}, 'warnings': warnings, 'q_start': q_start, 'locked': locked}

    if pulse_backend is not None:
        from ..hardware.commands import compile_schedule
        sched = compile_schedule(t, Q, arm, windows=[s['t_end_s'] for s in plan['segments']],
                                 **(pulse_options or {}))
        out['schedule'] = sched
        warnings.extend(sched['warnings'])
        out['pulses'] = pulse_backend.execute(sched)
    out['metrics'] = _metrics(out)
    return out


def _final_errors(arm, tip, rot, q, target_pos, R_req, res):
    return {'tip_error_m': float(np.linalg.norm(tip - target_pos)),
            'tip_error_vs_ik_m': float(np.linalg.norm(tip - res.achieved_pos)),
            'rotation_error_deg': float(np.degrees(rotation_angle(rot, R_req))),
            'pitch_error_deg': None if res.pitch_target_deg is None else float(np.degrees(arm.pitch(q)) - res.pitch_target_deg),
            'joint_error_deg': dict(zip(arm.joints, np.degrees(q - res.q).tolist()))}


def _metrics(o) -> dict:
    arm, plan, kin, dyn, res = o['arm'], o['plan'], o['kinematic'], o['dynamic'], o['ik']
    e_q = dyn['qpos'] - kin['q']
    e_tip = np.linalg.norm(dyn['tip'] - kin['tip'], axis=1)
    m = {'profile': arm.profile, 'joints': arm.joints, 'point': arm.point,
         'q_start_deg': np.degrees(o['q_start']).tolist(),
         'hardware_locked_joints': o['locked'],
         'targets': [{'pos': tg['pos'].tolist(), 'rotation': tg['rot'][0], 'rotation_value': tg['rot'][1],
                      'dwell_s': tg['dwell_s'], 'requested_rot': R.tolist(), 'ik': r.as_dict()}
                     for tg, r, R in zip(o['targets'], res, o['R_req'])],
         'all_reachable': bool(all(r.reachable for r in res)),
         'plan': {k: plan[k] for k in ('segments', 'motion_duration_s', 'auto_motion_duration_s',
                                       'total_duration_s', 'max_speed_ratio')},
         'speed_limits_respected': bool(max(plan['max_speed_ratio'].values()) <= 1 + 1e-9),
         'kinematic_final': _final_errors(arm, kin['tip'][-1], kin['rot'][-1], kin['q'][-1],
                                          o['target_pos'][-1], o['R_req'][-1], res[-1]),
         'dynamic_final': _final_errors(arm, dyn['tip'][-1], dyn['rot'][-1], dyn['qpos'][-1],
                                        o['target_pos'][-1], o['R_req'][-1], res[-1]),
         'dynamic_tracking': {'tip_rmse_m': float(np.sqrt(np.mean(e_tip ** 2))), 'tip_max_m': float(e_tip.max()),
                              'joint_rmse_deg': {n: float(np.degrees(np.sqrt(np.mean(e_q[:, j] ** 2))))
                                                 for j, n in enumerate(arm.joints)},
                              'unstable': bool(dyn['unstable']), 'ticks_with_contacts': int((dyn['ncon'] > 0).sum())},
         'warnings': o['warnings']}
    if 'pulses' in o:
        p = o['pulses']
        m['pulses'] = {'n_pulses': len(o['schedule']['pulses']), 'schedule_duration_s': o['schedule']['duration_s'],
                       'held_joints': o['schedule']['held_joints'], 'backend': p.get('backend'),
                       'dry_run': p.get('dry_run'), 'sent': p.get('sent')}
        if 'qpos' in p:
            m['pulses']['final'] = _final_errors(arm, p['tip'][-1], p['rot'][-1], p['qpos'][-1],
                                                 o['target_pos'][-1], o['R_req'][-1], res[-1])
    return m


# ------------------------------------------------------------------- output
def write_outputs(o: dict, output, render=True, camera='overview_close'):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    kin, dyn, plan, arm = o['kinematic'], o['dynamic'], o['plan'], o['arm']
    joints = arm.joints
    t, Q = kin['t'], kin['q']
    seg = plan['segment']
    tp = o['target_pos'][seg]
    pd.DataFrame({'t_s': t, 'segment': seg, **{f'q_{n}': Q[:, j] for j, n in enumerate(joints)},
                  'target_x': tp[:, 0], 'target_y': tp[:, 1], 'target_z': tp[:, 2]}
                 ).to_csv(out / 'actions.csv', index=False, float_format='%.6f')
    pd.DataFrame({'t_s': t, **{f'q_{n}': Q[:, j] for j, n in enumerate(joints)},
                  'tip_x': kin['tip'][:, 0], 'tip_y': kin['tip'][:, 1], 'tip_z': kin['tip'][:, 2],
                  'bucket_pitch': kin['pitch'],
                  **{f'bucket_q{c}': Rotation.from_matrix(kin['rot']).as_quat()[:, i] for i, c in enumerate('xyzw')}}
                 ).to_csv(out / 'fk_kinematic.csv', index=False, float_format='%.6f')
    pd.DataFrame({'t_s': t, **{f'ctrl_{n}': dyn['ctrl'][:, j] for j, n in enumerate(joints)},
                  **{f'qpos_{n}': dyn['qpos'][:, j] for j, n in enumerate(joints)},
                  **{f'qvel_{n}': dyn['qvel'][:, j] for j, n in enumerate(joints)},
                  'tip_x': dyn['tip'][:, 0], 'tip_y': dyn['tip'][:, 1], 'tip_z': dyn['tip'][:, 2],
                  'ncon': dyn['ncon']}).to_csv(out / 'fk_dynamic.csv', index=False, float_format='%.6f')
    if 'schedule' in o:
        from ..hardware.commands import write_schedule
        write_schedule(o['schedule'], out)
        p = o['pulses']
        if 'qpos' in p:
            pd.DataFrame({'t_s': p['t'], **{f'cmd_{n}': p['command'][:, j] for j, n in enumerate(joints)},
                          **{f'target_{n}': p['target_q'][:, j] for j, n in enumerate(joints)},
                          **{f'qpos_{n}': p['qpos'][:, j] for j, n in enumerate(joints)},
                          'tip_x': p['tip'][:, 0], 'tip_y': p['tip'][:, 1], 'tip_z': p['tip'][:, 2]}
                         ).to_csv(out / 'fk_pulses.csv', index=False, float_format='%.6f')
        (out / 'bridge_log.json').write_text(json.dumps(p.get('log', []), indent=1)) if 'log' in p else None
    plot_drive(out / 'joints.png', o)
    if render:
        render_drive(o, out / 'replay.mp4', dyn['qpos'], dyn['tip'], t, 'DYNAMIC (position actuators)', camera)
        if 'pulses' in o and 'qpos' in o['pulses']:
            p = o['pulses']
            render_drive(o, out / 'replay_pulses.mp4', p['qpos'], p['tip'], p['t'], 'PULSES (sim of hardware cmds)',
                         camera)
    (out / 'metrics.json').write_text(json.dumps(o['metrics'], indent=2, default=_json_default))
    return out


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    raise TypeError(type(x))


def render_drive(o, path, qpos, tip, t, title, camera='overview_close', width=640, height=480, fps=20):
    arm = o['arm']; m, d = arm.model, arm.data
    tgt, R_req, plan_t = o['target_pos'], o['R_req'], o['plan']['t']
    renderer = mujoco.Renderer(m, height=height, width=width)
    centre = np.r_[0.5 * (tgt.min(0) + tgt.max(0))[:2] * 0.5, 0.15]
    cam = make_camera(m, camera, centre)
    writer = open_writer(path, fps)
    try:
        for k in range(len(qpos)):
            d.qpos[arm.qadr] = qpos[k]; d.qvel[:] = 0
            ks = min(int(np.searchsorted(plan_t, t[k] - 1e-9)), len(plan_t) - 1)
            active = int(o['plan']['segment'][ks])
            d.mocap_pos[0] = tgt[active]
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam)
            sc = renderer.scene
            for i in range(len(tgt)):
                add_frame(sc, tgt[i], R_req[i], 0.04 if i == active else 0.025, 0.002 if i == active else 0.0012,
                          1.0 if i == active else 0.45)
            add_frame(sc, arm.point_pos(), arm.bucket_rot(), 0.025, 0.0012, 0.9)
            for j in range(max(0, k - 80), k):
                add_segment(sc, tip[j], tip[j + 1], (1, .6, .1, 1), 0.0014)
            frame = np.ascontiguousarray(renderer.render())
            err = 1000 * np.linalg.norm(tip[k] - tgt[active])
            rerr = np.degrees(rotation_angle(arm.bucket_rot(), R_req[active]))
            lines = [title, f't = {t[k]:5.2f} s   target {active + 1}/{len(tgt)}'
                     + ('' if o['ik'][active].reachable else '  UNREACHABLE (nearest)'),
                     f'tip error: {err:5.1f} mm   orientation error: {rerr:5.1f} deg',
                     'axes: target pose (big) / bucket (small)   orange: tip trail']
            writer.append_data(hud(frame, lines))
    finally:
        writer.close(); renderer.close()


def plot_drive(path, o):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kin, dyn = o['kinematic'], o['dynamic']
    t = kin['t']; p = o.get('pulses', {})
    joints = o['arm'].joints
    rows = max(len(joints), 4)
    fig, ax = plt.subplots(rows, 2, figsize=(13, 10), sharex=True)
    for j in range(len(joints), rows):
        ax[j, 0].axis('off')
    for j, n in enumerate(joints):
        a = ax[j, 0]
        a.plot(t, np.degrees(kin['q'][:, j]), color='#1f5fbf', lw=1.6, label='planned action')
        a.plot(t, np.degrees(dyn['qpos'][:, j]), color='#e07b00', lw=1.3, ls='--', label='dynamic')
        if 'qpos' in p:
            a.plot(p['t'], np.degrees(p['qpos'][:, j]), color='#7a3fbf', lw=1.2, ls=':', label='pulses (sim)')
        for r in o['ik']:
            a.axhline(np.degrees(r.q[j]), color='#2a9d3a', lw=.7, alpha=.5)
        a.set_ylabel(f'{n} [deg]'); a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=8); ax[0, 0].set_title('Joint angles (green lines: IK goals)')
    for i, c in enumerate('xyz'):
        a = ax[i, 1]
        a.plot(t, 1000 * kin['tip'][:, i], color='#1f5fbf', lw=1.4, label='kinematic')
        a.plot(t, 1000 * dyn['tip'][:, i], color='#e07b00', lw=1.2, ls='--', label='dynamic')
        if 'qpos' in p:
            a.plot(p['t'], 1000 * p['tip'][:, i], color='#7a3fbf', lw=1.2, ls=':', label='pulses (sim)')
        for tp in o['target_pos']:
            a.axhline(1000 * tp[i], color='#2a9d3a', lw=.7, alpha=.5)
        a.set_ylabel(f'{o["arm"].point} {c} [mm]'); a.grid(alpha=.3)
    ax[0, 1].legend(fontsize=8); ax[0, 1].set_title('Controlled point (green lines: targets)')
    a = ax[3, 1]
    seg = o['plan']['segment']
    a.plot(t, 1000 * np.linalg.norm(kin['tip'] - o['target_pos'][seg], axis=1), color='#1f5fbf', label='kinematic')
    a.plot(t, 1000 * np.linalg.norm(dyn['tip'] - o['target_pos'][seg], axis=1), color='#e07b00', ls='--', label='dynamic')
    a.set_ylabel('dist. to active target [mm]'); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[len(joints) - 1, 0].set_xlabel('t [s]'); ax[3, 1].set_xlabel('t [s]')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pose', type=float, nargs=3, metavar=('X', 'Y', 'Z'), help='final point position [m]')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--pitch', type=float, help='bucket pitch [deg], nose-up +, 0 = arm2 rest')
    g.add_argument('--quat', type=float, nargs=4, metavar=('W', 'X', 'Y', 'Z'), help='full bucket orientation')
    g.add_argument('--rotvec', type=float, nargs=3, metavar=('RX', 'RY', 'RZ'), help='bucket orientation [rad]')
    p.add_argument('--dwell', type=float, default=0.0, help='dwell at the final pose [s]')
    p.add_argument('--from-q', type=float, nargs=4, metavar=('SLEW', 'BOOM', 'STICK', 'BUCKET'),
                   help='start joints [deg] (default: rest pose, stick at its top stop)')
    p.add_argument('--from-tip', type=float, nargs=3, metavar=('X', 'Y', 'Z'), help='start point position (IK)')
    p.add_argument('--from-pitch', type=float, help='start bucket pitch [deg] with --from-tip')
    p.add_argument('--waypoints', type=str, help='JSON list of {pos, pitch|quat|rotvec, dwell_s} before --pose')
    p.add_argument('--output', required=True)
    p.add_argument('--profile', default='full4', choices=['full4', 'full4_calibrated'])
    p.add_argument('--point', default='tooth_tip', choices=['tooth_tip', 'bucket'])
    p.add_argument('--duration', default='auto', help="total motion time [s] or 'auto' (speed-limit respecting)")
    p.add_argument('--settle', type=float, default=1.0, help='hold time after the last pose [s]')
    p.add_argument('--mode', default='both', choices=['both', 'pulses'],
                   help="'both' = kinematic + dynamic; 'pulses' also compiles/executes a hardware pulse schedule")
    p.add_argument('--backend', default='sim', choices=['sim', 'bridge'])
    p.add_argument('--bridge-host', default='localhost:8888', help='GPIO bridge: the hub host:port')
    p.add_argument('--dry-run', dest='live', action='store_false', help='bridge: log only (default)')
    p.add_argument('--live', dest='live', action='store_true', help='bridge: actually send pulses')
    p.add_argument('--i-understand-open-loop', action='store_true',
                   help='required with --live: no joint feedback, pulses are timed open loop')
    p.add_argument('--record', action='store_true', help='bridge: wrap the run in record_start/record_stop')
    p.add_argument('--min-pulse-ms', type=int, default=50)
    p.add_argument('--gap-ms', type=int, default=150)
    p.add_argument('--camera', default='overview_close')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--view', action='store_true', help='play the result in the interactive MuJoCo viewer (mjpython on macOS)')
    p.add_argument('--view-mode', default='dynamic', choices=['kinematic', 'dynamic', 'pulses'])
    p.add_argument('--loop', action='store_true', help='repeat the viewer playback until the window closes')
    p.add_argument('--time-scale', type=float, default=1.0, help='viewer playback slow-down factor (>1 slower)')
    p.add_argument('--view-seconds', type=float, help=argparse.SUPPRESS)   # auto-close (smoke tests)
    p.set_defaults(live=False)
    a = p.parse_args(argv)
    if a.view:
        from digital_twin.sim.viewer import guard
        if not guard('digital_twin.plan.goto_pose', argv):
            raise SystemExit(2)
        if a.view_mode == 'pulses' and a.mode != 'pulses':
            p.error('--view-mode pulses needs --mode pulses')

    targets = json.loads(Path(a.waypoints).read_text()) if a.waypoints else []
    if a.pose:
        tg = {'pos': a.pose, 'dwell_s': a.dwell}
        if a.pitch is not None: tg['pitch'] = a.pitch
        if a.quat is not None: tg['quat'] = a.quat
        if a.rotvec is not None: tg['rotvec'] = a.rotvec
        targets.append(tg)
    if not targets:
        p.error('give --pose and/or --waypoints')
    start_pose = None
    if a.from_tip is not None:
        start_pose = {'pos': a.from_tip, **({'pitch': a.from_pitch} if a.from_pitch is not None else {})}
    q_start = None if a.from_q is None else np.radians(a.from_q)
    backend = None
    hardware = a.mode == 'pulses'
    if hardware:
        from ..hardware.commands import BridgePulseBackend, SimPulseBackend
        if a.backend == 'sim':
            backend = SimPulseBackend(Arm(a.profile, point=a.point))
        else:
            backend = BridgePulseBackend(a.bridge_host, dry_run=not a.live,
                                         i_understand_open_loop=a.i_understand_open_loop, record=a.record)
    o = drive_to_pose(targets, q_start, start_pose=start_pose, profile=a.profile, point=a.point,
                      duration=a.duration, settle_s=a.settle, hardware=hardware, pulse_backend=backend,
                      pulse_options={'min_pulse_ms': a.min_pulse_ms, 'gap_ms': a.gap_ms})
    write_outputs(o, a.output, not a.no_render, a.camera)
    m = o['metrics']
    print(json.dumps({'all_reachable': m['all_reachable'], 'motion_duration_s': m['plan']['motion_duration_s'],
                      'kinematic_final': m['kinematic_final'], 'dynamic_final': m['dynamic_final'],
                      'pulses': m.get('pulses'), 'warnings': m['warnings']}, indent=2, default=_json_default))
    print(f'outputs in {a.output}')
    if a.view:
        view_drive(o, a.view_mode, loop=a.loop, time_scale=a.time_scale, max_seconds=a.view_seconds)


def view_drive(o, mode='dynamic', loop=False, time_scale=1.0, max_seconds=None):
    """Play a drive_to_pose result in the interactive viewer (needs mjpython on macOS)."""
    from digital_twin.sim.viewer import play
    arm, plan = o['arm'], o['plan']
    if mode == 'kinematic':
        t, q, tip = o['kinematic']['t'], o['kinematic']['q'], o['kinematic']['tip']
    elif mode == 'dynamic':
        t, q, tip = o['dynamic']['t'], o['dynamic']['qpos'], o['dynamic']['tip']
    else:
        p = o['pulses']
        t, q, tip = p['t'], p['qpos'], p['tip']
    ks = np.clip(np.searchsorted(plan['t'], np.asarray(t) - 1e-9), 0, len(plan['t']) - 1)
    active = plan['segment'][ks]
    play(arm.model, arm.qadr, t, q, target_frames=list(zip(o['target_pos'], o['R_req'])), active_target=active,
         trail=tip, mocap=o['target_pos'][active], time_scale=time_scale, loop=loop, max_seconds=max_seconds,
         title=f'goto_pose ({mode})')


if __name__ == '__main__':
    main()
