"""End-to-end model of what the real excavator can do, and a pulse-space planner for it.

The calibrated twin ``full4_calibrated`` (models/full4_calibrated.xml, configs/hardware/robot_full4_calibrated.json
from ``python -m digital_twin.sim.build --calibrated``): calibrated boom + bucket, a free stick
(measured range, anchored at its top stop) and a continuous slew. The machine is driven through six buttons
(commands.ChannelModel). On the repaired machine (hardware_map.json ``slew`` calibrated) these are
boom+/-, bucket+/-, stick+/- (23 / 22) and slew+/- (19 left / 21 right), each moving one joint. On the
old wiring they were boom+/-, bucket+/- and the coupled channels A (GPIO 27) and B (GPIO 22), each of
which yawed the base AND moved the stick; the stick stops at its limit while the yaw continues. One
button at a time.

:func:`plan_pulses` turns a desired tip trajectory (100 ms ticks) into one pressed channel per tick
with a duty in [0.3, 1] (>= 30 ms, the rest of the tick idle), chosen by a two-step receding-horizon
search over the channel model and the twin's FK. Per tick the result is a signed duty vector
``[duty_boom, duty_bucket, duty_stick, duty_slew]`` (all in [-1, 1]; old wiring: ``[duty_boom,
duty_bucket, duty_A, duty_B]`` with A/B in [0, 1]) - the v2 training action space - plus the pulse list the bridge would receive.

:func:`run_coupled` is ``digital_twin.sim.replay --planner coupled``: retarget the handheld path into the
twin's workspace, plan, and write ``actions_pulse.csv``, ``pulses.csv``, ``metrics.json`` and ``replay_kinematic.mp4``.
Approximations: kinematic (no onset latency, inertia or backlash); boom/bucket yaw leak ignored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .commands import COUPLED_JOINTS, ChannelModel

TICK_S = 0.1
DUTIES = np.round(np.arange(0.3, 1.0001, 0.1), 2)
DUTIES_2 = (0.5, 1.0)                                   # coarser second step of the search


def joint_cfg(arm) -> dict:
    """{joint: {lo, hi}} of a kinematics.Arm for the channel model; a continuous joint is unbounded."""
    lim = arm.model.jnt_limited[arm.jid].astype(bool)
    return {n: {'lo': float(arm.lo[i]) if lim[i] else -np.inf, 'hi': float(arm.hi[i]) if lim[i] else np.inf}
            for i, n in enumerate(COUPLED_JOINTS)}


def plan_pulses(targets, q0, cm: ChannelModel, arm, tick_s: float = TICK_S,
                w_next: float = 1.0, w_second: float = 1.0, yaw_cost: float = 2e-4) -> dict:
    """Greedy two-step receding-horizon channel schedule tracking ``targets`` (N, 3) at ``tick_s``.

    At tick k every (channel, duty) is simulated for one tick, followed by the best of a coarse
    second tick; the first choice minimising w_next |tip_(k+1) - T_(k+1)| + w_second |tip_(k+2) -
    T_(k+2)| (+ a small cost per unit of A/B duty, which also yaws the base) is executed.
    Returns per-tick channel/duty/q/tip and the pulse list."""
    T = np.asarray(targets, float); n = len(T)
    first = [(None, 0.0)] + [(c, d) for c in cm.rates for d in DUTIES]
    second = [(None, 0.0)] + [(c, d) for c in cm.rates for d in DUTIES_2]
    q = np.asarray(q0, float).copy()
    qs, tips, chans, duties = [q.copy()], [arm.fk(q)], [], []
    for k in range(n - 1):
        t1, t2 = T[k + 1], T[min(k + 2, n - 1)]
        best, best_cost = (None, 0.0), np.inf
        for c, d in first:
            q1 = cm.step(q, c, d * tick_s)
            e1 = np.linalg.norm(arm.fk(q1) - t1)
            e2 = min(np.linalg.norm(arm.fk(cm.step(q1, c2, d2 * tick_s)) - t2) for c2, d2 in second)
            cost = w_next * e1 + w_second * e2 + (yaw_cost * d if c in ('A', 'B', 'slew+', 'slew-') else 0.0) + 1e-6 * d
            if cost < best_cost:
                best, best_cost = (c, d), cost
        c, d = best
        q = cm.step(q, c, d * tick_s)
        chans.append(c or 'none'); duties.append(d); qs.append(q.copy()); tips.append(arm.fk(q))
    chans.append('none'); duties.append(0.0)
    pulses = [{'t_start_s': round(k * tick_s, 4), 'channel': c, 'gpio': int(cm.gpio[c]),
               'duration_ms': int(round(d * tick_s * 1000))}
              for k, (c, d) in enumerate(zip(chans, duties)) if c != 'none']
    return {'q': np.array(qs), 'tip': np.array(tips), 'channel': chans, 'duty': np.array(duties),
            'duties': pd.DataFrame([cm.duty_row(c, d) for c, d in zip(chans, duties)]), 'pulses': pulses}


def run_coupled(trajectory, output, mount=None, yaw_deg=None, scale=None, fill=0.9, time_scale=1.0,
                smooth_sigma=1.0, render=True, heading='backhoe', mount_file=None, camera='overview') -> dict:
    """digital_twin.sim.replay --planner coupled."""
    from .kinematics import Arm, CONTROL_HZ, locked_joints
    from .replay import (_stats, apply_mount, apply_transform, fit_retarget, heading_info, load_trajectory,
                                    parse_mount, pitch_track, reachable_envelope, render_video, resample, resolve_mount,
                                    solve_ik)
    from scipy.spatial.transform import Rotation
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    df, load_rep = load_trajectory(trajectory)
    df = df.copy(); df['t_s'] = (df['t_s'] - df['t_s'].iloc[0]) * time_scale
    res = resample(df, CONTROL_HZ, smooth_sigma)
    mount_vals, mount_info = resolve_mount(mount, mount_file)
    P_world, R_bucket = apply_mount(res, parse_mount(mount_vals))
    arm = Arm('full4_calibrated')
    # fit the path where boom / bucket reach with the base straight and the stick at its top stop
    # (where the machine starts); the plan then follows its sideways part by yawing (A/B)
    stick = locked_joints()['stick']
    env = reachable_envelope(arm, locked={'slew': 0.0, 'stick': stick})
    hinfo = heading_info(P_world, res.rot, res.status, heading)
    yaw = hinfo['yaw_rad'] if yaw_deg is None else float(np.radians(yaw_deg))
    dig_dir = Rotation.from_euler('z', yaw).inv().apply([1.0, 0, 0])
    theta = pitch_track(R_bucket, dig_dir, smooth_sigma)
    tf = fit_retarget(P_world, env, yaw, scale, fill, theta)
    targets20 = apply_transform(tf, P_world)
    step = max(1, int(round(TICK_S * CONTROL_HZ)))
    t = res.t[::step]; T = targets20[::step]
    # start with the base straight and the stick at its top stop, where homing leaves the machine
    q_first, _, _ = solve_ik(arm, T[:1], None, locked={'slew': 0.0, 'stick': stick})
    cm = ChannelModel.from_config(joint_cfg=joint_cfg(arm))
    q0 = q_first[0]
    plan = plan_pulses(T, q0, cm, arm)
    err = np.linalg.norm(plan['tip'] - T, axis=1)
    lateral = T[:, 1] - arm.y0
    cols = {'t_s': t, **{c: plan['duties'][c].to_numpy() for c in plan['duties']},
            **{f'q_est_{n}': plan['q'][:, i] for i, n in enumerate(COUPLED_JOINTS)},
            'tip_target_x': T[:, 0], 'tip_target_y': T[:, 1], 'tip_target_z': T[:, 2],
            'tip_achieved_x': plan['tip'][:, 0], 'tip_achieved_y': plan['tip'][:, 1], 'tip_achieved_z': plan['tip'][:, 2],
            'tip_err_m': err, 'channel': plan['channel']}
    pd.DataFrame(cols).to_csv(out / 'actions_pulse.csv', index=False, float_format='%.6f')
    pd.DataFrame(plan['pulses'], columns=['t_start_s', 'channel', 'gpio', 'duration_ms']).to_csv(out / 'pulses.csv', index=False)
    use = pd.Series(plan['channel']).value_counts().to_dict()
    metrics = {'input': str(trajectory), 'profile': 'full4_calibrated', 'planner': 'coupled', 'tick_s': TICK_S, 'ticks': int(len(t)),
               'duration_s': float(t[-1]), 'loader': load_rep, 'channel_model': cm.source,
               'channel_rates_deg_s': {k: dict(zip(COUPLED_JOINTS, np.degrees(v).round(2).tolist())) for k, v in cm.rates.items()},
               'retarget': tf, 'yaw_deg': float(np.degrees(yaw)), 'heading_source': hinfo['heading_source'],
               'lateral_kept_m': {**_stats(lateral), 'note': 'offset from the slew-0 arm plane; reached by yawing (A/B)'},
               'q0': dict(zip(COUPLED_JOINTS, map(float, q0))),
               'tip_tracking': {'rmse_m': float(np.sqrt(np.mean(err ** 2))), **{f'err_{k}_m': v for k, v in _stats(err).items()},
                                'rmse_after_1s_m': float(np.sqrt(np.mean(err[t >= 1.0] ** 2))) if (t >= 1.0).any() else None},
               'channel_ticks': {k: int(v) for k, v in use.items()}, 'n_pulses': len(plan['pulses']),
               'slew_range_deg': [float(np.degrees(plan['q'][:, 0].min())), float(np.degrees(plan['q'][:, 0].max()))],
               'stick_range_deg': [float(np.degrees(plan['q'][:, 2].min())), float(np.degrees(plan['q'][:, 2].max()))],
               'model': 'kinematic channel model; no onset latency, inertia or boom/bucket yaw leak',
               'mount_source': mount_info['mount_source']}
    if render:
        status = res.status[::step]
        render_video(arm, out / 'replay_kinematic.mp4', plan['q'], plan['tip'], T, status, np.ones(len(T), bool), t,
                     'PULSE PLAN (coupled channel model)', camera, fps=int(round(1 / TICK_S)))
    metrics['outputs'] = sorted(p.name for p in out.iterdir())
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=float))
    return metrics
