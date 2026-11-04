"""Compile planned joint trajectories into hardware button-pulse schedules and execute them.

The real excavator only accepts timed button pulses through the GPIO bridge
(hardware/rc-control/src/main.cpp): ``{"cmd": "trigger", "pin": <gpio>, "ms": N}``, one press at a time, with
no joint feedback. This module turns a joint trajectory ``(t, Q)`` into a sequential list of
pulses ``{t_start_s, joint, direction, duration_ms, gpio}`` with
``duration = |dq| / calibrated speed`` (configs/hardware/robot.json), and runs it on

* :class:`SimPulseBackend`  - the MuJoCo twin with env.py's command model (ternary motor
  command, start latency, velocity-limited position target, finite-torque actuator);
* :class:`BridgePulseBackend` - the GPIO bridge (hardware.gpio: in-process, or the hub's websocket). Dry run by default; live sending needs
  ``dry_run=False`` *and* ``i_understand_open_loop=True`` (CLI: ``--live --i-understand-open-loop``).

All four joints can have pins (hardware_map.json ``joints.<j>.positive`` / ``negative``, digital_twin.sim.pins; a
joint is available unless its ``status`` says otherwise). Plans are twin trajectories
``[slew, boom, stick, bucket]`` (or hardware-only ``[boom, bucket]`` columns). By default the
compiler emits only boom and bucket (``kinematics.HARDWARE_JOINTS``) and holds slew / stick at their start
value with a warning when the plan moves them; ``compile_schedule(..., joints=JOINTS)`` emits every available
joint (the 4-joint policy runner). Track drive pins are never mapped to a joint.

    # an IK label replay from the labels stage (python -m policy labels configs/policy/<config>.yaml)
    python -m digital_twin.hardware.commands --actions dataset/labels/<config>/<recording>/actions.csv --output DIR

The compiler, the hardware map helpers, the channel model and the simulator backend live in
digital_twin.sim.commands (offline planning and label generation need them); they are re-exported here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from digital_twin.sim.commands import (
    CHANNELS, COUPLED_JOINTS, DECOUPLED_CHANNELS, DRIVE_WORDS, DUTY_COLUMNS,
    PLACEHOLDER_COUPLING, ChannelModel, SimPulseBackend, _check_pins, compile_schedule, emit_joints,
    held_joints, load_hardware_map, validate_hardware_map, write_schedule)
from digital_twin.sim.kinematics import HARDWARE_JOINTS, JOINTS, NQ, Arm, full_chain_q
from digital_twin.sim.pins import pin_labels

from ..paths import HARDWARE_MAP

__all__ = ['CHANNELS', 'COUPLED_JOINTS', 'DECOUPLED_CHANNELS', 'DRIVE_WORDS', 'DUTY_COLUMNS', 'HARDWARE_MAP',
           'PLACEHOLDER_COUPLING', 'BridgePulseBackend', 'ChannelModel', 'SimPulseBackend', 'compile_schedule',
           'emit_joints', 'held_joints', 'load_hardware_map', 'main', 'pin_labels', 'validate_hardware_map',
           'write_schedule']


# -------------------------------------------------------------- bridge backend
class BridgePulseBackend:
    """Send a pulse schedule to the GPIO bridge (hardware.gpio) - open loop.

    ``bridge``: a :class:`gpio.GpioBridge` (in-process) or a hub ``'host:port'``. ``dry_run=True``
    (default) only logs the schedule and opens no connection. Live mode needs ``dry_run=False`` and
    ``i_understand_open_loop=True``. Live: open a bridge session; wait for ``bridge.esp32 == true``;
    send one ``trigger`` at a time, matched to its ``ack``/``error`` by id, then
    wait out the pulse + gap; always ``release all`` at the end.
    """

    def __init__(self, bridge='localhost:8888', dry_run: bool = True, i_understand_open_loop: bool = False,
                 connect_timeout: float = 10.0, ack_timeout: float = 6.0, log=print):
        self.bridge, self.dry_run, self.ok_open_loop = bridge, dry_run, i_understand_open_loop
        self.host = bridge if isinstance(bridge, str) else None
        self.connect_timeout, self.ack_timeout, self.log = connect_timeout, ack_timeout, log

    def execute(self, schedule: dict) -> dict:
        _check_pins(schedule)
        lines = [f"{p['t_start_s']:7.3f}s  trigger GPIO {p['gpio']:>2} ({p['joint']} {p['direction']:+d}, "
                 f"{p['label']}) {p['duration_ms']} ms" for p in schedule['pulses']]
        if self.dry_run:
            from . import gpio
            self.log(f'[dry run] {len(lines)} pulses for {gpio.describe(self.bridge)} (nothing sent):')
            for s in lines:
                self.log('  ' + s)
            return {'backend': 'bridge', 'dry_run': True, 'sent': 0, 'log': lines}
        if not self.ok_open_loop:
            raise PermissionError('live bridge execution needs i_understand_open_loop=True '
                                  '(CLI: --live --i-understand-open-loop)')
        return self._run(schedule)

    def _run(self, schedule: dict) -> dict:
        from . import gpio
        log, sent = [], []
        ids = itertools.count(1)
        waiting: dict = {}
        lock = threading.Lock()
        esp32_up = threading.Event()

        with gpio.connect(self.bridge, timeout=self.connect_timeout) as session:
            def reader():
                while True:
                    msg = session.recv(0.2)
                    if msg is None:
                        if session.closed:
                            esp32_up.clear()
                            return
                        continue
                    log.append({'rx': msg, 't': time.time()})
                    typ = msg.get('type')
                    if typ == 'bridge':
                        (esp32_up.set if msg.get('esp32') else esp32_up.clear)()
                    elif typ in ('ack', 'error'):
                        with lock:
                            slot = waiting.pop(msg.get('id'), None)
                        if slot is not None:
                            slot['reply'] = msg; slot['event'].set()

            def send(obj):
                i = next(ids)
                obj = {**obj, 'id': i}
                slot = {'event': threading.Event(), 'reply': None}
                with lock:
                    waiting[i] = slot
                session.send(obj); sent.append(obj); log.append({'tx': obj, 't': time.time()})
                if not slot['event'].wait(self.ack_timeout):
                    with lock:
                        waiting.pop(i, None)
                    raise TimeoutError(f'no reply from the bridge within {self.ack_timeout} s for {obj}')
                if slot['reply'].get('type') == 'error':
                    raise RuntimeError(f"bridge error for {obj}: {slot['reply'].get('error')}")
                return slot['reply']

            threading.Thread(target=reader, daemon=True, name='pulse-reader').start()
            try:
                if not esp32_up.wait(self.connect_timeout):
                    raise TimeoutError('the bridge reports the ESP32 offline')
                t0 = time.monotonic()
                gap = schedule.get('gap_ms', 0) / 1000
                for p in schedule['pulses']:
                    delay = t0 + p['t_start_s'] - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    if not esp32_up.is_set():
                        raise RuntimeError('ESP32 went offline; aborting')
                    send({'cmd': 'trigger', 'pin': int(p['gpio']), 'ms': int(p['duration_ms'])})
                    time.sleep(p['duration_ms'] / 1000 + gap)
            finally:
                try:
                    send({'cmd': 'release', 'pin': 'all'})
                except Exception as e:
                    log.append({'release_error': str(e)})
        return {'backend': 'bridge', 'dry_run': False, 'sent': sum(1 for s in sent if s['cmd'] == 'trigger'),
                'frames': sent, 'log': log}


# ----------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--actions', required=True, help='actions.csv with t_s and q_slew, q_boom, q_stick, q_bucket '
                                                     '(or only q_boom, q_bucket; rad)')
    p.add_argument('--output', required=True)
    p.add_argument('--chunk-s', type=float, default=0.5, help='compile window length [s]')
    p.add_argument('--min-pulse-ms', type=int, default=None)
    p.add_argument('--gap-ms', type=int, default=None)
    p.add_argument('--latency-s', type=float, default=0.2)
    p.add_argument('--backend', default='sim', choices=['sim', 'bridge'])
    p.add_argument('--bridge-host', default='localhost:8888', help='GPIO bridge: the hub host:port')
    p.add_argument('--live', action='store_true')
    p.add_argument('--i-understand-open-loop', action='store_true')
    p.add_argument('--render', action='store_true')
    a = p.parse_args(argv)
    df = pd.read_csv(a.actions)
    t = df['t_s'].to_numpy(float)
    if all(f'q_{n}' in df for n in JOINTS):
        Q = df[[f'q_{n}' for n in JOINTS]].to_numpy(float)
    else:
        Q = full_chain_q(df[[f'q_{n}' for n in HARDWARE_JOINTS]].to_numpy(float))
    arm = Arm()
    sched = compile_schedule(t, Q, arm, chunk_s=a.chunk_s, min_pulse_ms=a.min_pulse_ms, gap_ms=a.gap_ms)
    out = Path(a.output); write_schedule(sched, out)
    for w in sched['warnings']:
        print('WARNING:', w)
    metrics = {'actions': a.actions, 'n_pulses': len(sched['pulses']), 'schedule_duration_s': sched['duration_s'],
               'plan_duration_s': sched['plan_duration_s'], 'held_joints': sched['held_joints'],
               'compile_residual_deg': sched['residual_deg'], 'warnings': sched['warnings']}
    if a.backend == 'bridge':
        res = BridgePulseBackend(a.bridge_host, dry_run=not a.live, i_understand_open_loop=a.i_understand_open_loop
                                 ).execute(sched)
        metrics['bridge'] = {k: res[k] for k in ('dry_run', 'sent')}
    else:
        res = SimPulseBackend(arm, latency_s=a.latency_s).execute(sched)
        # what hardware can achieve: the plan with held joints frozen
        Qh = Q.copy()
        for n in sched['held_joints']:
            Qh[:, JOINTS.index(n)] = Q[0, JOINTS.index(n)]
        arm.fk(Qh[-1]); tip_goal = arm.point_pos()
        arm.fk(Q[-1]); tip_full = arm.point_pos()
        metrics['sim'] = {
            'final_qpos_deg': dict(zip(JOINTS, np.degrees(res['qpos'][-1]).tolist())),
            'final_joint_error_vs_plan_deg': dict(zip(JOINTS, np.degrees(res['qpos'][-1] - Q[-1]).tolist())),
            'final_tip_error_vs_held_plan_m': float(np.linalg.norm(res['tip'][-1] - tip_goal)),
            'final_tip_error_vs_full_plan_m': float(np.linalg.norm(res['tip'][-1] - tip_full)),
            'executed_duration_s': float(res['t'][-1])}
        pd.DataFrame({'t_s': res['t'], **{f'cmd_{n}': res['command'][:, j] for j, n in enumerate(JOINTS)},
                      **{f'target_{n}': res['target_q'][:, j] for j, n in enumerate(JOINTS)},
                      **{f'qpos_{n}': res['qpos'][:, j] for j, n in enumerate(JOINTS)},
                      'tip_x': res['tip'][:, 0], 'tip_y': res['tip'][:, 1], 'tip_z': res['tip'][:, 2]}
                     ).to_csv(out / 'fk_pulses.csv', index=False, float_format='%.6f')
        _plot(out / 'pulses.png', t, Q, res, sched)
        if a.render:
            from digital_twin.sim.viz import add_segment, hud, make_camera, open_writer
            m, d = arm.model, arm.data
            r = mujoco.Renderer(m, 480, 640); cam = make_camera(m); w = open_writer(out / 'replay_pulses.mp4')
            try:
                for k in range(len(res['t'])):
                    d.qpos[arm.qadr] = res['qpos'][k]; mujoco.mj_forward(m, d); r.update_scene(d, camera=cam)
                    for j in range(max(0, k - 80), k):
                        add_segment(r.scene, res['tip'][j], res['tip'][j + 1], (.6, .3, 1, 1), .0014)
                    act = [f"{n}{'+' if c > 0 else '-'}" for n, c in zip(JOINTS, res['command'][k]) if c]
                    w.append_data(hud(np.ascontiguousarray(r.render()),
                                      ['PULSE SCHEDULE (sim of hardware cmds)', f"t = {res['t'][k]:5.2f} s",
                                       'active: ' + (', '.join(act) or '-')]))
            finally:
                w.close(); r.close()
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def _plot(path, t, Q, res, sched):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(NQ, 1, figsize=(11, 3 + 3 * NQ), sharex=True, squeeze=False); ax = ax[:, 0]
    for j, n in enumerate(JOINTS):
        ax[j].plot(t, np.degrees(Q[:, j]), color='#1f5fbf', lw=1.5, label='planned')
        ax[j].plot(res['t'], np.degrees(res['qpos'][:, j]), color='#7a3fbf', lw=1.3, ls='--', label='pulse sim')
        ax[j].set_ylabel(f'{n} [deg]' + (' (held)' if n in sched['held_joints'] else '')); ax[j].grid(alpha=.3)
        for w in res['pulse_windows']:
            if w['joint'] == n:
                ax[j].axvspan(w['start_s'], w['end_s'], color='#7a3fbf', alpha=.12, lw=0)
    ax[0].legend(fontsize=8); ax[-1].set_xlabel('t [s] (shaded: pulses)')
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


if __name__ == '__main__':
    main()
