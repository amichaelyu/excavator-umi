"""Driving the excavator through the GPIO bridge (hardware.gpio, hardware/rc-control/src/main.cpp): pin-map checks, sequential pulses, homing and the stall watchdog.

The machine only takes timed button presses. :class:`BridgeExecutor` sends them one at a time over one
bridge session (hardware.gpio; or only logs them in a dry run), refusing any GPIO that is not a pin of an allowed joint in
configs/hardware/hardware_map.json; :func:`homing_pulses` drives each joint into its top hard stop; :class:`Watchdog`
releases every pin if the caller stops beating.
"""
from __future__ import annotations

import threading
import time

from digital_twin.sim import pins

from ..paths import HARDWARE_MAP
from . import gpio
from .state_estimator import POLICY_JOINTS, load_joint_config


class RunnerError(RuntimeError):
    """A refusal or configuration problem with a user-facing message."""


STALL_S = 2.0
POLICY_PINS_JOINTS = POLICY_JOINTS                # the joints a policy loop may pulse unless the profile names its own
HOMING_JOINTS = ('stick', 'boom', 'bucket')


def load_hw(path=None) -> dict:
    """The hardware map (digital_twin.sim.pins.read)."""
    return pins.read(path or HARDWARE_MAP)


joint_entry = pins.joint_entry       # a joint's pins in ``joints``


def allowed_pins(hw: dict, phase: str, joints=POLICY_PINS_JOINTS) -> dict[int, str]:
    """{gpio: joint} the runner may send in a phase: the profile's ``joints`` (default boom/bucket) when
    available in the pin map; plus the stick for homing. Drive pins are never listed."""
    out = {}
    for n in joints:
        j = joint_entry(hw, n) or {}
        if j.get('status', 'available') == 'available':
            out[pins.entry_pin(j, 1)] = n; out[pins.entry_pin(j, -1)] = n
    if phase == 'homing':
        s = stick_entry(hw)
        if s:
            out[pins.entry_pin(s, 1)] = 'stick'; out[pins.entry_pin(s, -1)] = 'stick'
    return out


def stick_entry(hw: dict) -> dict | None:
    return joint_entry(hw, 'stick')


def sign_verified(hw: dict, n: str) -> bool:
    """Pin direction verified on hardware. The map records only failures (``sign_verified: false``, on the
    top-level ``slew`` for the slew, else the ``joints`` entry); without one a joint counts as verified."""
    if n == 'slew' and 'sign_verified' in (hw.get('slew') or {}):
        return bool(hw['slew']['sign_verified'])
    return bool((joint_entry(hw, n) or {}).get('sign_verified', True))


def stick_blocked(hw: dict) -> str | None:
    """Why the stick may not be pulsed (None when it may)."""
    return None if stick_entry(hw) else 'no stick pins in hardware_map.json'


def check_pulse_pins(pulses, hw: dict, phase: str, joints=POLICY_PINS_JOINTS) -> None:
    ok = allowed_pins(hw, phase, joints)
    for p in pulses:
        g = int(p['gpio'])
        if g not in ok or ok[g] != p['joint']:
            raise RunnerError(f"refusing GPIO {g} for {p['joint']} in phase {phase!r} (allowed: {sorted(ok)})")


def bridge_target(args) -> 'gpio.GpioBridge | str':
    """The bridge an argparse namespace names: its ``bridge`` attribute (a :class:`gpio.GpioBridge`, set by
    in-process callers such as the hub) takes precedence over ``bridge_host`` (CLI ``--bridge-host``)."""
    b = getattr(args, 'bridge', None)
    return b if b is not None else (getattr(args, 'bridge_host', None) or gpio.DEFAULT_HOST)


class BridgeExecutor:
    """Sequential pulses over one persistent session on the GPIO bridge.

    ``bridge``: a :class:`gpio.GpioBridge` (in-process, the hub's) or a hub ``'host:port'`` (websocket).
    ``dry_run`` prints the triggers and opens no connection (it still waits out the schedule when
    ``pace`` so the loop timing matches a live run). Live: open a session, wait for ``bridge.esp32``,
    then one ``trigger`` at a time matched to its ``ack`` by id; ``release all`` on every exit path, and
    from the watchdog thread on a stall.
    """

    def __init__(self, hw: dict, bridge: 'gpio.GpioBridge | str' = gpio.DEFAULT_HOST, dry_run: bool = True,
                 pace: bool = True, connect_timeout: float = 10.0, ack_timeout: float = 6.0, log=print,
                 joints=POLICY_PINS_JOINTS):
        self.hw, self.bridge, self.dry_run, self.pace, self.log = hw, bridge, dry_run, pace, log
        self.host = bridge if isinstance(bridge, str) else None
        self.joints = tuple(joints)                    # the profile's joints: the pins it may send
        self.connect_timeout, self.ack_timeout = connect_timeout, ack_timeout
        self.session = None
        self.esp32 = threading.Event()
        self._ids, self._waiting, self._lock = 0, {}, threading.Lock()
        self.abort = threading.Event()
        self.sent = []
        self.latency_samples = []               # (kind, seconds): 'state' probes and live 'trigger' acks
        self._state_evt = threading.Event()

    @property
    def time(self) -> float:
        return time.monotonic()

    # --- live connection
    def connect(self):
        if self.dry_run:
            return
        self.log(f'connecting to hardware bridge: {gpio.describe(self.bridge)}')
        self.session = gpio.connect(self.bridge, timeout=self.connect_timeout)
        threading.Thread(target=self._reader, args=(self.session,), daemon=True, name='bridge-reader').start()
        if not self.esp32.wait(self.connect_timeout):
            raise RunnerError('bridge connected but the ESP32 is offline')
        self.log('hardware bridge connected; ESP32 online')

    def _reader(self, session):
        try:
            while True:
                msg = session.recv(0.5)
                if msg is None:
                    if session.closed:
                        break
                    continue
                typ = msg.get('type')
                if typ == 'bridge':
                    (self.esp32.set if msg.get('esp32') else self.esp32.clear)()
                elif typ == 'state':
                    self._state_evt.set()
                elif typ in ('ack', 'error'):
                    with self._lock:
                        slot = self._waiting.get(msg.get('id'))
                    if slot:
                        slot['reply'] = msg; slot['event'].set()
        except Exception:
            pass
        self.esp32.clear()                              # session closed: the link is gone

    def _send(self, obj, expect_reply=True, heartbeat=None):
        with self._lock:
            self._ids += 1
            i = self._ids
            slot = {'event': threading.Event(), 'reply': None}
            if expect_reply:
                self._waiting[i] = slot
        obj = {**obj, 'id': i}
        t_send = time.monotonic()
        try:
            self.session.send(obj)
        except ConnectionError as ex:
            raise RunnerError(f'bridge connection closed: {ex}') from None
        self.sent.append({**obj, 't': time.time()})
        if not expect_reply:
            return None
        # Keep the watchdog fed while waiting: a trigger is a timed press the ESP32 ends by itself, so a
        # slow ack (bridge / Wi-Fi hiccup) is not a stuck pin. ack_timeout stays the hard bound.
        deadline = t_send + self.ack_timeout
        while not slot['event'].wait(min(0.05, max(0.0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                raise RunnerError(f'no ack from the bridge for {obj}')
            if heartbeat:
                heartbeat()
        ack_s = time.monotonic() - t_send
        if obj.get('cmd') == 'trigger':
            self.latency_samples.append(('trigger', ack_s))
            if ack_s > 0.5:
                self.log(f"WARNING: bridge ack for GPIO {obj.get('pin')} took {ack_s * 1000:.0f} ms")
        with self._lock:
            self._waiting.pop(i, None)
        if slot['reply'].get('type') == 'error':
            raise RunnerError(f"bridge error for {obj}: {slot['reply'].get('error')}")
        return slot['reply']

    def probe_latency(self, n: int = 5, timeout: float = 2.0, heartbeat=None) -> list[float]:
        """Round trips of the harmless ``{"cmd": "state", "fresh": true}`` query: ``fresh`` makes the bridge ask
        the ESP32 instead of answering from its cache (hardware.gpio). Read-only, so it also runs in a
        dry run, over a short-lived session (nothing that moves the machine is sent). ``heartbeat`` is called while
        waiting, so a caller's stall watchdog does not trip on a slow reply."""
        out = []
        beat = heartbeat or (lambda: None)
        if self.session is not None:                      # live: the reader thread flags 'state' replies
            for _ in range(n):
                self._state_evt.clear()
                t0 = time.monotonic()
                self.session.send({'cmd': 'state', 'fresh': True, 'id': f'probe-{len(self.latency_samples)}'})
                while not self._state_evt.wait(0.1) and time.monotonic() - t0 < timeout:
                    beat()
                if self._state_evt.is_set():
                    out.append(time.monotonic() - t0)
        else:
            with gpio.connect(self.bridge, timeout=timeout) as s:
                t_end = time.monotonic() + 0.3            # drain the on-connect burst
                while time.monotonic() < t_end and s.recv(max(0.01, t_end - time.monotonic())) is not None:
                    pass
                for i in range(n):
                    t0 = time.monotonic()
                    s.send({'cmd': 'state', 'fresh': True, 'id': f'probe-{i}'})
                    while time.monotonic() - t0 < timeout:
                        beat()
                        msg = s.recv(max(0.01, min(0.1, timeout - (time.monotonic() - t0))))
                        if msg is None:
                            continue
                        if msg.get('type') == 'state':
                            out.append(time.monotonic() - t0); break
                        if msg.get('type') == 'error':
                            break
        self.latency_samples += [('state', x) for x in out]
        return out

    # --- execution
    def execute(self, pulses, gap_ms: float = 0.0, phase: str = 'policy', heartbeat=None) -> dict:
        check_pulse_pins(pulses, self.hw, phase, self.joints)
        start = time.monotonic()
        tag = '[dry run]' if self.dry_run else '[LIVE]'
        prev_end, sent = start, 0
        for p in pulses:
            t_send = max(start + p['t_start_s'], prev_end + (gap_ms / 1000.0 if sent else 0.0))
            if not self.pace:                            # virtual schedule time for the log
                prev_end = t_send + p['duration_ms'] / 1000.0
            self._wait_until(t_send, heartbeat)
            if self.abort.is_set():
                raise RunnerError('aborted (watchdog / stop requested)')
            self.log(f"{tag} {phase}: t+{t_send - start:6.3f}s trigger GPIO {p['gpio']:>2} "
                     f"({p['joint']} {p['direction']:+d}, {p.get('label', '')}) {p['duration_ms']} ms")
            if not self.dry_run:
                if not self.esp32.is_set():
                    raise RunnerError('ESP32 went offline; aborting')
                check_pulse_pins([p], self.hw, phase, self.joints)
                self._send({'cmd': 'trigger', 'pin': int(p['gpio']), 'ms': int(p['duration_ms'])}, heartbeat=heartbeat)
            else:
                self.sent.append({'cmd': 'trigger', 'pin': int(p['gpio']), 'ms': int(p['duration_ms']), 'dry_run': True})
            sent += 1
            if self.pace:
                prev_end = time.monotonic() + p['duration_ms'] / 1000.0
        self._wait_until(prev_end, heartbeat)
        return {'t_start': start, 't_end': time.monotonic(), 'sent': sent}

    def _wait_until(self, t_abs, heartbeat):
        if not self.pace:
            return
        while True:
            left = t_abs - time.monotonic()
            if left <= 0 or self.abort.is_set():
                return
            if heartbeat:
                heartbeat()
            time.sleep(min(left, 0.05))

    def advance(self, dt: float):
        if self.pace and dt > 0:
            time.sleep(dt)

    def release_all(self):
        if self.dry_run or self.session is None:
            return
        try:
            self._send({'cmd': 'release', 'pin': 'all'}, expect_reply=False)
        except Exception as ex:                          # pragma: no cover - best effort
            self.log(f'WARNING: release all failed: {ex}')


    def close(self):
        self.release_all()
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None


class Watchdog:
    """Calls ``on_stall`` (release all pins) and sets ``tripped`` when ``beat()`` stops for > stall_s."""

    def __init__(self, on_stall, stall_s: float = STALL_S, clock=time.monotonic):
        self.on_stall, self.stall_s, self.clock = on_stall, stall_s, clock
        self.last = clock()
        self.tripped = threading.Event()
        self._stop = threading.Event()
        self._armed = False
        self._thread = None

    def beat(self):
        self.last = self.clock()

    def arm(self):
        self.beat(); self._armed = True
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name='watchdog')
            self._thread.start()

    def disarm(self):
        self._armed = False

    def check(self) -> bool:
        if self._armed and not self.tripped.is_set() and self.clock() - self.last > self.stall_s:
            self.tripped.set()
            self.on_stall()
            return True
        return False

    def _run(self):
        while not self._stop.wait(0.1):
            self.check()

    def stop(self):
        self._stop.set()


def homing_pulses(hw: dict, gap_ms: float, joint_cfg: dict | None = None, joints=HOMING_JOINTS,
                  compute: bool = False) -> list[dict]:
    """Ordered homing pulses (stick, boom, bucket): each drives its joint into its top hard stop for
    ``home_duration_s`` (or, when null, range / speed-in-that-direction x 1.5 + 1 s). With
    ``compute`` (``joint_cfg`` holds the measured real ranges / speeds, as in the runner) the
    stored value is ignored and the duration is range / slower-direction speed x 1.5 + 1 s."""
    cfg = joint_cfg or load_joint_config()
    home = hw['homing']
    out, t = [], 0.0
    for n in home['order']:
        if n not in joints:
            continue
        h = home['joints'][n]
        d = int(h['home_direction'])
        c = cfg[n]
        v = c['v_pos'] if d > 0 else c['v_neg']
        dur = None if compute else h.get('home_duration_s')
        if compute:
            dur = (c['hi'] - c['lo']) / min(c['v_pos'], c['v_neg']) * 1.5 + 1.0
        elif dur is None:
            dur = (c['hi'] - c['lo']) / v * 1.5 + 1.0
        jm = joint_entry(hw, n)
        gpio = pins.entry_pin(jm, d)
        ms = int(min(max(round(dur * 1000), 1), 60000))    # bridge limit 1..60000 ms
        out.append({'t_start_s': round(t, 4), 'joint': n, 'direction': d, 'duration_ms': ms, 'gpio': gpio,
                    'label': jm[pins.side(d)]['label'],
                    'sign_verified': bool(jm.get('sign_verified', True)), 'phase': 'homing'})
        t += ms / 1000.0 + gap_ms / 1000.0
    return out


def home_values(hw: dict) -> dict[str, float]:
    return {n: float(h['home_value_rad']) for n, h in hw['homing']['joints'].items()}
