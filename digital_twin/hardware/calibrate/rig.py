"""The rigs the calibration runs on: :class:`LiveArm` (the real machine: Pi dashboard IMUs + GPIO bridge), its
full-rate :class:`ImuStream`, and :class:`SimArm` (a planar arm with a base turn and three randomly mounted IMUs).

Rig interface (also policy.runner's ``_RunnerArmIO``): ``press(joint, direction, ms)`` / ``pulse``, ``samples(link,
t0, t1)`` -> (t, accel [g], gyro [deg/s]), ``read()`` -> {link: (accel, gyro, t)}, ``sleep(s)``, ``now()``,
``operator(key, message)`` and ``close()``.
"""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

from digital_twin.sim import pins

from .. import gpio
from ..excavator import BridgeExecutor, RunnerError
from ..pi import PiSensorSource
from ..state_estimator import ARM_LINKS

PRESS_JOINTS = ('boom', 'stick', 'bucket', 'slew')       # the joints the calibration may press


class CalibrationError(RuntimeError):
    """A refusal or failed check with a user-facing message."""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def joint_pins(hw: dict, joint: str) -> dict:
    """The map entry with the joint's pins (``joints``)."""
    return hw['joints'][joint]


class ImuStream:
    """Full-rate samples of every arm IMU from the Pi (``/api/imu/recent?imu=<name>``), polled in a
    thread and mapped to the host clock (``time.time``) through ``server_monotonic_s``."""

    def __init__(self, base: str, headers: dict, links=ARM_LINKS, period_s: float = 0.25):
        self.base, self.headers, self.links, self.period = base, headers, links, period_s
        self.samples = {n: {} for n in links}               # read_monotonic_ns -> (accel, gyro)
        self.offsets: list[float] = []
        self.error = None
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name='imu-stream')
        self._thread.start()

    def _get(self, name):
        req = urllib.request.Request(f'{self.base}/api/imu/recent?imu={name}', headers=self.headers)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=2) as r:
            d = json.loads(r.read())
        return d, (t0 + time.time()) / 2

    def _run(self):
        while not self._stop.is_set():
            try:
                for n in self.links:
                    d, t_mid = self._get(n)
                    with self._lock:
                        self.offsets = (self.offsets + [t_mid - d['server_monotonic_s']])[-50:]
                        for s in d['samples']:
                            self.samples[n][s['read_monotonic_ns']] = (s['accel'], s['gyro'])
                self.error = None
            except Exception as ex:                      # noqa: BLE001 - reported by wait_until
                self.error = f'{type(ex).__name__}: {ex}'
            self._stop.wait(self.period)

    def offset(self) -> float:
        with self._lock:
            return float(np.median(self.offsets)) if self.offsets else 0.0

    def latest(self, link) -> float | None:
        with self._lock:
            ks = self.samples[link]
            return max(ks) * 1e-9 + float(np.median(self.offsets)) if ks and self.offsets else None

    def wait_until(self, t: float, timeout_s: float = 3.0):
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            if all((self.latest(n) or -1e18) >= t for n in self.links):
                return
            time.sleep(0.05)
        raise CalibrationError(f'IMU stream from {self.base} is behind or down ({self.error or "no new samples"})')

    def between(self, link, t0, t1):
        off = self.offset()
        with self._lock:
            items = sorted((k, v) for k, v in self.samples[link].items() if t0 <= k * 1e-9 + off <= t1)
        if not items:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
        t = np.array([k for k, _ in items]) * 1e-9 + off
        return t, np.array([v[0] for _, v in items], float), np.array([v[1] for _, v in items], float)

    def close(self):
        self._stop.set()
        self._thread.join(2)


class LiveArm:
    """Real machine: the Pi dashboard's IMUs (status polling + :class:`ImuStream`) and the GPIO bridge
    (hardware.excavator.BridgeExecutor, live). ``bridge``: a :class:`gpio.GpioBridge` or a hub ``'host:port'``.
    ``stop`` (a threading.Event, optional) interrupts the next press / wait (KeyboardInterrupt)."""

    def __init__(self, hw, pi_host, token_file, bridge, yes=False, log=print, stream=True, stop=None):
        self.hw, self.log, self.yes = hw, log, yes
        self.stop = stop
        self.sensors = PiSensorSource(pi_host, token_file, stream=False, log=log)
        self.sensors.select_host('arm IMU calibration')
        self.stream = ImuStream(self.sensors.base, self.sensors.headers) if stream else None
        self.executor = None
        self._bridge_target = bridge

    def _check_stop(self):
        if self.stop is not None and self.stop.is_set():
            raise KeyboardInterrupt('stop requested')

    def _bridge(self):
        if self.executor is None:
            self.executor = BridgeExecutor(self.hw, self._bridge_target, dry_run=False, log=self.log, joints=PRESS_JOINTS)
            self.executor.connect()
        return self.executor

    def read(self) -> dict:
        return self.sensors.imus()[0]

    def press(self, joint, direction, ms, retries=3):
        """Press one pin of ``joint`` for ``ms``; returns when the press has ended. If the ESP32 drops off
        Bluetooth (it resets when the motors sag its supply), wait for it to reconnect and press again."""
        self._check_stop()
        j = joint_pins(self.hw, joint)
        pin = pins.entry_pin(j, direction)
        pulse = [{'t_start_s': 0.0, 'joint': joint, 'direction': int(direction), 'duration_ms': int(ms),
                  'gpio': pin, 'label': f'calibration {joint} {direction:+d}'}]
        for attempt in range(retries + 1):
            try:
                self._bridge().execute(pulse, phase='homing')
                return
            except RunnerError as ex:
                if attempt == retries or not any(k in str(ex) for k in ('did not reply', 'offline', 'closed')):
                    raise CalibrationError(f'bridge: {ex}') from None
                self.log(f'  ESP32 dropped ({ex}); waiting for it to reconnect')
                self._reconnect()

    def _reconnect(self, timeout_s=60.0):
        try:
            self.executor.close()
        except Exception:                                          # noqa: BLE001 - the link is already gone
            pass
        self.executor = None
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            self._check_stop()
            try:
                if gpio.health(self._bridge_target, timeout=2).get('esp32'):
                    self._bridge()
                    return
            except Exception:                                      # noqa: BLE001 - keep polling
                pass
            time.sleep(1.0)
        raise CalibrationError(f'the ESP32 did not reconnect to the bridge within {timeout_s:.0f} s (power / reset?)')

    pulse = press                                                   # servo_joint / policy.runner rig interface

    def samples(self, link, t0, t1):
        self.stream.wait_until(t1)
        return self.stream.between(link, t0, t1)

    def sleep(self, s):
        if self.stop is None:
            time.sleep(s)
        elif self.stop.is_set() or self.stop.wait(max(0.0, s)):
            raise KeyboardInterrupt('stop requested')

    def now(self):
        return time.time()

    def operator(self, key, message) -> bool:
        if self.yes and key == 'confirm':
            self.log(f'{message} -> yes (--yes)')
            return True
        try:
            a = input(f'{message} ' + ('[y/n] ' if key == 'confirm' else '[Enter when done] '))
        except EOFError:
            return False
        return key != 'confirm' or a.strip().lower().startswith('y')

    def close(self):
        try:
            if self.executor is not None:
                self.executor.release_all()
        finally:
            if self.executor is not None:
                self.executor.close()
            if self.stream is not None:
                self.stream.close()
            self.sensors.close()


class SimArm:
    """Planar arm without hard stops on a base that turns, three randomly mounted IMUs streamed at ``rate_hz``.
    A press drives a joint at its speed from ``onset_s`` after the press; the motor runs on for up to ``run_on_s``
    after release (the base for up to ``slew_run_on_s``); at either end of travel a driven joint chatters (gyro / accel
    noise, no net motion) and settles up to ``settle_deg`` off the end. The base (``slew``, + = left =
    counter-clockwise seen from above) has no ends.
    ``dead`` = {(joint, direction)} pins that move nothing; ``flip`` = joints whose pins are reversed; ``stop`` (a
    threading.Event) interrupts the next press."""

    def __init__(self, hw, seed=0, rate_hz=100.0, q0=None, ends=None, speeds=None, tilt_deg=1.5, noise_g=0.004,
                 gyro_noise_dps=0.3, gyro_bias_dps=6.0, run_on_s=0.8, settle_deg=2.0, onset_s=0.1, dead=(), flip=(), swap=None,
                 slew_speeds=(1.0, 1.05), slew_run_on_s=0.15, stop=None):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(seed)
        h = hw['homing']['joints']
        span = {'boom': 0.70, 'stick': 1.9, 'bucket': 2.35}
        self.ends = ends or {n: ((h[n]['home_value_rad'] - span[n], h[n]['home_value_rad']) if h[n]['home_direction'] > 0
                                 else (h[n]['home_value_rad'], h[n]['home_value_rad'] + span[n])) for n in ARM_LINKS}
        self.stops = self.ends
        self.speeds = speeds or {'boom': (0.35, 0.5), 'stick': (0.4, 0.4), 'bucket': (1.6, 1.9)}   # (+, -) rad/s
        self.slew_speeds = slew_speeds                     # (+ left, - right) rad/s
        self.q = dict(q0 or {n: float(np.mean(self.ends[n])) for n in ARM_LINKS})
        self.slew = 0.0
        self.mount = {n: Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix() for n in ARM_LINKS}
        self.tilt = Rotation.from_rotvec(np.radians(tilt_deg) * np.array([0.3, -1.0, 0.0]) / np.hypot(0.3, 1.0)).as_matrix()
        self.rng, self.dt = rng, 1.0 / rate_hz
        self.noise_g, self.gyro_noise = noise_g, gyro_noise_dps
        self.true_bias = {n: rng.uniform(-gyro_bias_dps, gyro_bias_dps, 3) for n in ARM_LINKS}   # sensor gyro bias
        self.run_on_s, self.settle_deg, self.onset_s, self.slew_run_on_s = run_on_s, settle_deg, onset_s, slew_run_on_s
        self.dead, self.flip = set(dead), set(flip)
        self.swap = swap                                   # (a, b): IMU names a and b report each other's link
        self.stop = stop
        self.t = 0.0
        self.motor = {}                                    # joint -> (direction, t_on, t_off)
        self.buf = {n: [] for n in ARM_LINKS}              # (t, accel, gyro)
        self.pulses = []

    @staticmethod
    def _rot_y(angle):                       # nose-up pitch = rotation about world -y (twin convention)
        c, s = math.cos(-angle), math.sin(-angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def _rot_z(angle):                       # base turn, + = counter-clockwise seen from above
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _R(self, link):
        pitch = sum(self.q[n] for n in ARM_LINKS[:ARM_LINKS.index(link) + 1])
        return self.tilt @ self._rot_z(self.slew) @ self._rot_y(pitch) @ self.mount[link]

    def up_imu(self, link):
        return self._R(link).T @ np.array([0.0, 0.0, 1.0])

    def _step(self):
        rate = {n: 0.0 for n in ARM_LINKS}
        yaw = 0.0
        chatter = set()
        for j, (d, t_on, t_off) in list(self.motor.items()):
            if self.t < t_on:
                continue
            if self.t >= t_off:
                del self.motor[j]
                if j == 'slew':
                    continue
                lo, hi = self.ends[j]
                if self.q[j] <= lo + 1e-9 or self.q[j] >= hi - 1e-9:     # settles off the end
                    off = math.radians(self.rng.uniform(0, self.settle_deg))
                    self.q[j] = hi - off if self.q[j] >= hi - 1e-9 else lo + off
                continue
            if j == 'slew':
                yaw = d * self.slew_speeds[0 if d > 0 else 1]
                self.slew += yaw * self.dt
                continue
            v = self.speeds[j][0 if d > 0 else 1]
            lo, hi = self.ends[j]
            q1 = float(np.clip(self.q[j] + d * v * self.dt, lo, hi))
            rate[j] = (q1 - self.q[j]) / self.dt
            if abs(q1 - self.q[j]) < 1e-12:
                chatter.add(j)
            self.q[j] = q1
        pitch_axis = self.tilt @ self._rot_z(self.slew) @ np.array([0.0, -1.0, 0.0])
        up_axis = self.tilt @ np.array([0.0, 0.0, 1.0])
        for i, n in enumerate(ARM_LINKS):
            R = self._R(n)
            w = sum(rate[m] for m in ARM_LINKS[:i + 1])
            gyro = (np.degrees(R.T @ pitch_axis * w + R.T @ up_axis * yaw) + self.true_bias[n]
                    + self.rng.normal(0, self.gyro_noise, 3))
            acc = R.T @ np.array([0.0, 0.0, 1.0]) + self.rng.normal(0, self.noise_g, 3)
            if any(c in ARM_LINKS[:i + 1] for c in chatter):
                gyro = gyro + self.rng.normal(0, 30.0, 3)
                acc = acc + self.rng.normal(0, 0.4, 3)
            src = n
            if self.swap and n in self.swap:
                src = self.swap[1] if n == self.swap[0] else self.swap[0]
            self.buf[src].append((self.t, acc, gyro))
        self.t += self.dt

    def sleep(self, s):
        for _ in range(max(1, int(round(s / self.dt)))):
            self._step()

    def press(self, joint, direction, ms):
        if self.stop is not None and self.stop.is_set():
            raise KeyboardInterrupt('stop requested')
        self.pulses.append((joint, int(direction), int(ms)))
        if (joint, int(direction)) not in self.dead:
            t_on = self.t + self.onset_s
            run_on = self.rng.uniform(0, self.slew_run_on_s if joint == 'slew' else self.run_on_s)
            self.motor[joint] = (-int(direction) if joint in self.flip else int(direction), t_on, t_on + ms / 1000.0 + run_on)
        self.sleep(ms / 1000.0)

    pulse = press

    def samples(self, link, t0, t1):
        rows = [r for r in self.buf[link] if t0 <= r[0] <= t1]
        if not rows:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])

    def read(self) -> dict:
        self._step()
        return {n: (self.buf[n][-1][1], self.buf[n][-1][2], self.buf[n][-1][0]) for n in ARM_LINKS}

    def now(self):
        return self.t

    def operator(self, key, message) -> bool:
        return True

    def close(self):
        pass
