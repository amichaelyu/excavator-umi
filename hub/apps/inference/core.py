"""Inference dashboard for running the CLOAK π0.5 policies (served on Lambda) on the excavator: the :class:`Inference`
service of the hub app ``inference`` (routes in hub/apps/inference/__init__.py; ``.venv/bin/python -m hub``, then
http://127.0.0.1:8888/inference). Its options are set in configs/hub/apps.json or saved from the hub page into
configs/hub/local.json. The page is the view module hub/web/views/inference/.

No simulation: the backend is always the GPIO bridge, in DRY RUN (default: pulses compiled, paced and
shown, nothing sent) or LIVE (armed per run), and the observation is always the real Pi camera.

:class:`Inference` holds the state: one background worker job at a time (a
:class:`digital_twin.policy.runner.PolicyRunner`, or ``digital_twin.hardware.calibrate.run``, both
in-process on a worker thread and handed the hub's :class:`digital_twin.hardware.gpio.GpioBridge`), a
sensor poller thread and an
:class:`~hub.apps.inference.events.EventBus` the router streams as Server-Sent Events. This module has the
arming, the policy / home jobs, their lifecycle (stop watchdog, E-STOP, deadman), the checkpoint
registry and the run records; the rest of the package:

    events.py            EventBus (SSE fan-out)
    pi.py                PiClient (Pi dashboard status + camera over the host list), bridge description
    twin.py              TwinRenderer (MuJoCo renders), TwinState (the twin panel's live joint state)
    jobs.py              Job, DashboardRunner (the policy runner streaming to the page), run summaries
    calibration.py       the calibration panel: what the hardware map holds, the last runs and their logs
    calibration_jobs.py  CalibrationJobs: the Inference mixin that validates and runs the calibration job
    live.py              LiveViews: the Inference mixin for the sensor poller, estimate, observation, twin / camera views
    util.py              Refused, JSON-safe values, JSON files, UTC stamps

Safety: the hub binds 127.0.0.1 and rejects foreign Host / Origin headers. Every job is a dry run
unless the request carries ``live: true`` and a fresh arm token (one click on ARM LIVE, single
use, per run, 120 s); live then maps to ``--live --i-understand-open-loop`` and the worker applies the CLI's
own flags (no calibration gate: the policy is runnable whatever the calibration state). Any
unexpected server error, a crashed worker, or losing every page connection during a live job
triggers the emergency stop.
"""
from __future__ import annotations

import io
import math
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from hub.apps._common import DEFAULT_PI_HOST
from hub.apps.inference.calibration import CAL_ROOT
from hub.apps.inference.calibration_jobs import CalibrationJobs
from hub.apps.inference.events import EventBus
from hub.apps.inference.jobs import DashboardRunner, Job, filter_runner_argv, run_summary
from hub.apps.inference.live import LiveViews
from hub.apps.inference.pi import BRIDGE_UNAVAILABLE, DEFAULT_TOKEN_FILE, PiClient, describe_bridge
from hub.apps.inference.twin import TWIN_CAMERAS, TwinRenderer, TwinState
from hub.apps.inference.util import Refused, dumps, utc_stamp
from digital_twin.paths import HARDWARE_MAP, REPO
from digital_twin.sim import pins
from digital_twin.policy import checkpoints as local_checkpoints
from digital_twin.policy import runner as pr
from policy.configs.schema import TASKS

RUNS_DIR = REPO / 'artifacts' / 'inference-runs'
STOP_TIMEOUT_S = 6.0                               # stop -> worker must end, else the job is abandoned
ARM_TTL_S = 120.0
DEADMAN_S = 3.0                                    # live job with no page connected -> e-stop
RUN_ID_RE = re.compile(r'^[A-Za-z0-9_.-]{1,80}$')
FILE_RE = re.compile(r'^(frames/)?[A-Za-z0-9_.-]{1,120}$')
FACE_FORWARD_TOL_DEG = 2.0                         # quick home: leave the base alone within this
FACE_FORWARD_MAX_MS = 3000                         # ... and never turn it for longer than this
# The runner profile the dashboard offers: the 4-joint CLOAK π0.5 contract.
PROFILE_LABELS = {pr.PROFILE: 'CLOAK π0.5 · slew, boom, stick, bucket'}


class Inference(CalibrationJobs, LiveViews):
    """Dashboard state: one job at a time, arming, sensors, estimate, observation, e-stop."""

    def __init__(self, pi_host=DEFAULT_PI_HOST, bridge=None, token_file=DEFAULT_TOKEN_FILE,
                 hardware_map=HARDWARE_MAP, runs_dir=RUNS_DIR,
                 poll: bool = True, pace: bool = True,
                 stop_timeout_s: float = STOP_TIMEOUT_S, pi_period_s: float = 0.2, bridge_period_s: float = 1.0,
                 cal_root=CAL_ROOT, local_registry=local_checkpoints.DEFAULT_REGISTRY):
        """``bridge``: the GPIO bridge (a :class:`digital_twin.hardware.gpio.GpioBridge`: ``status()``,
        ``release_all()``, ``trigger()``, ``session()``), or a zero-argument callable returning the current
        one (the hub's gpio app service, which may be restarted or absent: None = unavailable)."""
        self.bus = EventBus()
        self.pi = PiClient(pi_host, token_file)
        self._bridge = bridge
        self.hub = None                                     # set by hub.apps.inference.start (reset restarts the app)
        self.token_file = Path(token_file).expanduser() if token_file else None
        self.hardware_map = Path(hardware_map)
        self.runs_dir = Path(runs_dir)
        self.pace = pace                                    # dry run waits out schedules like a live run
        self.stop_timeout_s = stop_timeout_s
        self.cal_root = Path(cal_root)
        self.local_registry = local_registry
        self.cal_run = None                                 # None: digital_twin.hardware.calibrate.run (tests inject fakes)
        self.pi_period_s, self.bridge_period_s = pi_period_s, bridge_period_s
        self.job: Job | None = None
        self._job_lock = threading.Lock()
        self._arm = None                                    # (token, expires monotonic)
        self._arm_lock = threading.Lock()
        self._lock = threading.Lock()
        self.q_est = None
        self.q_source = None
        self.q_est_t = None                                 # monotonic time of the last set_estimate
        self.live_twin = TwinState(self.hardware_map)       # the twin panel's joint state
        self._twin_state_jpg = None                         # (q key, camera, jpeg): skip identical renders
        self.obs = None                                     # (jpeg, job id, time, state)
        self._masker = None                                 # (self_mask.json mtime, SelfMasker) for previews
        self._cam_mask = None                               # (pose / shape / masker key, mask) on the Pi camera
        self.sensors = {'pi': {'ok': None}, 'bridge': {'ok': None}}
        self._ckpt_cache = None
        self._ckpt_lock = threading.Lock()
        self._registry_lock = threading.Lock()
        self.twin = TwinRenderer()
        self._closing = threading.Event()
        self._no_viewer_since = None
        self._poller = None
        if poll:
            self._poller = threading.Thread(target=self._poll_loop, daemon=True, name='inference-sensors')
            self._poller.start()

    # ---- bridge
    @property
    def bridge(self):
        """The current GPIO bridge, or None when the hub's gpio app is not running."""
        b = self._bridge
        if b is not None and callable(b) and not hasattr(b, 'release_all'):
            b = b()
        return b

    def bridge_release_all(self) -> dict:
        """Release every pin through the bridge; never raises (E-STOP path)."""
        b = self.bridge
        if b is None:
            return {'ok': False, 'sent': False, 'error': BRIDGE_UNAVAILABLE}
        try:
            return b.release_all()
        except Exception as ex:                             # GpioBridge.release_all never raises; a fake might
            return {'ok': False, 'sent': False, 'error': f'{type(ex).__name__}: {ex}'}

    def bridge_trigger(self, pin: int, ms: int) -> dict:
        b = self.bridge
        if b is None:
            raise RuntimeError(BRIDGE_UNAVAILABLE)
        return b.trigger(int(pin), int(ms))

    # ---- state
    def state(self) -> dict:
        with self._arm_lock:
            armed = self._arm is not None and self._arm[1] > time.monotonic()
            left = max(0.0, self._arm[1] - time.monotonic()) if armed else 0.0
        job = self.job
        return {'job': job.info() if job else None, 'busy': bool(job and job.running),
                'armed': armed, 'arm_expires_in_s': round(left, 1), 'q_est': self.q_est, 'q_source': self.q_source,
                'pi_host': self.pi.base, 'bridge': describe_bridge(self.bridge), 'viewers': self.bus.n_subscribers}

    def info(self) -> dict:
        """Shown on the hub page."""
        return {'pi': self.pi.base, 'bridge': describe_bridge(self.bridge)}

    def publish_state(self):
        self.bus.publish('state', self.state())

    def log(self, job: Job | None, line: str):
        line = str(line)
        self.bus.publish('log', {'job': job.id if job else None, 'kind': job.kind if job else None, 'line': line})
        if job is not None:
            try:
                with open(job.dir / 'log.txt', 'a') as f:
                    f.write(f'{datetime.now().isoformat(timespec="milliseconds")} {line}\n')
            except OSError:
                pass

    # ---- arming
    def arm(self) -> dict:
        """One click on ARM LIVE: a single-use token for the next job, expiring after ARM_TTL_S."""
        tok = secrets.token_urlsafe(18)
        with self._arm_lock:
            self._arm = (tok, time.monotonic() + ARM_TTL_S)
        self.publish_state()
        return {'arm_token': tok, 'expires_in_s': ARM_TTL_S}

    def disarm(self):
        with self._arm_lock:
            self._arm = None
        self.publish_state()

    def consume_arm(self, token) -> bool:
        with self._arm_lock:
            ok = (self._arm is not None and isinstance(token, str) and token
                  and secrets.compare_digest(token, self._arm[0]) and self._arm[1] > time.monotonic())
            self._arm = None                                  # any live attempt consumes the arming
        return bool(ok)

    # ---- job control
    def _new_dir(self, kind: str) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        base = f'{utc_stamp()}-{kind}'
        d, i = self.runs_dir / base, 1
        while d.exists():
            i += 1
            d = self.runs_dir / f'{base}-{i}'
        d.mkdir(parents=True)
        return d

    @staticmethod
    def _bridge_only(body: dict):
        if body.get('backend', 'bridge') != 'bridge' or body.get('camera', 'live') != 'live':
            raise Refused(400, 'policy rollout has no simulation: the backend is always the bridge (DRY RUN or '
                               'LIVE) and the observation is always the real Pi camera')

    def _check_live(self, body: dict) -> bool:
        live = bool(body.get('live'))
        if not live:
            return False
        if not self.consume_arm(body.get('arm_token')):
            raise Refused(403, 'live mode refused: not armed (press ARM LIVE; the arming is single use '
                               f'and expires after {ARM_TTL_S:.0f} s)')
        return True

    def _begin(self, kind: str, params: dict, live: bool) -> Job:
        with self._job_lock:
            if self.job is not None and self.job.running:
                raise Refused(409, f'a {self.job.kind} job is running ({self.job.id}); stop it first')
            job = Job(kind, params, live, self._new_dir(kind))
            self.job = job
        self._write_record(job)
        return job

    def _write_record(self, job: Job):
        data = {**job.info(), 'summary': job.summary, 'python': sys.executable}
        tmp = job.dir / f'.job.json.{os.getpid()}.partial'
        try:
            tmp.write_text(dumps(data))
            os.replace(tmp, job.dir / 'job.json')
        except OSError:
            pass

    @staticmethod
    def _num(body, key, default, lo, hi, cast=float, optional=False):
        v = body.get(key, default)
        if v in (None, '') and optional:
            return None
        try:
            v = cast(v)
        except (TypeError, ValueError):
            raise Refused(400, f'{key} must be a number') from None
        if not (lo <= v <= hi) or (isinstance(v, float) and not math.isfinite(v)):
            raise Refused(400, f'{key} must be within [{lo}, {hi}]')
        return v

    def _common_argv(self, live: bool, body: dict, record: Path) -> list[str]:
        argv = ['--pi-host', self.pi.spec,
                '--hardware-map', str(self.hardware_map), '--record', str(record)]
        if self.token_file:
            argv += ['--token-file', str(self.token_file)]
        argv.append('--home-stick' if body.get('home_stick', True) else '--no-home-stick')
        if live:
            argv += ['--live', '--i-understand-open-loop']
        elif not self.pace:
            argv.append('--no-pace')
        return argv

    def _finish_args(self, argv: list[str], homing: bool):
        """Parse with the runner's own CLI. No calibration gate here: the policy runs whatever the
        calibration state (sign_verified / arm IMU calibration are not checked); live needs
        only the arming (+ --live --i-understand-open-loop and the max-seconds bound)."""
        import contextlib
        argv, dropped = filter_runner_argv(argv)
        if dropped:
            self.bus.publish('log', {'job': None, 'line': f'runner CLI has no {dropped}; not passed'})
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                args = pr.build_parser().parse_args(argv)
        except SystemExit:                                   # the runner CLI changed under us: refuse, do not crash
            raise Refused(400, 'policy.runner rejected the arguments: ' + err.getvalue().strip().splitlines()[-1]) from None
        if getattr(args, 'exec_latency_s', 0.2) is None:
            args.exec_latency_s = 0.2
        if args.live and args.max_seconds is None:
            raise Refused(400, 'live needs a max-seconds bound')
        if homing and getattr(args, 'home_stick', False) and pr.stick_blocked(pr.load_hw(self.hardware_map)):
            raise Refused(400, 'Stick homing is unavailable: hardware_map.json has no stick pins. Turn off home stick and position the stick manually.')
        args.bridge = self.bridge                            # the hub's GpioBridge (takes precedence over bridge_host)
        if args.live and args.bridge is None:
            raise Refused(503, f'live refused: {BRIDGE_UNAVAILABLE}')
        return args

    def start_policy(self, body: dict) -> dict:
        profile = body.get('profile')
        if profile not in pr.PROFILES:
            raise Refused(400, f'profile must be one of {sorted(pr.PROFILES)}')
        # Only checkpoints in the local (Lambda) registry run; no homing, arming token consumption or
        # worker creation may precede this check.
        url = str(body.get('gateway_url') or '').strip()
        entry = next((e for e in local_checkpoints.entries(self.local_registry) if e['gateway_url'] == url), None)
        if entry is None:
            raise Refused(400, 'gateway_url must be a checkpoint from configs/hub/local_checkpoints.json')
        if entry['profile'] != profile:
            raise Refused(400, f"{entry['label']} is a {entry['profile']} policy; choose the {entry['profile']} profile")
        prompt = str(body.get('prompt') or '').strip()
        if not prompt or len(prompt) > 512:
            raise Refused(400, 'prompt must be 1..512 characters')
        self._bridge_only(body)
        chunk = self._num(body, 'chunk', 3, 1, 50, int)
        hz = self._num(body, 'hz', pr.POLICY_HZ, 0.5, 30.0)
        max_seconds = self._num(body, 'max_seconds', 30.0, 0.1, 900.0)
        max_steps = self._num(body, 'max_steps', None, 1, 100000, int, optional=True)
        start_pose = str(body.get('start_pose') or 'none')
        if start_pose != 'none' and not re.match(r'^-?[\d.]+(,-?[\d.]+){3}$', start_pose):
            raise Refused(400, "start_pose must be 'none' or 'slew_deg,boom_deg,stick_deg,bucket_deg'")
        imu = body.get('imu', 'auto')
        if imu not in ('auto', 'arm', 'off'):
            raise Refused(400, 'imu must be auto, arm or off')
        quick_calibration = body.get('quick_calibration', False) is True      # off unless asked for
        if hz != pr.POLICY_HZ or chunk > 10:
            raise Refused(400, f'π0.5 requires joint targets, {pr.POLICY_HZ} Hz, and chunk 1..10')
        live = self._check_live(body)
        params = {'profile': profile, 'gateway_url': url, 'prompt': prompt, 'mode': 'LIVE' if live else 'DRY RUN',
                  'chunk': chunk, 'hz': hz, 'max_seconds': max_seconds, 'max_steps': max_steps,
                  'start_pose': start_pose, 'imu': imu, 'home': body.get('home', False) is True,
                  'home_stick': bool(body.get('home_stick', True)), 'quick_calibration': quick_calibration,
                  'policy_space': entry['policy_space']}
        argv = ['--profile', profile, '--gateway-url', url, '--prompt', prompt,
                '--chunk', str(chunk), '--hz', str(hz), '--max-seconds', str(max_seconds),
                '--start-pose', start_pose, '--imu', imu]
        if max_steps:
            argv += ['--max-steps', str(max_steps)]
        if not params['home']:
            argv.append('--no-home')
        if not quick_calibration:
            argv.append('--no-quick-calibration')
        job = self._begin('policy', params, live)
        try:
            args = self._finish_args(argv + self._common_argv(live, body, job.dir), params['home'])
        except Exception:
            self._abandon(job)
            raise
        pr.validate_prompt(prompt, log=lambda s: self.log(job, s))
        return self._launch_runner(job, args)

    def face_forward_pulse(self) -> dict | None:
        """The base-turn press that brings the twin's slew estimate back to 0 (facing forward), from the
        calibrated turn rates (hardware_map ``slew``); None when it is already within FACE_FORWARD_TOL_DEG."""
        sl_est = self.live_twin.snapshot()['joints']['slew']
        if sl_est['source'] != 'imu' or sl_est['age_s'] is None or sl_est['age_s'] > 1.0:
            return {'skipped': 'no live base-turn estimate: face the base forward by hand'}
        deg = float(sl_est['deg'])
        if abs(deg) <= FACE_FORWARD_TOL_DEG:
            return None
        hw = pr.load_hw(self.hardware_map)
        sl = hw.get('slew') or {}
        if not sl.get('sign_verified', True) or not (sl.get('yaw_deg_s') or {}).get('left'):
            return {'skipped': 'the base turn is not calibrated (run the hardware calibration): face it forward by hand'}
        word = 'right' if deg > 0 else 'left'                           # + slew = left, so turn back right
        pin = pins.pin(hw, 'slew', -1 if deg > 0 else 1)
        rate = abs(float(sl['yaw_deg_s'][word]))
        onset = float((sl.get('onset_s') or {}).get(word) or 0.0)
        ms = int(round(1000 * (abs(deg) / rate + onset)))
        return {'slew_deg': round(deg, 1), 'direction': word, 'gpio': pin, 'ms': min(ms, FACE_FORWARD_MAX_MS)}

    def start_home(self, body: dict) -> dict:
        """Home stick, boom and bucket into their hard stops. With ``face_forward`` (the Quick home button)
        first turn the base back to facing forward, from the IMU slew estimate and the calibrated turn rate
        (skipped, with a log line, when there is no live estimate or no base-turn calibration)."""
        self._bridge_only(body)
        want_face = bool(body.get('face_forward', False))
        face = self.face_forward_pulse() if want_face else None
        live = self._check_live(body)
        max_seconds = self._num(body, 'max_seconds', 60.0, 1.0, 900.0)
        params = {'mode': 'LIVE' if live else 'DRY RUN', 'home_stick': bool(body.get('home_stick', True)),
                  'face_forward': face,
                  # the base ends facing forward: turned back here, or already within tolerance
                  'faces_forward': bool(want_face and not (face or {}).get('skipped'))}
        argv = ['--home-only', '--imu', 'auto', '--max-seconds', str(max_seconds)]
        job = self._begin('home', params, live)
        if face and face.get('skipped'):
            self.log(job, 'face forward skipped: ' + face['skipped'])
        elif face:
            self.log(job, f"face forward: slew {face['slew_deg']:+.1f} deg -> GPIO {face['gpio']} ({face['direction']}) "
                          f"{face['ms']} ms" + ('' if live else '  [dry run: not sent]'))
            if live:
                try:
                    self.bridge_trigger(face['gpio'], face['ms'])
                    time.sleep(face['ms'] / 1000 + 0.5)                 # let the turn finish before homing
                except Exception as ex:
                    self.bridge_release_all()
                    self._abandon(job)
                    raise Refused(502, f'face-forward turn failed: {ex}') from None
                self.log(job, f"face forward: slew now {self.live_twin.snapshot()['joints']['slew']['deg']:+.1f} deg")
        elif want_face:
            self.log(job, 'face forward: base already within '
                          f'{FACE_FORWARD_TOL_DEG:g} deg of forward')
        try:
            args = self._finish_args(argv + self._common_argv(live, body, job.dir), True)
        except Exception:
            self._abandon(job)
            raise
        return self._launch_runner(job, args)

    # ---- job lifecycle: launch, finalize, stop watchdog, stop, E-STOP
    def _abandon(self, job: Job):
        job.status, job.ended_at = 'refused', datetime.now(timezone.utc).isoformat()
        self._write_record(job)
        self.publish_state()

    def _launch_runner(self, job: Job, args) -> dict:
        job.thread = threading.Thread(target=self._run_runner, args=(job, args), daemon=True, name=f'inference-{job.kind}')
        job.status = 'running'
        job.thread.start()
        self.publish_state()
        return job.info()

    def _run_runner(self, job: Job, args):
        self.log(job, f'{job.kind} job {job.id}: bridge {describe_bridge(getattr(args, "bridge", None))}, '
                      f'{"LIVE" if args.live else "DRY RUN (pulses compiled and shown, nothing sent)"}; '
                      f'camera {self.pi.base}; record -> {job.dir}')
        crashed, status, error = None, 'done', None
        try:
            runner = DashboardRunner(args, self, job)
            job.runner = runner
            runner.run()
            status = 'stopped' if job.stop.is_set() else 'done'
        except pr.RunnerError as ex:
            status, error = ('stopped' if job.stop.is_set() else 'error'), str(ex)
        except Exception as ex:                               # a worker crash is a server error
            status, error, crashed = 'error', f'{type(ex).__name__}: {ex}', ex
        finally:
            if crashed is not None and not job.finalized:
                self.emergency_stop(f'worker crashed: {error}')
            self._finalize(job, status=status, error=error, summary=getattr(job.runner, 'summary', None))

    def _finalize(self, job: Job, status=None, error=None, summary=None) -> bool:
        """End a job exactly once (the worker's own exit, or the stop watchdog abandoning it)."""
        with job.final_lock:
            if job.finalized:
                return False
            job.finalized = True
            if status is not None:
                job.status = status
            if job.status in ('starting', 'running'):
                job.status = 'stopped' if job.stop.is_set() else 'error'
            if error is not None:
                job.error = error
            if summary is not None:
                job.summary = summary
            job.ended_at = datetime.now(timezone.utc).isoformat()
        self._write_record(job)
        if job.kind == 'home' and job.live and job.status == 'done':
            # the homed pose is the reference: the runner reset the job estimate to the hard stops; the twin's
            # slew restarts at 0 once the base was brought back to facing forward
            if (job.params or {}).get('faces_forward'):
                self.live_twin.zero_slew()
                self.bus.publish('log', {'job': job.id, 'line': 'state estimate: homed pose, base facing forward (slew = 0)'})
            else:
                self.bus.publish('log', {'job': job.id, 'line': 'state estimate: homed pose (slew left as estimated)'})
        if job.live and job.status != 'done' and not job.estopped:
            self.bus.publish('log', {'job': job.id, 'line': 'live job ended early: extra release-all via the bridge'})
            self.bus.publish('release', self.bridge_release_all())
        self.bus.publish('job_end', job.info() | {'summary': job.summary})
        self.publish_state()
        return True

    def _stop_watchdog(self, job: Job):
        """Hard timeout: a worker that does not end within stop_timeout_s after a stop (a call that
        never returns) is abandoned: its executor is aborted so it cannot send another pulse, the
        bridge gets a release-all, and the job slot is freed."""
        def watch():
            t = job.thread
            if t is not None:
                t.join(self.stop_timeout_s)
            if (t is None or t.is_alive()) and not job.finalized:
                ex = getattr(job.runner, 'executor', None)
                if ex is not None and hasattr(ex, 'abort'):
                    ex.abort.set()
                self.log(job, f'STOP: worker did not end within {self.stop_timeout_s:.0f} s; job abandoned '
                              '(executor aborted, release-all sent)')
                self.bus.publish('release', self.bridge_release_all())
                self._finalize(job, status='killed',
                               error=f'worker did not stop within {self.stop_timeout_s:.0f} s (abandoned)')
        threading.Thread(target=watch, daemon=True, name='inference-stop-watchdog').start()

    def stop(self, reason: str = 'stop requested') -> dict:
        job = self.job
        if job is None or not job.running:
            return {'stopped': False, 'reason': 'no job running'}
        job.stop.set()
        self.log(job, f'STOP: {reason}')
        if hasattr(job.runner, 'stop'):
            job.runner.stop(reason)                          # the runner's cooperative stop (thread-safe)
        ex = getattr(job.runner, 'executor', None)
        if ex is not None and hasattr(ex, 'abort'):
            ex.abort.set()
        self._stop_watchdog(job)
        return {'stopped': True, 'job': job.id, 'status': 'stopping',
                'hard_timeout_s': self.stop_timeout_s}

    def emergency_stop(self, reason: str = 'EMERGENCY STOP pressed') -> dict:
        """Abort the job, release through the runner's own connection, and send release-all to the
        bridge over a fresh WebSocket (always, dry run or not: releasing is always safe)."""
        self.bus.publish('estop', {'reason': reason, 'phase': 'begin'})
        with self._arm_lock:
            self._arm = None
        out = {'reason': reason}
        job = self.job
        if job is not None and job.running:
            job.estopped = True
            job.stop.set()
            self.log(job, f'EMERGENCY STOP: {reason}')
            if hasattr(job.runner, 'stop'):
                try:
                    job.runner.stop('EMERGENCY STOP')
                except Exception:
                    pass
            ex = getattr(job.runner, 'executor', None)
            if ex is not None:
                if hasattr(ex, 'abort'):
                    ex.abort.set()
                try:
                    ex.emergency_release()
                    out['runner_release'] = 'sent' if getattr(ex, 'ws', None) is not None else 'no live connection'
                except Exception as e:
                    out['runner_release'] = f'failed: {e}'
            self._stop_watchdog(job)
            out['job'] = job.id
        out['bridge_release'] = self.bridge_release_all()
        self.bus.publish('estop', {**out, 'phase': 'done'})
        self.publish_state()
        return out

    # ---- deadman (run by the sensor poller, LiveViews._poll_loop)
    def _deadman(self):
        job = self.job
        if job is None or not job.running or not job.live:
            self._no_viewer_since = None
            return
        if self.bus.n_subscribers > 0:
            self._no_viewer_since = None
            return
        now = time.monotonic()
        if self._no_viewer_since is None:
            self._no_viewer_since = now
        elif now - self._no_viewer_since > DEADMAN_S:
            self._no_viewer_since = None
            self.emergency_stop(f'no dashboard page connected for {DEADMAN_S:.0f} s during a live job')

    # ---- checkpoints, config, run records
    def checkpoints(self, refresh=False) -> dict:
        with self._ckpt_lock:
            if refresh or self._ckpt_cache is None or time.time() - self._ckpt_cache['fetched_at'] > 120:
                runs, error = [], None
                try:
                    runs = local_checkpoints.runs(self.local_registry)
                except Exception as ex:
                    error = f'{type(ex).__name__}: {ex}'
                default = next((r for r in runs if r.get('preferred')), None)
                self._ckpt_cache = {'source': 'local registry', 'error': error, 'runs': runs, 'fetched_at': time.time(),
                                    'default': default and {**default['checkpoints'][0], 'run': default['name'],
                                        'key': default['key'], 'version': default['version'], 'profile': default['profile'],
                                        'policy_space': default['policy_space'], 'backend': default['backend']}}
            return self._ckpt_cache

    def add_checkpoint(self, body):
        if self.local_registry is None:
            raise Refused(409, 'No checkpoint registry configured')
        with self._registry_lock:
            try:
                entry = local_checkpoints.add(body.get('label'), body.get('gateway_url'), body.get('prompt'),
                                              self.local_registry)
            except ValueError as ex:
                raise Refused(400, str(ex)) from ex
            except Exception as ex:                       # refused / timed out / wrong contract on the wire
                raise Refused(502, f'Could not use {body.get("gateway_url")}: {ex}') from ex
            self._ckpt_cache = None
        return entry

    def remove_checkpoint(self, body):
        if self.local_registry is None:
            raise Refused(409, 'No checkpoint registry configured')
        with self._registry_lock:
            try:
                local_checkpoints.remove(body.get('checkpoint_id'), self.local_registry)
            except KeyError:
                raise Refused(404, 'Unknown checkpoint') from None
            self._ckpt_cache = None
        return {'removed': body.get('checkpoint_id')}

    def config(self) -> dict:
        profiles = {k: {**{kk: v[kk] for kk in ('state', 'action', 'state_dim', 'action_dim')},
                        'joints': list(v['joints']), 'label': PROFILE_LABELS.get(k, k)}
                    for k, v in pr.PROFILES.items()}
        return {'profiles': profiles,
                'default_profile': pr.PROFILE,
                'stick_homing_blocked': pr.stick_blocked(pr.load_hw(self.hardware_map)),
                'default_prompt': pr.DEFAULT_PROMPT,
                'group_prompts': dict(TASKS),
                'known_prompts': sorted(pr.known_prompts()),
                'defaults': {'chunk': 8, 'hz': pr.POLICY_HZ, 'max_seconds': 31.1, 'start_pose': 'none'},
                'twin_cameras': list(TWIN_CAMERAS),
                'pi_host': self.pi.base, 'pi_hosts': self.pi.spec, 'bridge': describe_bridge(self.bridge), 'token_file_present':
                    bool(self.token_file and self.token_file.is_file()),
                'runs_dir': str(self.runs_dir), 'python': sys.executable, 'arm_ttl_s': ARM_TTL_S,
                'backend': 'bridge', 'modes': ['DRY RUN', 'LIVE'], 'camera': 'live (Pi)', 'pace': self.pace,
                'stop_timeout_s': self.stop_timeout_s}

    def list_runs(self, limit=100) -> list[dict]:
        if not self.runs_dir.is_dir():
            return []
        out = []
        for d in sorted((p for p in self.runs_dir.iterdir() if p.is_dir() and RUN_ID_RE.match(p.name)),
                        key=lambda p: p.name, reverse=True)[:limit]:
            try:
                out.append(run_summary(d))
            except Exception as ex:
                out.append({'id': d.name, 'error': f'unreadable: {ex}'})
        return out

    def run_file(self, run_id: str, rel: str) -> Path | None:
        if not RUN_ID_RE.match(run_id) or not FILE_RE.match(rel):
            return None
        d = (self.runs_dir / run_id).resolve()
        f = (d / rel).resolve()
        if not f.is_relative_to(self.runs_dir.resolve()) or not f.is_file():
            return None
        return f

    def close(self):
        self._closing.set()
        job = self.job
        if job is not None and job.running:
            if job.live:
                self.emergency_stop('dashboard shutting down during a live job')
            else:
                self.stop('dashboard shutting down')
            if job.thread is not None:
                job.thread.join(timeout=10)
        self.twin.close()
