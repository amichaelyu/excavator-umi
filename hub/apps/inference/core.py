"""Inference dashboard for running the CLOAK π0.5 policies (served on Lambda) on the excavator: the :class:`Inference`
service of the hub app ``inference`` (routes in hub/apps/inference/__init__.py; ``.venv/bin/python -m hub``, then
http://127.0.0.1:8888/inference). Its options come from configs/hub/apps.json and configs/hub/local.json. The page
is the view module hub/web/views/inference/.

No simulation and no dry run: every job drives the machine through the GPIO bridge, and the observation is always
the real Pi camera.

:class:`Inference` holds the state: one background worker job at a time (a
:class:`digital_twin.policy.runner.PolicyRunner`, or ``digital_twin.hardware.calibrate.run``, both
in-process on a worker thread and handed the hub's :class:`digital_twin.hardware.gpio.GpioBridge`), a
sensor poller thread and an
:class:`~hub.apps.inference.events.EventBus` the router streams as Server-Sent Events. This module has the
policy job, its lifecycle (stop watchdog, emergency stop, deadman), the checkpoint
registry and the run records; the rest of the package:

    events.py            EventBus (SSE fan-out)
    pi.py                PiClient (Pi dashboard status + camera over the host list), bridge description
    twin.py              TwinRenderer (MuJoCo renders), TwinState (the twin panel's live joint state)
    jobs.py              Job, DashboardRunner (the policy runner streaming to the page), run summaries
    calibration.py       the calibration panel: what the hardware map holds, the last runs and their logs
    calibration_jobs.py  CalibrationJobs: the Inference mixin that validates and runs the calibration job
    live.py              LiveViews: the Inference mixin for the sensor poller, estimate, observation, twin / camera views
    util.py              Refused, JSON-safe values, JSON files, UTC stamps

Safety: the hub binds 127.0.0.1 and rejects foreign Host / Origin headers. Start starts the job on the machine
(no arming); a policy run is bounded by max-seconds and refused by the runner without live, calibrated arm IMUs.
Stop is a cooperative cancel with a hard timeout (the worker is abandoned and every pin released). Any unexpected
server error, a crashed worker, or losing every page connection during a job triggers the emergency stop (the
policy's Reset runs it too, then clears the last job and observation); a job that ends early gets a release-all.
"""
from __future__ import annotations

import io
import math
import os
import re
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
from digital_twin.policy import checkpoints as local_checkpoints
from digital_twin.policy import runner as pr
from policy.configs.schema import TASKS

RUNS_DIR = REPO / 'artifacts' / 'inference-runs'
STOP_TIMEOUT_S = 6.0                               # stop -> worker must end, else the job is abandoned
DEADMAN_S = 3.0                                    # job running with no page connected -> e-stop
RUN_ID_RE = re.compile(r'^[A-Za-z0-9_.-]{1,80}$')
FILE_RE = re.compile(r'^(frames/)?[A-Za-z0-9_.-]{1,120}$')
# The runner profile the dashboard offers: the 4-joint CLOAK π0.5 contract.
PROFILE_LABELS = {pr.PROFILE: 'π0.5 · standard tabletop excavator'}


class Inference(CalibrationJobs, LiveViews):
    """Dashboard state: one job at a time, sensors, estimate, observation, e-stop."""

    def __init__(self, pi_host=DEFAULT_PI_HOST, bridge=None, token_file=DEFAULT_TOKEN_FILE,
                 hardware_map=HARDWARE_MAP, runs_dir=RUNS_DIR,
                 poll: bool = True,
                 stop_timeout_s: float = STOP_TIMEOUT_S, pi_period_s: float = 0.2, bridge_period_s: float = 1.0,
                 cal_root=CAL_ROOT, local_registry=local_checkpoints.DEFAULT_REGISTRY):
        """``bridge``: the GPIO bridge (a :class:`digital_twin.hardware.gpio.GpioBridge`: ``status()``,
        ``release_all()``, ``trigger()``, ``session()``), or a zero-argument callable returning the current
        one (the hub's gpio app service, which may be restarted or absent: None = unavailable)."""
        self.bus = EventBus()
        self.pi = PiClient(pi_host, token_file)
        self._bridge = bridge
        self.token_file = Path(token_file).expanduser() if token_file else None
        self.hardware_map = Path(hardware_map)
        self.runs_dir = Path(runs_dir)
        self.stop_timeout_s = stop_timeout_s
        self.cal_root = Path(cal_root)
        self.local_registry = local_registry
        self.cal_run = None                                 # None: digital_twin.hardware.calibrate.run (tests inject fakes)
        self.pi_period_s, self.bridge_period_s = pi_period_s, bridge_period_s
        self.job: Job | None = None
        self._job_lock = threading.Lock()
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

    # ---- state
    def state(self) -> dict:
        job = self.job
        return {'job': job.info() if job else None, 'busy': bool(job and job.running),
                'q_est': self.q_est, 'q_source': self.q_source,
                'pi_host': self.pi.base, 'bridge': describe_bridge(self.bridge), 'viewers': self.bus.n_subscribers}

    def info(self) -> dict:
        """Short status for the app's entry in the hub's app listing (hub.app App.describe)."""
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
            raise Refused(400, 'policy rollout has no simulation: the backend is always the bridge (the machine '
                               'moves) and the observation is always the real Pi camera')

    def _begin(self, kind: str, params: dict) -> Job:
        with self._job_lock:
            if self.job is not None and self.job.running:
                raise Refused(409, f'a {self.job.kind} job is running ({self.job.id}); stop it first')
            job = Job(kind, params, self._new_dir(kind))
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

    def _common_argv(self, record: Path) -> list[str]:
        argv = ['--pi-host', self.pi.spec,
                '--hardware-map', str(self.hardware_map), '--record', str(record)]
        if self.token_file:
            argv += ['--token-file', str(self.token_file)]
        return argv

    def _finish_args(self, argv: list[str]):
        """Parse with the runner's own CLI. No calibration gate here (pin signs are not checked); the runner
        itself refuses a run without live, calibrated arm IMUs or without the max-seconds bound."""
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
        args.bridge = self.bridge                            # the hub's GpioBridge (takes precedence over bridge_host)
        if args.bridge is None:
            raise Refused(503, f'run refused: {BRIDGE_UNAVAILABLE}')
        return args

    def start_policy(self, body: dict) -> dict:
        profile = body.get('profile')
        if profile not in pr.PROFILES:
            raise Refused(400, f'profile must be one of {sorted(pr.PROFILES)}')
        # Only checkpoints in the local (Lambda) registry run; no job or worker creation may precede this check.
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
        if hz != pr.POLICY_HZ or chunk > 10:
            raise Refused(400, f'π0.5 requires joint targets, {pr.POLICY_HZ} Hz, and chunk 1..10')
        params = {'profile': profile, 'gateway_url': url, 'prompt': prompt, 'chunk': chunk, 'hz': hz, 'max_seconds': max_seconds, 'max_steps': max_steps,
                  'start_pose': start_pose, 'policy_space': entry['policy_space']}
        argv = ['--profile', profile, '--gateway-url', url, '--prompt', prompt,
                '--chunk', str(chunk), '--hz', str(hz), '--max-seconds', str(max_seconds),
                '--start-pose', start_pose]
        if max_steps:
            argv += ['--max-steps', str(max_steps)]
        job = self._begin('policy', params)
        try:
            args = self._finish_args(argv + self._common_argv(job.dir))
        except Exception:
            self._abandon(job)
            raise
        pr.validate_prompt(prompt, log=lambda s: self.log(job, s))
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
        self.log(job, f'{job.kind} job {job.id}: bridge {describe_bridge(getattr(args, "bridge", None))} (pulses '
                      f'are sent); camera {self.pi.base}; record -> {job.dir}')
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
        if job.status != 'done' and not job.estopped:
            self.bus.publish('log', {'job': job.id, 'line': 'job ended early: extra release-all via the bridge'})
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

    def emergency_stop(self, reason: str) -> dict:
        """Abort the job, release through the runner's own connection, and send release-all to the
        bridge over a fresh WebSocket (always, job or not: releasing is always safe)."""
        self.bus.publish('estop', {'reason': reason, 'phase': 'begin'})
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
                    ex.release_all()
                    out['runner_release'] = 'sent' if getattr(ex, 'session', None) is not None else 'no live connection'
                except Exception as e:
                    out['runner_release'] = f'failed: {e}'
            self._stop_watchdog(job)
            out['job'] = job.id
        out['bridge_release'] = self.bridge_release_all()
        self.bus.publish('estop', {**out, 'phase': 'done'})
        self.publish_state()
        return out

    def reset(self) -> dict:
        """The policy's Reset: back to the starting state without restarting the app. Emergency stop (abort
        the job, release every pin) and forget the last job and observation once the job has ended.
        The joint estimate and the twin panel are the machine's state and are kept."""
        out = self.emergency_stop('reset pressed on the dashboard')
        job = self.job
        cleared = job is None or not job.running
        if cleared:
            self.job = None
            self.obs = None
        self.publish_state()
        return {'reset': True, 'cleared': cleared, 'stop': out}

    # ---- deadman (run by the sensor poller, LiveViews._poll_loop)
    def _deadman(self):
        job = self.job
        if job is None or not job.running:
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
            self.emergency_stop(f'no dashboard page connected for {DEADMAN_S:.0f} s during a job')

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
                'default_prompt': pr.DEFAULT_PROMPT,
                'group_prompts': dict(TASKS),
                'known_prompts': sorted(pr.known_prompts()),
                'defaults': {'chunk': 8, 'hz': pr.POLICY_HZ, 'max_seconds': 31.1, 'start_pose': 'none'},
                'twin_cameras': list(TWIN_CAMERAS),
                'pi_host': self.pi.base, 'pi_hosts': self.pi.spec, 'bridge': describe_bridge(self.bridge), 'token_file_present':
                    bool(self.token_file and self.token_file.is_file()),
                'runs_dir': str(self.runs_dir), 'python': sys.executable,
                'backend': 'bridge', 'camera': 'live (Pi)',
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
            self.emergency_stop('dashboard shutting down during a job')
            if job.thread is not None:
                job.thread.join(timeout=10)
        self.twin.close()
