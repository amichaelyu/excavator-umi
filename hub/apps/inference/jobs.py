"""One inference job and its worker: the :class:`Job` record (one directory under artifacts/inference-runs/), the
:class:`DashboardRunner` (policy.runner.PolicyRunner streaming to the page), the runner CLI filter and
the run-record summaries the runs pages list."""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from hub.apps.inference.util import load_json
from digital_twin.paths import REPO
from digital_twin.policy import runner as pr

if TYPE_CHECKING:
    from hub.apps.inference.core import Inference


class StopRequested(pr.RunnerError):
    """Raised inside the runner thread when the page asked to stop."""


def filter_runner_argv(argv: list[str], parser=None) -> tuple[list[str], list[str]]:
    """Keep only the options the runner's CLI accepts right now (it is edited concurrently: flags come
    and go). Returns (kept argv, dropped option names)."""
    try:
        parser = parser or pr.build_parser()
        takes = {o: a.nargs != 0 for a in parser._actions for o in a.option_strings}
    except Exception:
        return list(argv), []
    out, dropped, i = [], [], 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith('--'):
            known = tok in takes
            has_val = takes.get(tok, i + 1 < len(argv) and not argv[i + 1].startswith('--'))
            span = argv[i:i + (2 if has_val else 1)]
            (out.extend(span) if known else dropped.append(tok))
            i += len(span)
        else:
            out.append(tok); i += 1
    return out, dropped


def run_summary(d: Path, full: bool = False) -> dict:
    record = load_json(d / 'job.json', {}) or {}
    run = load_json(d / 'run.json', None)
    files = sorted(str(p.relative_to(d)) for p in d.iterdir() if p.is_file())
    n_frames = len(list((d / 'frames').glob('*.jpg'))) if (d / 'frames').is_dir() else 0
    out = {'id': d.name, 'kind': record.get('kind'), 'status': record.get('status'),
           'started_at': record.get('started_at'), 'ended_at': record.get('ended_at'), 'error': record.get('error'),
           'params': record.get('params'), 'files': files, 'n_frames': n_frames}
    if run is not None:
        s = {k: v for k, v in run.items() if k not in ('events', 'homing', 'server_metadata')} if not full else run
        out['summary'] = s
    elif record.get('summary') is not None:
        out['summary'] = record['summary'] if full else {k: v for k, v in record['summary'].items() if k != 'homing'}
    if full:
        out['job'] = record
    return out


class Job:
    def __init__(self, kind: str, params: dict, directory: Path):
        self.kind, self.params, self.dir = kind, params, directory
        self.id = directory.name
        self.stop = threading.Event()
        self.estopped = False
        self.status = 'starting'
        self.error = None
        self.summary = None
        self.runner = None
        self.thread = None
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at = None
        self.steps = 0
        self.finalized = False
        self.final_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.status in ('starting', 'running')

    def info(self) -> dict:
        return {'id': self.id, 'kind': self.kind, 'status': self.status, 'error': self.error,
                'started_at': self.started_at, 'ended_at': self.ended_at, 'steps': self.steps,
                'params': self.params, 'dir': str(self.dir.relative_to(REPO) if self.dir.is_relative_to(REPO) else self.dir)}


class DashboardRunner(pr.PolicyRunner):
    """policy.runner.PolicyRunner (bridge backend, real Pi camera; every run moves the machine) streaming to the
    dashboard: log lines, the pulses sent, start-pose / pulse / infer progress, per-step records, the
    observation exactly as sent, and the real-space joint estimate with the runner's label ("arm IMUs (measured)").

    Stopping uses the runner's own cooperative ``stop()`` (checked between pulses, every 50 ms while
    a pulse is waited out, around the bounded ``infer()``); this class also checks the job's stop
    event in the pulse heartbeat. Inference.stop adds a hard timeout for anything that never returns."""

    PROGRESS_EVERY_S = 0.25

    def __init__(self, args, inference: 'Inference', job: Job, **kw):
        super().__init__(args, log=lambda s: inference.log(job, s), **kw)
        self.inference, self.job = inference, job
        self.skipped_total = 0
        self.est_label = self.estimate_label                  # the runner's label; set again after setup
        self._phase = {'phase': 'setup', 'step': None, 't0': time.monotonic(), 'planned_s': None}
        self._last_progress = 0.0

    # ---- cancel + progress (worker thread only, except _phase_event)
    def _job_stop(self):
        if self.job.stop.is_set():
            if hasattr(self, 'stop_event'):
                self.stop_event.set()
            raise StopRequested('stopped from the dashboard' + (' (EMERGENCY STOP)' if self.job.estopped else ''))

    def _phase_event(self, phase, step=None, planned_s=None, q=None):
        self._phase = {'phase': phase, 'step': step, 't0': time.monotonic(), 'planned_s': planned_s}
        self._last_progress = time.monotonic()
        self.inference.bus.publish('progress', {'job': self.job.id, 'phase': phase, 'step': step, 'elapsed_s': 0.0,
                                          'planned_s': planned_s, 'q': q, 'estimate': self.est_label})

    def _estimate(self):
        try:
            return self.estimator.as_dict(self.now()) if self.estimator is not None else None
        except Exception:
            return None

    def _progress(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_progress < self.PROGRESS_EVERY_S:
            return
        self._last_progress = now
        ph, q = self._phase, self._estimate()
        self.inference.bus.publish('progress', {'job': self.job.id, 'phase': ph['phase'], 'step': ph['step'],
                                          'elapsed_s': round(now - ph['t0'], 2), 'planned_s': ph['planned_s'],
                                          'q': q, 'estimate': self.est_label})
        if q is not None:
            self.inference.set_estimate(q, self.est_label, publish=False)

    def _beat(self):                                   # executor heartbeat (every 50 ms while pacing) + loop beats
        super()._beat()
        self._job_stop()
        self._progress()

    def _publish_estimate(self):
        q = self._estimate()
        if q is not None:
            self.inference.set_estimate(q, self.est_label)

    # ---- runner hooks
    def setup(self):
        self._phase_event('setup')
        super().setup()
        self.est_label = self.estimate_label                  # 'arm IMUs (measured)' once the arm IMUs are set up
        if self.policy is not None:
            orig = self.policy.infer

            cloak = getattr(self, 'summary', {}).get('cloak')  # pi05_tabletop_excavator: server mode, needs_mask

            def infer(image, state, *mask):            # runs inside the runner's bounded worker thread
                self.inference.set_observation(image, self.job.id, state, cloak=cloak, mask=mask[0] if mask else None)
                self._phase_event('infer', self._phase.get('step'))
                return orig(image, state, *mask)       # pi05_tabletop_excavator also passes the robot mask
            self.policy.infer = infer
        self._publish_estimate()

    def execute(self, pulses, phase, step=None):
        self._job_stop()
        gap = self.hw.get('defaults', {}).get('gap_ms', 150) if self.args.gap_ms is None else self.args.gap_ms
        planned = sum(p['duration_ms'] for p in pulses) / 1000.0 + gap / 1000.0 * max(0, len(pulses) - 1)
        self._phase_event(phase, step, round(planned, 2), self._estimate())
        self.inference.bus.publish('pulses', {'job': self.job.id, 'phase': phase, 'step': step,
                                        'pulses': [{k: p.get(k) for k in ('t_start_s', 'joint', 'direction',
                                                                          'duration_ms', 'gpio', 'label')}
                                                   for p in pulses]})
        res = super().execute(pulses, phase, step)
        self._publish_estimate()
        return res

    def move_to(self, q_target, phase='start_pose'):
        self._job_stop()
        self._phase_event(phase)
        out = super().move_to(q_target, phase)
        self.inference.bus.publish('start_pose', {'job': self.job.id, 'phase': phase, 'target_real_rad': q_target,
                                            'n_pulses': len(out['pulses']), 'estimate': self.est_label})
        return out

    def step(self, k):
        self._job_stop()
        self._phase_event('step', k)
        rec = super().step(k)
        self.skipped_total += int(rec.get('n_skipped') or 0)
        self.job.steps = k + 1
        est_pitch = sum(rec.get(f'est_{j}') or 0.0 for j in ('boom', 'stick', 'bucket'))   # bucket pitch
        self.inference.bus.publish('step', {'job': self.job.id, **rec, 'skipped_total': self.skipped_total,
                                      'est_pitch_rad': est_pitch, 'estimate': self.est_label,
                                      'imu_minus_est_deg': None if rec.get('imu_pitch_rad') is None else
                                      math.degrees(rec['imu_pitch_rad'] - est_pitch)})
        self._publish_estimate()
        return rec
