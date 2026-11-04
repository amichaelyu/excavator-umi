"""Running the hardware calibration from the inference page: request validation and the calibrate job,
``digital_twin.hardware.calibrate.run`` called in-process on the job's worker thread with the hub's GPIO
bridge, the job's stop event and every line of its output in the inference log. ``sim`` runs the routine on the
simulated arm (nothing sent, the map untouched); ``live`` drives the machine and merges the result into
configs/hardware_map.json.

:class:`CalibrationJobs` is a mixin of :class:`hub.apps.inference.core.Inference`: it uses the Inference state (``hardware_map``,
``cal_root``, ``pi``, ``token_file``, ``bridge``, ``bus``, the test hook ``cal_run``) and its job plumbing
(``_bridge_only``, ``_check_live``, ``consume_arm``, ``_num``, ``_begin``, ``_finalize``, ``log``, ``publish_state``).
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from hub.apps.inference.calibration import calibration_status, last_run
from hub.apps.inference.pi import BRIDGE_UNAVAILABLE
from hub.apps.inference.util import Refused

if TYPE_CHECKING:
    from hub.apps.inference.jobs import Job

CLI = 'python -m digital_twin.hardware.calibrate'


class CalibrationJobs:
    """The calibration half of :class:`hub.apps.inference.core.Inference` (see the module docstring)."""

    def calibration_plan(self, body: dict) -> dict:
        """Validate a calibration request: ``{mode: sim | live, visits?, slew?, arm_token?, prereqs_ok?}``; returns
        the CLI argv (no side effects except consuming a LIVE arming token)."""
        self._bridge_only(body)
        mode = body.get('mode') or ('live' if body.get('live') else 'sim')
        if mode not in ('sim', 'live'):
            raise Refused(400, 'mode must be sim (simulated arm, nothing sent) or live')
        if mode == 'sim' and body.get('live'):
            raise Refused(400, 'a simulated run cannot be live')
        visits = self._num(body, 'visits', 2, 1, 5, int)
        slew = body.get('slew', True) is not False
        argv = ['--pi-host', self.pi.spec, '--hardware-map', str(self.hardware_map), '--artifacts', str(self.cal_root),
                '--visits', str(visits)]
        if self.token_file:
            argv += ['--token-file', str(self.token_file)]
        if not slew:
            argv.append('--no-slew')
        if mode == 'live':
            if not body.get('prereqs_ok'):
                self.consume_arm(body.get('arm_token'))
                raise Refused(400, 'confirm the manual prerequisites first (base facing forward, arm clear at full boom '
                                   'down with the stick and bucket out, link IMUs live)')
            self._check_live({**body, 'live': True})
            argv += ['--live', '--i-understand-open-loop', '--yes']
        return {'mode': mode, 'live': mode == 'live', 'visits': visits, 'slew': slew, 'argv': argv}

    def start_calibrate(self, body: dict) -> dict:
        plan = self.calibration_plan(body)
        if plan['live'] and self.bridge is None:
            raise Refused(503, f'live calibration refused: {BRIDGE_UNAVAILABLE}')
        job = self._begin('calibrate', {k: plan[k] for k in ('mode', 'visits', 'slew')}, plan['live'])
        job.params['command'] = f"{CLI} {' '.join(plan['argv'])}"
        job.thread = threading.Thread(target=self._run_calibrate, args=(job, plan['argv']), daemon=True,
                                      name='inference-calibrate')
        job.status = 'running'
        job.thread.start()
        self.publish_state()
        return job.info()

    # ---- the calibrate job (worker thread)
    def _run_calibrate(self, job: Job, argv: list[str]):
        status, error, rc = 'error', None, 1
        try:
            from digital_twin.hardware import calibrate
            self.log(job, f'$ {job.params["command"]}')
            args = calibrate.parse_args(argv)
            rc = (self.cal_run or calibrate.run)(args, log=lambda m: self.log(job, m), bridge=self.bridge, stop=job.stop)
            job.summary = {'returncode': rc, 'mode': job.params['mode'],
                           'run': last_run(self.cal_root, sim=job.params['mode'] == 'sim')}
            status = 'stopped' if job.stop.is_set() else ('done' if rc == 0 else 'error')
            if rc != 0 and not job.stop.is_set():
                error = f'calibration ended with code {rc} ({(job.summary["run"] or {}).get("status", "no run")})'
        except BaseException as ex:                          # SystemExit / KeyboardInterrupt too: the job must end
            status, error = ('stopped' if job.stop.is_set() else 'error'), f'{type(ex).__name__}: {ex}'
            self.log(job, f'calibration failed: {error}')
        finally:
            self._finalize(job, status=status, error=error)
            self.bus.publish('calibration', self.safe_calibration())

    # ---- views
    def safe_calibration(self):
        try:
            return calibration_status(self.hardware_map, self.cal_root)
        except Exception as ex:
            return {'error': f'{type(ex).__name__}: {ex}'}
