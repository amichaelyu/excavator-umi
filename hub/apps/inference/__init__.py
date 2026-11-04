"""Inference as a hub app: run the Lambda-served π0.5 policies on the real excavator, calibrate it,
E-STOP. The service is an :class:`hub.apps.inference.core.Inference` (module map and safety rules in its docstring);
GPIO goes through the hub's gpio app (its ``GpioBridge`` service), looked up on every use so a restarted
gpio app is picked up and a missing one reads as "bridge unavailable" instead of failing the dashboard.
This module has the routes only: the same status codes and JSON as the old standalone HTTP handler. A
``Refused`` becomes ``{"error": ...}`` with its status; any other server error first triggers the
emergency stop, then answers 500.

Endpoints (under /api/inference/; GET unless noted; POST bodies are JSON with
``Content-Type: application/json``):

    events                    SSE stream: state, log, progress (homing / start pose / pulses / infer, ~4 Hz),
                              pulses, homing, start_pose, step, estimate, sensors, job_end, estop
    state                     job / arming state snapshot
    config                    profile, prompts (policy.configs.schema.TASKS), defaults, hosts
    sensors[?refresh=1]       last Pi status (IMU accel/gyro, bucket pitch, camera fps/age) + bridge health,
                              and ``twin_state`` (as twin_state; so also on every SSE sensors event, ~5 Hz)
    camera.jpg                the Pi camera (/preview.jpg proxied with the umi_access cookie)
    twin_frame[?camera=overview_close|overview|side]  small "estimated pose" view: the twin drawn at
                              the runner's real-space joint estimate (assumed in DRY RUN)
    twin_state                live joint state for the twin panel: per joint (slew, boom, stick, bucket)
                              rad / deg, source (imu | welded | locked | dead_reckoned | assumed) and age,
                              live / stale overall, the link IMUs and their arm calibration (inference.twin.TwinState)
    twin_state.jpg[?camera=overview_close|overview|side]  the twin drawn at twin_state
                              (X-Twin-State: live | stale)
    observation.jpg           the observation as last sent (X-Observation: sent; native frame), or the live
                              camera letterboxed to 224x224 like the policy server when no run has sent one (preview)
    checkpoints[?refresh=1]   the Lambda-served checkpoints in configs/hub/local_checkpoints.json
    POST checkpoints/add {label, gateway_url, prompt}  probe a Lambda CLOAK server's contract, then register it
    POST checkpoints/remove {checkpoint_id}             unregister one
    runs, runs/<id>, runs/<id>/files/<path>  records under artifacts/inference-runs/ (Range)
    POST arm                  one click: single-use live token for the next job (120 s); POST disarm
    POST zero_slew            no-op: the slew is locked at 0 (facing forward)
    POST policy/start         {profile, gateway_url, prompt, chunk, hz, max_seconds, max_steps?, start_pose?,
                               imu?, home?, live?, arm_token?}
    POST home                 {live?, arm_token?, home_stick?, face_forward?}; face_forward (Quick home) first turns the base back to
                              facing forward (IMU slew estimate + calibrated turn rate), then every arm joint
                              into its hard stop (stick up, boom up, bucket in)
    POST calibrate            the hardware calibration (digital_twin.hardware.calibrate, link IMUs) in-process:
                              {mode: sim} = simulated arm, nothing sent, map untouched (no arming);
                              {mode: live, arm_token, prereqs_ok} = --live --i-understand-open-loop --yes, merged
                              into the hardware map; optional visits (1-5, default 2), slew (base turn, default
                              true). Every line goes to the inference log; Stop = the job's stop event (the routine stops
                              at the next press / wait and releases), then release-all
    calibration               summary line, what the hardware map holds (arm IMUs, ranges, speeds, dead times, base
                              turn, pin signs), the last live and simulated runs (calibration.json), prerequisites
    calibration/report        the newest run's log.txt and summary
    POST stop                 stop the current job: cooperative cancel (between pulses, in the pulse
                              heartbeat, around grab/infer), then a hard timeout abandons the worker
    POST estop                EMERGENCY STOP: abort the job and release every pin through the GPIO bridge
    POST reset                e-stop + disarm, then the hub restarts the inference app
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import anyio.to_thread
from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from hub.app import rel

KEEPALIVE_S = 5.0           # SSE comment while idle, so a closed page is noticed (and the deadman counts right)
SSE_POLL_S = 0.05
BODY_MAX = 64 << 10
MEDIA_TYPES = {'.mp4': 'video/mp4', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
               '.json': 'application/json', '.csv': 'text/csv; charset=utf-8', '.txt': 'text/plain; charset=utf-8'}


def start(hub, pi_host=None, token_file=None, hardware_map=None, runs_dir=None, calibration_root=None,
          stop_timeout_s=None):
    """Options left empty keep the Inference defaults (core / pi / calibration / digital_twin.paths)."""
    from hub.apps.inference.core import Inference
    if hub is not None:
        hub.service('gpio', wait=30)                    # the bridge starts first; wait while it is still starting
    paths = {'token_file': token_file, 'hardware_map': hardware_map, 'runs_dir': runs_dir, 'cal_root': calibration_root}
    kw = {k: str(rel(v)) for k, v in paths.items() if v}
    if pi_host:
        kw['pi_host'] = pi_host
    if stop_timeout_s:
        kw['stop_timeout_s'] = stop_timeout_s
    inference = Inference(bridge=(lambda: hub.service('gpio')) if hub is not None else None, **kw)
    inference.hub = hub
    return inference


# ------------------------------------------------------------------ helpers
def _json(obj, status: int = 200, headers=None) -> Response:
    from hub.apps.inference.util import dumps
    return Response(dumps(obj), status, headers=headers, media_type='application/json')


def _hval(v) -> str:
    """A header-safe value (latin-1, one line)."""
    return ' '.join(str(v).split()).encode('latin-1', 'replace').decode('latin-1')


async def json_body(request: Request) -> dict:
    """The POST body as a JSON object (415 / 413 / 400 like the old handler)."""
    from hub.apps.inference.util import Refused
    ctype = (request.headers.get('content-type') or '').split(';')[0].strip().lower()
    if ctype != 'application/json':
        raise Refused(415, 'Content-Type must be application/json')
    try:
        n = int(request.headers.get('content-length') or 0)
    except ValueError:
        raise Refused(400, 'bad Content-Length') from None
    if n > BODY_MAX:
        raise Refused(413, 'body too large')
    raw = await request.body()
    if len(raw) > BODY_MAX:
        raise Refused(413, 'body too large')
    try:
        body = json.loads(raw or b'{}')
    except (UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise Refused(400, f'invalid JSON: {ex}') from None
    if not isinstance(body, dict):
        raise Refused(400, 'body must be a JSON object')
    return body


def send_file(f: Path, request: Request) -> Response:
    """A run file with single-range support (run videos seek)."""
    from hub.apps._common import range_response
    return range_response(f, request.headers.get('range'), request.method == 'HEAD',
                          MEDIA_TYPES.get(f.suffix.lower(), 'application/octet-stream'))


def _route_class(get):
    """Every inference route: ``Refused`` -> its status + {"error"}; any other server error -> emergency stop,
    then 500 (as the old handler); responses default to no-store / nosniff."""
    from hub.apps.inference.util import Refused

    class OpsRoute(APIRoute):
        def get_route_handler(self):
            handler = super().get_route_handler()

            async def run(request: Request) -> Response:
                try:
                    resp = await handler(request)
                except Refused as ex:
                    resp = JSONResponse({'error': str(ex)}, ex.status)
                except (StarletteHTTPException, RequestValidationError):
                    raise
                except Exception as ex:
                    msg = f'{type(ex).__name__}: {ex}'
                    try:
                        inference = get()
                        await anyio.to_thread.run_sync(
                            inference.emergency_stop, f'server error on {request.method} {request.url.path}: {msg}')
                    except Exception:
                        pass
                    resp = JSONResponse({'error': msg}, 500)
                resp.headers.setdefault('Cache-Control', 'no-store')
                resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
                return resp
            return run
    return OpsRoute


def _restart(hub, inference, delay_s: float = 0.5):
    """Reset: after the response went out, the hub stops this app (closing Inference) and starts a fresh one."""
    app_id = next((a.id for a in hub.apps.values() if a.service is inference), 'inference')

    def run():
        time.sleep(delay_s)
        hub.stop(app_id)
        hub.start(app_id)
    threading.Thread(target=run, daemon=True, name='inference-reset').start()


# ------------------------------------------------------------------ routes
def routes(get) -> APIRouter:
    from hub.apps.inference.calibration import cal_report, calibration_status
    from hub.apps.inference.core import RUN_ID_RE
    from hub.apps.inference.jobs import run_summary
    from hub.apps.inference.twin import TWIN_CAMERAS, TwinUnavailable
    from hub.apps.inference.util import clean
    r = APIRouter(route_class=_route_class(get))
    Body = Depends(json_body)

    def camera_arg(camera: str):
        if camera not in TWIN_CAMERAS:
            return _json({'error': f'camera must be one of {list(TWIN_CAMERAS)}'}, 400)
        return None

    # ---- GET
    @r.get('/events')
    async def events():
        inference = get()

        async def gen():
            q = inference.bus.subscribe()
            try:
                hello = await anyio.to_thread.run_sync(
                    lambda: {'id': 0, 'type': 'hello', 't': time.time(),
                             'data': clean({'state': inference.state(), 'sensors': inference.snapshot_sensors()})})
                yield f'data: {json.dumps(hello)}\n\n'
                last = time.monotonic()
                while not inference._closing.is_set():
                    try:
                        ev = q.get_nowait()
                    except queue.Empty:
                        if time.monotonic() - last >= KEEPALIVE_S:
                            last = time.monotonic()
                            yield ': keepalive\n\n'
                        await asyncio.sleep(SSE_POLL_S)
                        continue
                    last = time.monotonic()
                    yield f"id: {ev['id']}\ndata: {json.dumps(ev, allow_nan=False)}\n\n"
            finally:
                inference.bus.unsubscribe(q)
        return StreamingResponse(gen(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    @r.get('/state')
    def state():
        return _json(get().state())

    @r.get('/config')
    def config():
        return _json(get().config())

    @r.get('/sensors')
    def sensors(refresh: str = '0'):
        inference = get()
        return _json(inference.refresh_sensors() if refresh == '1' else inference.snapshot_sensors())

    @r.get('/camera.jpg')
    def camera(mask: str = '0'):
        status, body, ctype, cloak = get().camera_frame(mask=mask == '1')
        return Response(body, status, media_type=ctype or 'application/json',
                        headers={'X-Cloak': _hval(cloak)} if cloak else None)

    @r.get('/twin_frame')
    def twin_frame(camera: str = 'overview_close'):
        if (bad := camera_arg(camera)) is not None:
            return bad
        try:
            jpg = get().twin_frame(camera)
        except TwinUnavailable as ex:
            return _json({'error': str(ex)}, 503)
        return Response(jpg, media_type='image/jpeg')

    @r.get('/twin_state')
    def twin_state():
        return _json(get().twin_state())

    @r.get('/twin_state.jpg')
    def twin_state_jpg(camera: str = 'overview_close'):
        if (bad := camera_arg(camera)) is not None:
            return bad
        try:
            jpg, st = get().twin_state_frame(camera)
        except TwinUnavailable as ex:
            return _json({'error': str(ex)}, 503)
        return Response(jpg, media_type='image/jpeg', headers={'X-Twin-State': 'live' if st['live'] else 'stale'})

    @r.get('/observation.jpg')
    def observation():
        jpg, src = get().observation()
        return Response(jpg, media_type='image/jpeg', headers={'X-Observation': _hval(src)})

    @r.get('/calibration')
    def calibration():
        inference = get()
        return _json(calibration_status(inference.hardware_map, inference.cal_root))

    @r.get('/calibration/report')
    def calibration_report():
        return _json(cal_report(get().cal_root))

    @r.get('/checkpoints')
    def checkpoints(refresh: str = '0'):
        return _json(get().checkpoints(refresh == '1'))

    @r.get('/runs')
    def runs():
        inference = get()
        return _json({'runs': inference.list_runs(), 'dir': str(inference.runs_dir)})

    @r.get('/runs/{run_id}')
    def run(run_id: str):
        inference = get()
        if RUN_ID_RE.match(run_id) and (inference.runs_dir / run_id).is_dir():
            return _json(run_summary(inference.runs_dir / run_id, full=True))
        return _json({'error': 'not found'}, 404)

    @r.api_route('/runs/{run_id}/files/{path:path}', methods=['GET', 'HEAD'])
    def run_file(run_id: str, path: str, request: Request):
        f = get().run_file(run_id, path)
        if f is None:
            return _json({'error': 'not found'}, 404)
        return send_file(f, request)

    # ---- POST
    @r.post('/estop')
    def estop():                                        # accept any body: never block an e-stop on parsing
        return _json(get().emergency_stop('EMERGENCY STOP pressed on the dashboard'))

    @r.post('/arm')
    def arm(body: dict = Body):
        return _json(get().arm())

    @r.post('/disarm')
    def disarm(body: dict = Body):
        get().disarm()
        return _json({'armed': False})

    @r.post('/zero_slew')
    def zero_slew(body: dict = Body):                   # the base faces forward now
        return _json({'slew_rad': get().live_twin.zero_slew()})

    @r.post('/reset')
    def reset(body: dict = Body):
        """Stop everything, release the pins, then the hub restarts this app (a fresh Inference)."""
        inference = get()
        out = inference.emergency_stop('reset pressed on the dashboard')
        inference.disarm()
        hub = getattr(inference, 'hub', None)
        if hub is not None:
            _restart(hub, inference)
        return _json({'reset': True, 'restarting': hub is not None, 'stop': out})

    @r.post('/stop')
    def stop(body: dict = Body):
        return _json(get().stop('stop pressed on the dashboard'))

    @r.post('/checkpoints/add')
    def checkpoint_add(body: dict = Body):
        return _json(get().add_checkpoint(body))

    @r.post('/checkpoints/remove')
    def checkpoint_remove(body: dict = Body):
        return _json(get().remove_checkpoint(body))

    @r.post('/policy/start')
    def policy_start(body: dict = Body):
        return _json(get().start_policy(body))

    @r.post('/home')
    def home(body: dict = Body):
        return _json(get().start_home(body))

    @r.post('/calibrate')
    def calibrate(body: dict = Body):
        return _json(get().start_calibrate(body))

    return r
