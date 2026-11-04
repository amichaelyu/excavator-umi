"""Twin replay viewer: episodes next to their MuJoCo twin replay, plus success labelling.

Service: :class:`TwinViewer` (an ``EpisodeIndex`` over the policy configs named by the app's ``sources``
option, rescanned per request, the ``LabelStore``, the live-camera proxy settings and the offscreen
``SimRenderer``). Domain code: hub/apps/twin_viewer/core.py (time mapping, episodes, labels),
hub/apps/_common.py (live-camera fetch, HTTP Range). The replays are the labels stage's output
(``python -m policy labels <config>``); the viewer never makes them.

    GET  index                    episodes, cameras, live settings, plus the index's info()
    GET  path_error_summary       IK path / actuated twin vs SLAM path pooled over all episodes
    GET  episode?id=&dz=          per-episode arrays (dz: path height offset, +-MAX_DZ_M m)
    GET  sim_frame?episode=&t=&camera=&mode=&dz=   offscreen MuJoCo JPEG of the twin at t
    GET  live_frame               latest live JPEG via the server (204 no new frame, 502 unreachable)
    GET  labels                   the labels file (soil_success_labels.v2) + resolved_prompts
    GET  labels/summary           counts, verdicts, readiness
    POST labels/<episode>         {verdict?, prompt?}      POST labels/meta  {default_prompt?, group_prompts?}
    GET  media/<episode>/<file>   replay.mp4, joints.png, overlay.mp4, trajectory.png, video.mp4 (Range)

Errors are ``{"error": ...}``.
"""
from __future__ import annotations

import json
import math
import mimetypes
import threading
import time
from pathlib import Path

import anyio.to_thread
from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from hub.apps._common import fetch_live_frame, range_response
from hub.apps.twin_viewer import core as wv
from hub.app import rel

NO_STORE = {'Cache-Control': 'no-store'}


class TwinViewer:
    """``index``: an EpisodeIndex (rescanned per request) or a fixed {id: episode} dict."""

    def __init__(self, index, labels: wv.LabelStore, live_url: str | None = None, live_token: str = ''):
        self.index = index
        self.labels = labels
        self.renderer = wv.SimRenderer()
        self.live_url, self.live_token = None, ''
        self._live_lock = threading.Lock()
        self._live_fail = (0.0, b'')           # (monotonic time, error body) of the last failed fetch
        if live_url:
            url, tok = wv.validate_live_url(live_url)
            self.live_url, self.live_token = url, live_token or tok

    @property
    def episodes(self) -> dict:
        return self.index.get() if isinstance(self.index, wv.EpisodeIndex) else self.index

    def info(self) -> dict:
        return {'episodes': len(self.episodes)}

    def index_payload(self) -> dict:
        body = {'episodes': [e.index_entry() for e in self.episodes.values()],
                'cameras': list(wv.CAMERAS), 'modes': list(wv.MODES),
                'live': {'configured': self.live_url is not None,
                         'url': wv.redact_url(self.live_url) if self.live_url else None,
                         'default': wv.DEFAULT_LIVE_URL}}
        info = getattr(self.index, 'info', None)        # index extras (EpisodeIndex: the source picker)
        if callable(info):
            body.update(info())
        return body

    def episode_groups(self) -> dict[str, str]:
        """{recording id: group} as in the index ('dig' unless episodes.csv gives a label). Labels are
        per recording: the reconstructions of one recording share them."""
        return {wv.recording_of(i, e): (getattr(e, 'label', '') or 'dig') for i, e in self.episodes.items()}

    def label_exempt(self) -> set[str]:
        """Recordings only in sources without a cohort selection yet (no split to check them against)."""
        if not isinstance(self.index, wv.EpisodeIndex):
            return set()
        eps, free = self.episodes, self.index.split_exempt()
        return ({wv.recording_of(i, eps[i]) for i in free if i in eps}
                - {wv.recording_of(i, e) for i, e in eps.items() if i not in free})

    def live_frame(self) -> tuple[int, bytes, str]:
        """Proxy one live frame. After a failure, the cached error is returned for LIVE_RETRY_S:
        unreachable .local names can block for seconds in DNS/mDNS, and the page polls often."""
        with self._live_lock:
            failed_at, err = self._live_fail
            if time.monotonic() - failed_at < wv.LIVE_RETRY_S:
                return 502, err, 'application/json'
            status, body, ctype = fetch_live_frame(self.live_url, self.live_token)
            self._live_fail = (time.monotonic(), body) if status == 502 else (0.0, b'')
            return status, body, ctype

    def close(self):
        self.renderer.close()


def start(hub, sources, labels_file, live_url=None, live_token_file=None) -> TwinViewer:
    """``sources``: policy configs (``"<config>"`` or ``{"config": "<config>"}``, core.config_source) whose
    reconstruct / labels outputs are listed; ``labels_file``: the verdicts file (repo-relative). Both are
    required options (configs/hub/apps.json)."""
    index = wv.EpisodeIndex(wv.config_sources(sources))
    index.get()                                   # warm the cache before serving
    labels = wv.LabelStore(rel(labels_file), index.split_files())
    token = rel(live_token_file).read_text().strip() if live_token_file else ''
    return TwinViewer(index, labels, live_url, token)


# ---------------------------------------------------------------------------------------- routes

class _Route(APIRoute):
    """Any unexpected exception -> 500 ``{"error": "Type: message"}`` (as the old handler did)."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def wrapped(request):
            try:
                return await handler(request)
            except (StarletteHTTPException, RequestValidationError):
                raise
            except Exception as ex:
                return JSONResponse({'error': f'{type(ex).__name__}: {ex}'}, 500, headers=NO_STORE)
        return wrapped


def _json(body, status: int = 200) -> Response:
    return Response(json.dumps(body), status, media_type='application/json',
                    headers={**NO_STORE, 'X-Content-Type-Options': 'nosniff'})


def _dz(value: str) -> float:
    dz = wv.parse_dz(value)
    if dz is None:
        raise HTTPException(400, f'dz must be a number within +-{wv.MAX_DZ_M} m')
    return dz


def _media(f: Path, range_header: str | None, head: bool) -> Response:
    return range_response(f, range_header, head, mimetypes.guess_type(f.name)[0] or 'application/octet-stream',
                          {'X-Content-Type-Options': 'nosniff'})


def routes(get) -> APIRouter:
    r = APIRouter(route_class=_Route)
    read = lambda path: r.api_route(path, methods=['GET', 'HEAD'])    # noqa: E731

    @read('/index')
    def index():
        return _json(get().index_payload())

    @read('/path_error_summary')
    def path_error_summary():
        return _json(wv.overall_path_error(list(get().episodes.values())))

    @read('/episode')
    def episode(id: str = '', dz: str = '0'):
        ep = get().episodes.get(id)
        if ep is None:
            raise HTTPException(404, 'unknown episode')
        return _json(ep.detail(_dz(dz)))

    @read('/sim_frame')
    def sim_frame(episode: str = '', t: str = '', camera: str = 'overview_close', mode: str = 'twin', dz: str = '0'):
        v = get()
        ep = v.episodes.get(episode)
        if ep is None:
            raise HTTPException(404, 'unknown episode')
        if camera not in wv.CAMERAS:
            raise HTTPException(400, f'camera must be one of {list(wv.CAMERAS)}')
        if mode not in wv.MODES or not ep.has_mode(mode):
            raise HTTPException(400, f'mode must be one of {list(wv.MODES)}')
        try:
            tt = float(t)
        except ValueError:
            raise HTTPException(400, 't must be a number (episode seconds)') from None
        if not math.isfinite(tt) or tt < 0 or tt > ep.duration + 1e-6:
            raise HTTPException(400, f't must be within [0, {ep.duration:.3f}] s')
        jpg = v.renderer.render(ep, tt, camera, mode, _dz(dz))
        return Response(jpg, media_type='image/jpeg', headers=NO_STORE)

    @read('/live_frame')
    def live_frame():
        v = get()
        if v.live_url is None:
            raise HTTPException(404, 'no --live-url configured')
        status, body, ctype = v.live_frame()
        if status == 204:
            return Response(status_code=204, headers=NO_STORE)
        return Response(body, status, media_type=ctype or 'application/json', headers=NO_STORE)

    @read('/labels')
    def labels():
        v = get()
        try:
            data = v.labels.payload(v.episode_groups())
        except ValueError as ex:
            raise HTTPException(500, f'labels file is not valid JSON: {ex}') from None
        return _json(data)

    @read('/labels/summary')
    def labels_summary():
        v = get()
        return _json(v.labels.summary(v.episode_groups()))

    @r.post('/labels/{name:path}')
    async def labels_post(name: str, request: Request):
        v = get()
        body = await _read_json(request)
        return await anyio.to_thread.run_sync(_labels_write, v, name, body)

    @read('/media/{rest:path}')
    def media(rest: str, request: Request):
        parts = rest.split('/')
        if len(parts) != 2:
            raise HTTPException(404, 'not found')
        ep = get().episodes.get(parts[0])
        f = ep.media().get(parts[1]) if ep else None
        if f is None:
            raise HTTPException(404, 'not found')
        return _media(f, request.headers.get('range'), request.method == 'HEAD')

    return r


async def _read_json(request: Request) -> dict:
    """Parsed JSON object body, or 400."""
    ctype = (request.headers.get('content-type') or '').split(';')[0].strip().lower()
    if ctype != 'application/json':
        # also keeps cross-site HTML forms (no CORS preflight) from writing labels
        raise HTTPException(400, 'Content-Type must be application/json')
    try:
        n = int(request.headers.get('content-length') or 0)
    except ValueError:
        raise HTTPException(400, 'bad Content-Length') from None
    if n > wv.LABELS_BODY_MAX:
        raise HTTPException(400, 'body too large')
    raw = await request.body()
    if not raw:
        raise HTTPException(400, 'missing JSON body')
    if len(raw) > wv.LABELS_BODY_MAX:
        raise HTTPException(400, 'body too large')
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise HTTPException(400, f'invalid JSON: {ex}') from None
    if not isinstance(body, dict):
        raise HTTPException(400, 'body must be a JSON object')
    return body


def _labels_write(v: TwinViewer, name: str, body: dict) -> Response:
    try:
        if name == 'meta':
            data = v.labels.set_meta(body)
            out = {k: data[k] for k in ('default_prompt', 'group_prompts')}
        elif name == 'summary' or not wv.EPISODE_RE.match(name):
            raise HTTPException(404, 'not found')
        else:
            out = {'episode': name, 'entry': v.labels.upsert(name, body, v.episode_groups(), v.label_exempt())}
    except LookupError as ex:
        raise HTTPException(404, str(ex).strip("'")) from None
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from None
    out['summary'] = v.labels.summary(v.episode_groups())
    return _json(out)
