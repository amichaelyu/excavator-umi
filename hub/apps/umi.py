"""UMI data collection (hub app pi-dashboard): the hub relays to umi-data-collection/recorder running on the excavator Pi (hostname bucket).

The service is an HTTP relay, not a copy of the dashboard: hub ``/api/<id>/<path>`` goes to the Pi's
``/<path>`` (``api/status``, ``preview.jpg``, ``api/recordings/...``, ``stream.mjpg``, ``events``), with
the Pi's ``umi_access=<token>`` cookie added from the first token file that exists (default:
hosts.pi.token_file). Bodies stream through (MJPEG, SSE, chunked and file downloads), ``Range`` and the
dashboard's ``X-UMI-Action`` / ``X-Frame-*`` headers pass both ways. The Pi has no websockets, so there is
nothing else to relay.

``urls`` (default: hosts.pi.urls of configs/hub/local.json, policy.data.local_config) are tried in order; the
first that answers is used and remembered for ``PROBE_TTL`` seconds (a failed request forgets it at once).
``/api/<id>/status`` reports which one, and whether a token was found and accepted.

The page is the Pi's own ES modules (hub/web/data-collection/, served at /web/data-collection/; the Pi serves the same files), mounted
by hub/web/views/umi/view.js with ``ctx.url`` pointing here.
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import anyio
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from hub.app import rel
from policy.data import local_config

PROBE_TTL = 10.0          # seconds a reachable upstream is remembered
PROBE_TIMEOUT = 1.5       # per URL
HEADERS_TIMEOUT = 30.0    # until the Pi starts answering; the body then streams with no time limit
HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailer', 'trailers',
       'transfer-encoding', 'upgrade', 'host', 'content-length'}
DROP_REQUEST = HOP | {'cookie', 'origin', 'referer'}       # the Pi gets only its own token cookie
DROP_RESPONSE = (HOP - {'content-length'}) | {'set-cookie'}  # the token stays in the hub


class Relay:
    def __init__(self, urls, token_files=(), transport: httpx.AsyncBaseTransport | None = None,
                 probe_ttl: float = PROBE_TTL, prefix: str = ''):
        if isinstance(urls, str):
            urls = urls.split(',')
        self.urls = [u.strip().rstrip('/') for u in urls if u.strip()]
        if not self.urls:
            raise ValueError('pi-dashboard: no urls configured')
        self.token_files = [rel(f) for f in token_files]
        self.probe_ttl, self.prefix = probe_ttl, prefix
        self.client = httpx.AsyncClient(transport=transport, follow_redirects=False,
                                        timeout=httpx.Timeout(10.0, read=None, pool=10.0))
        self.upstream: str | None = None
        self.checked = -1e9
        self.authorized: bool | None = None
        self.last_error = ''
        self._probe_lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = threading.Event()

    # ---- token / upstream
    def token(self) -> tuple[str | None, Path | None]:
        for p in self.token_files:
            try:
                if p.is_file() and (t := p.read_text().strip()):
                    return t, p
            except OSError:
                continue
        return None, None

    def _cookie(self) -> dict:
        tok, _ = self.token()
        return {'Cookie': f'umi_access={tok}'} if tok else {}

    async def resolve(self, force: bool = False) -> str | None:
        """The first URL that answers (cached for probe_ttl seconds)."""
        self._loop = asyncio.get_running_loop()
        if not force and time.monotonic() - self.checked < self.probe_ttl:
            return self.upstream
        if self._probe_lock is None:
            self._probe_lock = asyncio.Lock()
        async with self._probe_lock:
            if not force and time.monotonic() - self.checked < self.probe_ttl:
                return self.upstream        # another request probed while we waited
            found, errors = None, []
            for base in self.urls:
                try:
                    r = await self.client.get(base + '/api/status', headers=self._cookie(), timeout=PROBE_TIMEOUT)
                    await r.aclose()
                except httpx.HTTPError as ex:
                    errors.append(f'{base}: {type(ex).__name__}')
                    continue
                found, self.authorized = base, r.status_code != 401
                break
            self.upstream, self.checked = found, time.monotonic()
            self.last_error = '' if found else '; '.join(errors)
            return found

    def forget(self, error: str = ''):
        self.upstream, self.checked, self.last_error = None, -1e9, error

    # ---- relay
    async def relay(self, request: Request, path: str) -> StreamingResponse:
        base = await self.resolve()
        if base is None:
            raise HTTPException(502, f'The Pi dashboard is not reachable (tried {", ".join(self.urls)})')
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in DROP_REQUEST]
        headers += list(self._cookie().items())
        body = await request.body()
        target = f'{base}/{path}' + (f'?{request.url.query}' if request.url.query else '')
        req = self.client.build_request(request.method, target, headers=headers, content=body or None)
        try:
            upstream = await asyncio.wait_for(self.client.send(req, stream=True), HEADERS_TIMEOUT)
        except (httpx.HTTPError, asyncio.TimeoutError) as ex:
            self.forget(f'{type(ex).__name__}: {ex}')
            raise HTTPException(502, f'The Pi did not answer: {type(ex).__name__}')
        if upstream.status_code == 401:
            self.authorized = False
        elif upstream.status_code < 500:
            self.authorized = True
        out = []
        for k, v in upstream.headers.multi_items():
            kl = k.lower()
            if kl in DROP_RESPONSE:
                continue
            if kl == 'location' and v.startswith('/'):
                v = self.prefix + v
            out.append((kl.encode('latin-1'), v.encode('latin-1')))

        async def stream():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            except httpx.HTTPError:
                pass                        # Pi went away mid-stream: end the response
            finally:
                with anyio.CancelScope(shield=True):     # also when the browser went away (cancelled)
                    await upstream.aclose()

        resp = StreamingResponse(stream(), status_code=upstream.status_code, background=BackgroundTask(upstream.aclose))
        resp.raw_headers = out
        return resp

    # ---- hub hooks
    async def status(self) -> dict:
        base = await self.resolve(force=True)
        tok, path = self.token()
        return {'reachable': base is not None, 'upstream': base, 'urls': self.urls,
                'token': tok is not None, 'token_file': str(path) if path else None,
                'authorized': self.authorized if base else None, 'error': self.last_error}

    def info(self) -> dict:
        tok, _ = self.token()
        reach = 'unknown' if self.checked < 0 else 'yes' if self.upstream else 'no'
        return {'upstream': self.upstream or '', 'reachable': reach, 'tries': ', '.join(self.urls),
                'token cookie': 'yes' if tok else 'none found'}

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        loop = self._loop          # the client's connections belong to the server's event loop
        if loop is not None and loop.is_running():
            # Don't wait: the hub may be stopping apps from that loop's own thread.
            asyncio.run_coroutine_threadsafe(self.client.aclose(), loop)


def start(hub, urls=(), token_files=(), transport=None):
    urls = urls or local_config.pi_urls()
    token_files = token_files or [local_config.pi_token_file()]
    prefix = ''
    if hub is not None:
        app = next((a for a in hub.apps.values() if a.module.__name__ == __name__), None)
        prefix = f'/api/{app.id}' if app else ''
    return Relay(urls, token_files, transport=transport, prefix=prefix)


def routes(get) -> APIRouter:
    r = APIRouter()

    @r.get('/status')
    async def status():
        return await get().status()

    @r.api_route('/{path:path}', methods=['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
    async def relay(path: str, request: Request):
        return await get().relay(request, path)

    return r
