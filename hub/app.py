"""The hub: every web UI in this repo as one FastAPI app, one process, one page.

    .venv/bin/python -m hub [--port 8888] [--only inference,twin-viewer]

Then open http://127.0.0.1:8888/. Each entry in ``configs/hub/apps.json`` is an app: a Python module in
``hub/apps/`` that builds a *service* (the app's state object: ``Inference``, ``EpisodeIndex``, the GPIO
bridge, ...) and an ``APIRouter`` whose routes call that service directly. Routers live under
``/api/<id>/``; the page is a single shell (``hub/web``) that loads each app's view module
(``hub/web/views/<view>/view.js``) into the same document. There are no child servers, no iframes and
no app-to-app HTTP: an app that needs another one asks the hub for its service (``hub.service('inference')``).

App module contract (``hub/apps/<module>.py``):

    def start(hub: Hub, **options) -> service        # options from apps.json, merged with configs/hub/local.json
    def routes(get: Callable[[], service]) -> APIRouter
    STATIC: dict[str, Path] = {}                     # optional: extra web roots, served at /web/<name>/

``get()`` returns the running service, or raises 503 while the app is stopped / starting. A service may
define ``close()``, ``info() -> dict`` (shown on the hub page) and ``status() -> dict``.

    /                         the shell (hub/web/index.html); /<id>/... too (client-side routing)
    /web/...                  hub/web static files (shell, theme, views) and the apps' STATIC roots
    /artifacts/...            the repository's artifacts/ (static review pages, videos; Range requests seek)
    /api/hub/apps             registry + live status
    /api/hub/apps/<id>/start|stop|restart     POST
    /api/hub/apps/<id>/settings               POST {"options": {...}} (saved to configs/hub/local.json)
    /api/<id>/...             the app's own routes

Safety: binds 127.0.0.1 and refuses any HTTP or WebSocket request whose Host or Origin isn't this
server, in front of every app (they drive the excavator and have no auth of their own).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / 'artifacts'
HUB = ROOT / 'hub'
WEB = HUB / 'web'
REGISTRY = ROOT / 'configs' / 'hub' / 'apps.json'
LOCAL = ROOT / 'configs' / 'hub' / 'local.json'

THREADS = 256          # sync routes and streams (SSE, MJPEG) each hold a worker thread while open


def rel(p) -> Path:
    """A path from apps.json: ``~`` expanded, relative to the repository root."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


@dataclass
class App:
    spec: dict
    local: dict                                     # this app's entry in configs/hub/local.json (mutable)
    module: Any
    router: APIRouter | None = None
    service: Any = None
    error: str = ''
    starting: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def id(self) -> str:
        return self.spec['id']

    @property
    def view(self) -> str:
        return self.spec.get('view', self.id)

    def options(self) -> dict:
        return {**self.spec.get('options', {}), **self.local.get('options', {})}

    def status(self) -> str:
        return 'starting' if self.starting else 'running' if self.service is not None \
            else 'crashed' if self.error else 'stopped'

    def describe(self) -> dict:
        d = {k: self.spec.get(k) for k in ('id', 'name', 'group', 'description')}
        d.update(view=self.view, preload=bool(self.spec.get('preload')), status=self.status(), options=self.options(), api=f'/api/{self.id}',
                 error=self.error.strip().splitlines()[-1] if self.error else '', traceback=self.error)
        svc = self.service
        if svc is not None and hasattr(svc, 'info'):
            try:
                d['info'] = svc.info()
            except Exception as ex:                 # never let one app break the listing
                d['info'] = {'error': f'{type(ex).__name__}: {ex}'}
        return d


class Hub:
    def __init__(self, port: int = 8888, only: set[str] | None = None, registry: Path = REGISTRY,
                 local: Path | None = LOCAL, autostart: bool = True):
        self.port, self.local_path, self.autostart = port, local, autostart
        self.local = json.loads(local.read_text()) if local and local.exists() else {}
        self.local.setdefault('apps', {})
        specs = [s for s in json.loads(Path(registry).read_text())['apps'] if not only or s['id'] in only]
        self.apps: dict[str, App] = {}
        for s in specs:
            mod = importlib.import_module(f"hub.apps.{s.get('module', s['id'].replace('-', '_'))}")
            self.apps[s['id']] = App(s, self.local['apps'].setdefault(s['id'], {}), mod)

    # ---- services
    def service(self, app_id: str, wait: float = 0.0):
        """The running service of ``app_id`` (None if absent / stopped). ``wait``: seconds to wait while it starts."""
        app = self.apps.get(app_id)
        if app is None:
            return None
        deadline = time.monotonic() + wait
        while app.starting and time.monotonic() < deadline:
            time.sleep(0.05)
        return app.service

    def getter(self, app: App) -> Callable[[], Any]:
        def get():
            svc = app.service
            if svc is None:
                what = 'starting' if app.starting else f'failed: {app.describe()["error"]}' if app.error else 'stopped'
                raise HTTPException(503, f'{app.spec["name"]} is {what}')
            return svc
        return get

    def start(self, app_id: str) -> dict:
        app = self.apps[app_id]
        with app.lock:
            if app.service is not None:
                return {'ok': True, 'note': 'already running'}
            app.starting, app.error = True, ''
            try:
                app.service = app.module.start(self, **app.options())
                return {'ok': True}
            except Exception as ex:
                app.error = ''.join(traceback.format_exception(ex))
                return {'ok': False, 'error': f'{type(ex).__name__}: {ex}'}
            finally:
                app.starting = False

    def stop(self, app_id: str) -> dict:
        app = self.apps[app_id]
        with app.lock:
            svc, app.service = app.service, None
            if svc is not None and hasattr(svc, 'close'):
                try:
                    svc.close()
                except Exception as ex:
                    return {'ok': True, 'note': f'close raised {type(ex).__name__}: {ex}'}
            return {'ok': True}

    def start_all(self):
        """In apps.json order for apps marked ``"first": true`` (the GPIO bridge: inference uses it), then the
        rest in parallel (MuJoCo / episode scans are slow)."""
        first = [a for a in self.apps.values() if a.spec.get('first')]
        rest = [a for a in self.apps.values() if not a.spec.get('first')]
        for a in rest:
            a.starting = True                       # so the page shows 'starting', not 'stopped'
        for a in first:
            self._report(a, self.start(a.id))
        with ThreadPoolExecutor(len(rest) or 1) as pool:
            for a, r in zip(rest, pool.map(lambda a: self.start(a.id), rest)):
                self._report(a, r)

    @staticmethod
    def _report(app: App, r: dict):
        print(f'  {app.id}: {"ok" if r.get("ok") else r.get("error")}', flush=True)

    def stop_all(self):
        threads = [threading.Thread(target=self.stop, args=(i,)) for i in reversed(list(self.apps))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def save_local(self):
        if self.local_path is None:
            return
        out = {**self.local, 'apps': {k: v for k, v in self.local['apps'].items() if any(v.values())}}
        self.local_path.write_text(json.dumps(out, indent=2) + '\n')


# ---------------------------------------------------------------------------------------- guard

class HostGuard:
    """Refuse HTTP and WebSocket requests whose Host isn't this server or whose Origin is another site."""

    def __init__(self, app, allowed_hosts: set[str]):
        self.app, self.allowed = app, allowed_hosts

    async def __call__(self, scope, receive, send):
        if scope['type'] not in ('http', 'websocket'):
            return await self.app(scope, receive, send)
        headers = {k.decode('latin-1'): v.decode('latin-1') for k, v in scope['headers']}
        host = headers.get('host', '').strip()
        name = host.split(']')[0] + ']' if host.startswith('[') else host.rsplit(':', 1)[0]
        origin = headers.get('origin')
        ok = name in self.allowed and (not origin or origin == 'null' or urlsplit(origin).netloc == host)
        if ok:
            return await self.app(scope, receive, send)
        if scope['type'] == 'websocket':
            return await send({'type': 'websocket.close', 'code': 1008})
        await JSONResponse({'error': 'forbidden host / origin'}, 403)(scope, receive, send)


# ---------------------------------------------------------------------------------------- app

def create_app(hub: Hub, allowed_hosts=('127.0.0.1', 'localhost', '[::1]', 'testserver'),
               artifacts: Path = ARTIFACTS) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_):
        import anyio.to_thread
        anyio.to_thread.current_default_thread_limiter().total_tokens = THREADS
        if hub.autostart:
            threading.Thread(target=hub.start_all, daemon=True, name='hub-start').start()
        try:
            yield
        finally:
            hub.stop_all()

    api = FastAPI(title='UMI Excavator hub', lifespan=lifespan, docs_url='/api/docs', openapi_url='/api/openapi.json',
                  redoc_url=None)
    api.state.hub = hub

    @api.exception_handler(HTTPException)
    async def http_error(_, ex: HTTPException):
        return JSONResponse({'error': ex.detail}, ex.status_code, headers=getattr(ex, 'headers', None))

    @api.get('/api/hub/apps')
    def list_apps():
        return {'apps': [a.describe() for a in hub.apps.values()]}

    @api.post('/api/hub/apps/{app_id}/{action}')
    async def app_action(app_id: str, action: str, request: Request):
        import anyio.to_thread
        if app_id not in hub.apps or action not in ('start', 'stop', 'restart', 'settings'):
            raise HTTPException(404, 'not found')
        app = hub.apps[app_id]
        if action == 'settings':
            try:
                body = await request.json()
            except ValueError:
                raise HTTPException(400, 'invalid JSON')
            if not isinstance(body, dict) or not isinstance(body.get('options'), dict):
                raise HTTPException(400, 'options must be a JSON object')
            app.local['options'] = body['options']
            hub.save_local()
            r = {'ok': True, 'note': 'restart the app to apply'}
        elif action == 'restart':
            r = await anyio.to_thread.run_sync(hub.stop, app_id)
            r = await anyio.to_thread.run_sync(hub.start, app_id) if r.get('ok') else r
        else:
            r = await anyio.to_thread.run_sync(getattr(hub, action), app_id)
        return JSONResponse({**r, 'app': app.describe()}, 200 if r.get('ok') else 409)

    for app in hub.apps.values():
        app.router = app.module.routes(hub.getter(app))
        api.include_router(app.router, prefix=f'/api/{app.id}')

    @api.api_route('/api/{rest:path}', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
    def api_not_found(rest: str):
        raise HTTPException(404, 'not found')

    class WebFiles(StaticFiles):
        """Static UI files the browser revalidates on every load, so edited views show on a plain reload."""
        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers['Cache-Control'] = 'no-cache'
            return response

    mounted = set()
    for app in hub.apps.values():
        for name, path in getattr(app.module, 'STATIC', {}).items():
            if name not in mounted:
                api.mount(f'/web/{name}', WebFiles(directory=path, follow_symlink=True), name=f'web-{name}')
                mounted.add(name)
    api.mount('/web', WebFiles(directory=WEB, follow_symlink=True), name='web')
    if artifacts.is_dir():
        api.mount('/artifacts', StaticFiles(directory=artifacts, html=True), name='artifacts')

    index = WEB / 'index.html'

    @api.get('/{path:path}', include_in_schema=False)
    def shell(path: str):
        return FileResponse(index, headers={'Cache-Control': 'no-store'})

    return HostGuard(api, set(allowed_hosts)) if allowed_hosts is not None else api


def main(argv=None):
    import uvicorn
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', default='127.0.0.1', help='bind address (default local only: no auth, and it '
                                                        'fronts the excavator controls)')
    p.add_argument('--port', type=int, default=8888)
    p.add_argument('--only', default='', help='comma-separated app ids to load (default: all)')
    a = p.parse_args(argv)
    os.chdir(ROOT)
    only = {s.strip() for s in a.only.split(',') if s.strip()} or None
    hub = Hub(a.port, only)
    allowed = {'127.0.0.1', 'localhost', '[::1]', a.host}
    if a.host not in ('127.0.0.1', 'localhost', '::1'):
        print(f'WARNING: binding {a.host}: anyone who can reach this port can drive the excavator (no auth)',
              file=sys.stderr)
    print(f'hub: http://127.0.0.1:{a.port}/  (starting apps...)', flush=True)
    uvicorn.run(create_app(hub, allowed), host=a.host, port=a.port, log_level='warning',
                timeout_graceful_shutdown=5, ws_ping_interval=20)


if __name__ == '__main__':
    main()
