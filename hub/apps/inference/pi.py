"""The excavator's Pi dashboard as inference reads it (status + camera, over an ordered host list), and how the
page describes the hub's GPIO bridge."""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

from hub.apps._common import fetch_live_frame
from digital_twin.hardware.pi import DEFAULT_TOKEN_FILE  # noqa: F401  (inference default; = runner's)

BRIDGE_UNAVAILABLE = 'GPIO bridge unavailable (the hub gpio app is not running)'


def normalize_host(host: str, default_scheme='http') -> str:
    """'raspberrypi.local:8000' or 'http://raspberrypi.local:8000/' -> 'http://raspberrypi.local:8000'."""
    host = (host or '').strip().rstrip('/')
    if not host:
        raise ValueError('empty host')
    if '://' not in host:
        host = f'{default_scheme}://{host}'
    p = urlsplit(host)
    if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password:
        raise ValueError(f'bad host {host!r}')
    return f'{p.scheme}://{p.netloc}'


def link_kind(base: str) -> str:
    """'wired' for the link-local cable (169.254.x.x), else 'wifi'."""
    return 'wired' if (urlsplit(base).hostname or '').startswith('169.254.') else 'wifi'


def describe_bridge(bridge) -> str:
    """What the page shows for the bridge: the GpioBridge's ESP32 target, or that there is none."""
    if bridge is None:
        return 'none'
    return f"hub gpio ({getattr(bridge, 'target', '') or '?'})"


class PiClient:
    """The excavator's Pi dashboard: /api/status and /preview.jpg with the umi_access cookie. ``host``
    may be an ordered comma-separated list (wired link-local first, then Wi-Fi): the first host that
    answers ``GET /api/status`` within ``probe_timeout`` becomes active; when it fails, the list is
    probed again. The token is never returned by any endpoint or logged."""

    def __init__(self, host: str, token_file=None, token: str = '', timeout: float = 2.0, probe_timeout: float = 1.0):
        self.spec = ','.join(h.strip() for h in str(host).split(',') if h.strip())   # passed verbatim to the runner / hardware.calibrate
        self.hosts = [normalize_host(h) for h in self.spec.split(',')]
        if not self.hosts:
            raise ValueError('no Pi host given')
        self.token_file = Path(token_file).expanduser() if token_file else None
        if not token and self.token_file and self.token_file.is_file():
            token = self.token_file.read_text().strip()
        self.token, self.timeout, self.probe_timeout = token, timeout, probe_timeout
        self._lock = threading.Lock()
        self._host_lock = threading.Lock()
        self.active = None                          # index into hosts once a probe succeeded
        self.last_probe = {}                        # {base: {'ok', 'ms', 'error'}}
        self.status_ms = None
        self._frame = (0.0, 0, b'', '')            # (monotonic, status, body, ctype)
        self._frame_times = deque(maxlen=30)

    @property
    def base(self) -> str:
        return self.hosts[self.active if self.active is not None else 0]

    @property
    def preview_url(self) -> str:
        return self.base + '/preview.jpg'

    def link(self) -> dict:
        b = self.base
        return {'base': b, 'host': urlsplit(b).netloc, 'kind': link_kind(b), 'active': self.active is not None,
                'status_ms': self.status_ms, 'hosts': self.hosts, 'probes': self.last_probe}

    def _get_status(self, base: str, timeout: float) -> dict:
        req = urllib.request.Request(base + '/api/status',
                                     headers={'Cookie': f'umi_access={self.token}'} if self.token else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def probe(self) -> int | None:
        """Try every host in order (probe_timeout each); the first that answers becomes active."""
        with self._host_lock:
            for i, base in enumerate(self.hosts):
                t0 = time.monotonic()
                try:
                    self._get_status(base, self.probe_timeout)
                    ms = round(1000 * (time.monotonic() - t0), 1)
                    self.last_probe[base] = {'ok': True, 'ms': ms}
                    if self.active != i:
                        self._frame = (0.0, 0, b'', '')      # do not serve another link's cached frame
                    self.active, self.status_ms = i, ms
                    return i
                except Exception as ex:
                    self.last_probe[base] = {'ok': False, 'error': f'{type(ex).__name__}: {getattr(ex, "reason", ex)}'}
            self.active = None
            return None

    def status(self) -> dict:
        """GET /api/status from the active host; on failure re-probe the list once and retry."""
        if self.active is None and self.probe() is None:
            raise ConnectionError('no Pi host answered: ' + '; '.join(
                f"{urlsplit(b).netloc} {self.last_probe.get(b, {}).get('error', '?')}" for b in self.hosts))
        base = self.base
        t0 = time.monotonic()
        try:
            st = self._get_status(base, self.timeout)
            self.status_ms = round(1000 * (time.monotonic() - t0), 1)
            return st
        except Exception:
            if self.probe() is None:
                raise
            t0 = time.monotonic()
            st = self._get_status(self.base, self.timeout)
            self.status_ms = round(1000 * (time.monotonic() - t0), 1)
            return st

    def frame(self, max_age_s: float = 0.08, fail_hold_s: float = 1.0) -> tuple[int, bytes, str]:
        """Latest camera JPEG, shared by every viewer (at most one fetch per max_age_s; an error is
        cached for fail_hold_s so an offline Pi does not stall the page)."""
        with self._lock:
            t, status, body, ctype = self._frame
            age = time.monotonic() - t
            if body and (age < max_age_s or (status != 200 and age < fail_hold_s)):
                return status, body, ctype
            status, body, ctype = fetch_live_frame(self.preview_url, self.token, self.timeout)
            deadline = time.monotonic() + 1.5
            while status == 204 and self._frame[1] != 200 and time.monotonic() < deadline:
                time.sleep(0.05)                          # the Pi prepares the preview on demand (cold: 204)
                status, body, ctype = fetch_live_frame(self.preview_url, self.token, self.timeout)
            if status == 502 and len(self.hosts) > 1 and self.probe() is not None:
                status, body, ctype = fetch_live_frame(self.preview_url, self.token, self.timeout)
            if status == 204 and self._frame[1] == 200:
                status, body, ctype = self._frame[1:]
            elif status == 200:
                self._frame_times.append(time.monotonic())
            self._frame = (time.monotonic(), status, body, ctype)
            return status, body, ctype

    def proxy_fps(self) -> float:
        ts = list(self._frame_times)
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
