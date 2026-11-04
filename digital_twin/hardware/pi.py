"""The excavator Pi's dashboard (hardware/umi-data-collection/recorder): host selection and the camera / IMU sensor source."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from policy.data import local_config

from .excavator import RunnerError


# hosts.pi in configs/hub/local.json (policy.data.local_config): the excavator Pi's dashboard URLs in preference
# order (e.g. wired link-local, then mDNS, then Wi-Fi) and its access token file.
DEFAULT_PI_HOST = ','.join(local_config.pi_urls())
DEFAULT_TOKEN_FILE = local_config.pi_token_file()


def normalize_host(host: str, default_scheme='http') -> str:
    """'raspberrypi.local:8000' or 'http://raspberrypi.local:8000/' -> 'http://raspberrypi.local:8000'."""
    from urllib.parse import urlsplit
    host = (host or '').strip().rstrip('/')
    if not host:
        raise ValueError('empty host')
    if '://' not in host:
        host = f'{default_scheme}://{host}'
    p = urlsplit(host)
    if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password:
        raise ValueError(f'bad host {host!r}')
    return f'{p.scheme}://{p.netloc}'


def parse_pi_hosts(spec) -> list[str]:
    """'a:8000,b:8000' (or a list) -> ['http://a:8000', 'http://b:8000'] in preference order."""
    items = spec if isinstance(spec, (list, tuple)) else str(spec).split(',')
    out = [normalize_host(h) for h in (x.strip() for x in items) if h]
    if not out:
        raise RunnerError('no Pi dashboard host given')
    return out


def link_kind(base: str) -> str:
    from urllib.parse import urlsplit
    h = urlsplit(base).hostname or ''
    if h.startswith('169.254.'):
        return 'wired (link-local)'
    if h.startswith(('192.168.', '10.')) or h.startswith('172.'):
        return 'wifi/LAN'
    return 'network'


def ipv4_base(base: str) -> str:
    """``http://name:port`` -> ``http://<IPv4>:port`` when ``name`` resolves to IPv4. Python tries a
    host's addresses in order, and macOS lists a ``.local`` name's IPv6 link-local first; that connect
    fails after ~1 s on every request, so a name costs 1-2 s per call where its IPv4 costs ~20 ms."""
    import ipaddress
    import socket
    from urllib.parse import urlsplit
    u = urlsplit(base)
    try:
        ipaddress.ip_address(u.hostname or '')
        return base                                    # already an address
    except ValueError:
        pass
    try:
        ip = socket.getaddrinfo(u.hostname, u.port or 80, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    except (OSError, IndexError):
        return base
    return u._replace(netloc=f'{ip}:{u.port}' if u.port else ip).geturl()


def probe_pi_hosts(hosts, headers: dict, timeout: float = 1.0) -> tuple[str | None, list[dict]]:
    """GET /api/status on each host in order; the first that answers 200 wins. Returns (base, attempts).
    Names are resolved to IPv4 first (see :func:`ipv4_base`)."""
    import urllib.request
    tried = []
    for base in (ipv4_base(h) for h in hosts):
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(base + '/api/status', headers=headers), timeout=timeout) as r:
                ok = r.status == 200
                r.read()
            tried.append({'host': base, 'ok': ok, 'rtt_ms': round((time.monotonic() - t0) * 1000, 1)})
            if ok:
                return base, tried
        except Exception as ex:
            tried.append({'host': base, 'ok': False, 'error': f'{type(ex).__name__}: {getattr(ex, "reason", ex)}'})
    return None, tried


class PiSensorSource:
    """The excavator's Pi dashboard (hardware/umi-data-collection/recorder), read with minimal latency.

    * Camera: a background thread holds ``/stream.mjpg`` open (multipart JPEG of the full-resolution
      1280x720 camera frames, pushed as they arrive, ~20 fps) and keeps only the newest frame, so
      ``grab`` normally waits only for the next frame and decodes it once. If the stream is refused
      (the dashboard allows 8 viewers) it falls back to ``/preview.jpg`` (640x360, produced on demand;
      204 while preparing) polled every 20 ms over a keep-alive connection.
    * IMU: a second thread polls ``/api/status`` every 50 ms over a keep-alive connection; ``imu()``
      returns the newest sample without a network round trip.

    Auth is the ``umi_access`` cookie from the token file (never printed). Timestamps are
    time.monotonic() at receipt; ``grab`` reports the frame age at use and how long it waited.
    """

    def __init__(self, pi_host=DEFAULT_PI_HOST, token_file=DEFAULT_TOKEN_FILE, timeout: float = 1.5,
                 frame_timeout: float = 3.0, stop_event=None, stream: bool = True, status_period_s: float = 0.05,
                 log=None, probe_timeout: float = 1.0, failover_after: int = 3):
        self.hosts = parse_pi_hosts(pi_host)
        self.log = log or (lambda m: None)
        self.probe_timeout, self.failover_after = probe_timeout, failover_after
        self._host_lock = threading.Lock()
        self._fail = 0
        self.host_events = []                   # (monotonic, host, reason)
        self._selected = False
        self._set_host(self.hosts[0])
        tf = Path(token_file).expanduser() if token_file else None
        self.token = tf.read_text().strip() if tf and tf.is_file() else ''
        if tf and not self.token:
            raise RunnerError(f'Pi dashboard token file {tf} is missing or empty')
        self.headers = {'Cookie': f'umi_access={self.token}'} if self.token else {}
        self.timeout, self.frame_timeout, self.status_period_s = timeout, frame_timeout, status_period_s
        self.stop_event = stop_event or threading.Event()
        self._closed = threading.Event()
        self._cond = threading.Condition()
        self._frame = None                      # (jpeg bytes, t_recv, seq)
        self._seq = 0
        self._used_seq = 0
        self._status = None                     # (t_recv, dict)
        self.source = 'stream.mjpg' if stream else 'preview.jpg'
        self.stream_error = None
        self.frames = 0
        self.last = None
        self._threads = []
        self._want_stream = stream
        self.status_listeners = []              # callables(t_recv, status dict) on every poll (the runner's yaw)

    # -- host selection / failover
    def _set_host(self, base: str):
        from urllib.parse import urlsplit
        u = urlsplit(base)
        self.base, self.host, self.port = base, u.hostname, u.port or 80
        self.url = self.base + '/preview.jpg'

    def select_host(self, reason: str = 'startup') -> str:
        """Probe the hosts in preference order (GET /api/status, ``probe_timeout``) and use the first that
        answers. Raises RunnerError when none does (at startup)."""
        with self._host_lock:
            base, tried = probe_pi_hosts(self.hosts, self.headers, self.probe_timeout)
            if base is None:
                msg = 'Pi dashboard unreachable on every host: ' + '; '.join(
                    f"{t['host']} ({t.get('error', 'HTTP error')})" for t in tried)
                if not self._selected:
                    raise RunnerError(msg)
                self.log('WARNING: ' + msg + f' - staying on {self.base}')
                return self.base
            changed = base != self.base or not self._selected
            self._set_host(base); self._selected = True; self._fail = 0
            rtt = next(t['rtt_ms'] for t in tried if t['host'] == base)
            self.host_events.append((time.monotonic(), base, reason))
            if changed or reason == 'startup':
                self.log(f'Pi dashboard: {base} [{link_kind(base)}], /api/status {rtt} ms ({reason})'
                         + ('' if len(tried) == 1 else '; tried ' + ', '.join(
                             f"{t['host']} {'ok' if t['ok'] else 'failed'}" for t in tried)))
            return base

    def _note_failure(self, what: str):
        """Count network failures (stream drops, failed polls); after ``failover_after`` re-probe the hosts."""
        with self._host_lock:
            self._fail += 1
            due = self._fail >= self.failover_after
        if due and self._alive():
            self.log(f'Pi dashboard: {self._fail} failures on {self.base} (last: {what}) - re-probing hosts')
            self.select_host(f'failover after {what}')

    # -- background readers
    def _conn(self):
        import http.client
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def start(self):
        if self._threads:
            return
        if not self._selected:
            self.select_host()
        for target in ((self._stream_loop if self._want_stream else self._preview_loop), self._status_loop):
            th = threading.Thread(target=target, daemon=True, name=f'pi-{target.__name__}')
            th.start(); self._threads.append(th)

    def _alive(self):
        return not (self._closed.is_set() or self.stop_event.is_set())

    def _publish(self, jpeg: bytes, t: float):
        with self._cond:
            self._seq += 1
            self._frame = (jpeg, t, self._seq)
            self._cond.notify_all()

    def _stream_loop(self):
        failures = 0
        while self._alive():
            conn = None
            try:
                conn = self._conn()
                conn.request('GET', '/stream.mjpg', headers=self.headers)
                resp = conn.getresponse()
                if resp.status != 200:
                    raise RuntimeError(f'/stream.mjpg HTTP {resp.status}')
                failures = 0
                base = self.base
                while self._alive() and self.base == base:     # a failover switches hosts: reconnect
                    length = None
                    while True:                               # part headers
                        line = resp.fp.readline()
                        if not line:
                            raise RuntimeError('stream closed')
                        line = line.strip()
                        if not line and length is not None:
                            break
                        if line.lower().startswith(b'content-length:'):
                            length = int(line.split(b':', 1)[1])
                    jpeg = resp.fp.read(length)
                    t = time.monotonic()
                    resp.fp.readline()                        # trailing CRLF
                    if jpeg[:2] == b'\xff\xd8':
                        self._publish(jpeg, t)
            except Exception as ex:                           # reconnect, then fall back to the preview
                self.stream_error = f'{type(ex).__name__}: {ex}'
                if not str(ex).startswith('/stream.mjpg HTTP'):
                    self._note_failure('stream drop'); time.sleep(0.2); continue
                failures += 1
                if failures >= 3:
                    self.source = 'preview.jpg (stream unavailable)'
                    return self._preview_loop()
                time.sleep(0.2)
            finally:
                if conn is not None:
                    conn.close()

    def _preview_loop(self):
        conn, last_seq, conn_base = None, None, None
        while self._alive():
            try:
                if conn is not None and conn_base != self.base:
                    conn.close(); conn = None
                conn_base = conn_base if conn is not None else self.base
                conn = conn or self._conn()
                path = '/preview.jpg' + (f'?after={last_seq}' if last_seq is not None else '')
                conn.request('GET', path, headers=self.headers)
                resp = conn.getresponse(); body = resp.read()
                if resp.status == 200 and body[:2] == b'\xff\xd8':
                    last_seq = resp.getheader('X-Frame-Sequence')
                    age = float(resp.getheader('X-Frame-Age-Ms') or 0.0) / 1000.0
                    self._publish(body, time.monotonic() - age)
                else:
                    time.sleep(0.02)                          # 204: being prepared / no newer frame
            except Exception as ex:
                self.stream_error = f'{type(ex).__name__}: {ex}'
                if conn is not None:
                    conn.close()
                conn = None; self._note_failure('preview poll'); time.sleep(0.1)

    def _status_loop(self):
        conn, conn_base = None, None
        while self._alive():
            try:
                if conn is not None and conn_base != self.base:
                    conn.close(); conn = None
                if conn is None:
                    conn, conn_base = self._conn(), self.base
                conn.request('GET', '/api/status', headers=self.headers)
                resp = conn.getresponse(); body = resp.read()
                if resp.status == 200:
                    self._status = (time.monotonic(), json.loads(body.decode()))
                    with self._host_lock:
                        self._fail = 0
                    for fn in list(self.status_listeners):
                        try:
                            fn(*self._status)
                        except Exception as ex:                # a listener never stops the poller
                            self.log(f'status listener failed: {type(ex).__name__}: {ex}')
            except Exception:
                if conn is not None:
                    conn.close()
                conn = None; self._note_failure('status poll'); time.sleep(0.1)
            time.sleep(self.status_period_s)

    def close(self):
        self._closed.set()

    # -- API used by the runner
    def status(self, max_age_s: float = 0.5) -> dict:
        """Newest /api/status (from the poller when younger than ``max_age_s``, else fetched now)."""
        st = self._status
        if st is not None and time.monotonic() - st[0] < max_age_s:
            return st[1]
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request(self.base + '/api/status', headers=self.headers),
                                    timeout=self.timeout) as r:
            d = json.loads(r.read().decode())
        self._status = (time.monotonic(), d)
        return d

    def imu(self):
        """((accel_g, gyro_dps, t_monotonic) or None when the IMU is not live, raw status)."""
        self.start()
        st = self.status(max_age_s=0.25)
        imu = st.get('imu') or {}
        s = imu.get('sample') or {}
        if not imu.get('live') or 'accel' not in s:
            return None, st
        t = self._status[0] if self._status else time.monotonic()
        return (np.asarray(s['accel'], float), np.asarray(s.get('gyro', [0, 0, 0]), float), t), st

    def imus(self):
        """({name: (accel_g, gyro_dps, t_sample)} for every live IMU, raw status). The dashboard lists the
        bucket, stick and boom IMUs under ``imus``; an older dashboard only has the bucket ``imu``.
        ``accel_g`` is the Pi's mean of the last ~58 ms when reported (noise / 3.5 at rest), else the newest
        sample; ``t_sample`` = receipt time minus the sample's age on the Pi."""
        self.start()
        st = self.status(max_age_s=0.25)
        t_recv = self._status[0] if self._status else time.monotonic()
        out = {}
        for name, e in (st.get('imus') or {'bucket': st.get('imu') or {}}).items():
            s = (e or {}).get('sample') or {}
            if e.get('live') and 'accel' in s:
                accel = e.get('accel_mean') or s['accel']          # the Pi's ~58 ms mean when it reports one
                out[name] = (np.asarray(accel, float), np.asarray(s.get('gyro', [0, 0, 0]), float),
                             t_recv - float(e.get('age_s') or 0.0))
        return out, st

    def grab(self, after: float | None = None):
        """Newest frame not used before and received after ``after`` (monotonic s) -> (rgb, t_recv, meta)."""
        import io

        from PIL import Image
        self.start()
        t0 = time.monotonic()
        deadline = t0 + self.frame_timeout
        with self._cond:
            while (self._frame is None or self._frame[2] == self._used_seq
                   or (after is not None and self._frame[1] < after)):
                if self.stop_event.is_set():
                    raise RunnerError('stop requested')
                if time.monotonic() > deadline:
                    raise RunnerError(f'Pi camera: no new frame from {self.base} ({self.source}) within '
                                      f'{self.frame_timeout:.1f} s' + (f'; {self.stream_error}' if self.stream_error else ''))
                self._cond.wait(0.005)
            jpeg, t_recv, seq = self._frame
            self._used_seq = seq
        waited = time.monotonic() - t0
        img = np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB'))
        now = time.monotonic()
        self.frames += 1; self.last = (img, t_recv)
        return img, t_recv, {'source': self.source, 'fetch_wait_ms': round(waited * 1000, 1),
                             'frame_age_ms': round((now - t_recv) * 1000, 1),
                             'decode_ms': round((now - t0 - waited) * 1000, 1), 'frame_shape': list(img.shape)}
