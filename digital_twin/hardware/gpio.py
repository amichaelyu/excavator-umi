"""The GPIO bridge to the ESP32, in-process.

    client (Session / hub websocket)  ──▶  GpioBridge  ──BLE (bleak)──▶  ESP32 ("gpio")

One :class:`GpioBridge` per process holds the single upstream link to the ESP32 and shares it between
clients. Each client is a :class:`Session` speaking the firmware's JSON protocol (hardware/rc-control/src/main.cpp) as dicts:
the bridge rewrites request ids so replies reach the right session, fans out pin-state broadcasts (only
when the state changed), sends ``{"cmd": "state", "fresh": true}`` reads to the firmware, releases any
``autorelease`` hold a session still owns when it closes and keeps pin names (the hardware map's joint pin labels, digital_twin.sim.pins).

The hub's ``gpio`` app service is a ``GpioBridge`` (``hub.service('gpio')``), served to browsers and
scripts at ``ws://<host>:8888/api/gpio/ws``. In-process code uses it directly; scripts outside the hub
use the same interface over that websocket::

    from digital_twin.hardware import gpio
    with gpio.connect('localhost:8888') as s:            # or gpio.connect(bridge) with a GpioBridge
        print(s.request({'cmd': 'trigger', 'pin': 26, 'ms': 300}))

Upstream targets: ``ble://<name>`` (scan by advertised name; ``ble://<address>`` with a MAC / macOS
UUID picks one device; the hub folds BLE_NAME / BLE_ADDRESS / ESP32_URL into its default target) and ``ws://host:port/ws`` (an ESP32
stand-in speaking the firmware protocol over a websocket). The link is pluggable: :data:`LINKS` maps
a URL scheme to a :class:`Link` class, so tests register an in-process fake ESP32.

Thread model: the public API is synchronous and thread-safe. Links run on the bridge's own asyncio
loop thread (bleak needs one); bridge state is guarded by one lock and nothing blocks while holding it.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import math
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from digital_twin.paths import hardware_map
from digital_twin.sim.pins import pin_labels, save_pin_labels

DEFAULT_TARGET = 'ble://gpio'
DEFAULT_HOST = 'localhost:8888'                       # the hub
WS_PATH = '/api/gpio/ws'

RECONNECT_S = 2.0
REPLY_TIMEOUT_S = 5.0
# The ESP32 can vanish without the link noticing straight away (power loss, out of range), so poll it
# and treat silence as a disconnect. Scanning and connecting report progress, so they count as activity.
HEARTBEAT_S = 5.0
UPSTREAM_TIMEOUT_S = 12.0

# Nordic-UART-style GATT service of the firmware (src/main.cpp)
SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e'
RX_UUID = '6e400002-b5a3-f393-e0a9-e50e24dcca9e'      # we write commands here
TX_UUID = '6e400003-b5a3-f393-e0a9-e50e24dcca9e'      # the ESP32 notifies here
SCAN_TIMEOUT_S = 10.0
CONNECT_TIMEOUT_S = 10.0

_NOID = object()                                      # "the client sent no id" (JS undefined)


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)


def _js_str(v) -> str:
    """JavaScript ``String(v)`` for JSON values (the key of an autorelease hold, as the page sends it)."""
    if v is None:
        return 'null'
    if v is _NOID:
        return 'undefined'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (int, float, str)):
        return str(v)
    return _dumps(v)


def _js_number(v) -> float:
    """JavaScript ``Number(v)`` for JSON values (NaN when not numeric)."""
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return 0.0
        try:
            return float(int(s, 0)) if s[:2].lower() in ('0x', '0o', '0b') else float(s)
        except ValueError:
            return math.nan
    return math.nan


# ======================================================================== upstream links
class Link:
    """One upstream connection to the ESP32. ``events(kind, data)`` reports ``'up'``, ``'msg'`` (a JSON
    line), ``'activity'`` (connecting progress) and ``'down'`` (once, when the link has ended for good;
    the bridge then makes a new link). ``send`` must not block; the bridge calls it under its lock."""

    def __init__(self, target: str, events: Callable[[str, Any], None], loop: asyncio.AbstractEventLoop):
        self.target, self.events, self.loop = target, events, loop

    def start(self):
        raise NotImplementedError

    def send(self, text: str):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class AsyncLink(Link):
    """A link whose ``run()`` coroutine runs on the bridge's loop; ``send`` queues lines for ``writes``."""

    def start(self):
        self._down = False
        self._out: asyncio.Queue | None = None
        self._task = None

        async def main():
            self._out = asyncio.Queue()
            try:
                await self.run()
            except asyncio.CancelledError:
                pass
            except Exception as ex:                     # noqa: BLE001 - any failure ends this link
                self.log(f'ESP32 link: {type(ex).__name__}: {ex}')
            finally:
                self._finish()

        def create():
            self._task = self.loop.create_task(main())
        self.loop.call_soon_threadsafe(create)

    def log(self, text):
        self.events('log', text)

    def _finish(self):
        if not self._down:
            self._down = True
            self.events('down', None)

    def send(self, text: str):
        def put():
            if self._out is not None:
                self._out.put_nowait(text)
        self.loop.call_soon_threadsafe(put)

    async def writes(self):
        """Yield the queued outgoing lines."""
        while True:
            yield await self._out.get()

    def close(self):
        def cancel():
            if self._task is not None:
                self._task.cancel()
        try:
            self.loop.call_soon_threadsafe(cancel)
        except RuntimeError:                            # loop already closed
            pass

    async def run(self):
        raise NotImplementedError


class BleLink(AsyncLink):
    """bleak: scan for the ESP32 by advertised name (``ble://gpio``) or address (``ble://<MAC or macOS UUID>``;
    :func:`default_target` folds BLE_NAME / BLE_ADDRESS into the target), subscribe to
    TX notifications (newline-delimited JSON) and write commands to RX in MTU-sized chunks. One
    connection per link: it ends when the device drops or can't be found."""

    async def run(self):
        from bleak import BleakClient, BleakScanner
        spec = self.target.split('://', 1)[1] if '://' in self.target else self.target
        address = spec if _looks_like_address(spec) else ''
        name = spec or 'gpio'
        self.events('activity', 'scanning')
        if address:
            device = await BleakScanner.find_device_by_address(address, timeout=SCAN_TIMEOUT_S)
        else:
            device = await BleakScanner.find_device_by_filter(
                lambda d, adv: adv.local_name == name or d.name == name, timeout=SCAN_TIMEOUT_S)
        if device is None:
            self.log('ESP32 not found over BLE')
            return
        lost = asyncio.Event()
        self.events('activity', f'connecting {device.address}')
        async with BleakClient(device, timeout=CONNECT_TIMEOUT_S, disconnected_callback=lambda _: lost.set()) as client:
            buf = bytearray()

            def on_notify(_, data: bytearray):
                buf.extend(data)
                while (i := buf.find(b'\n')) >= 0:
                    line = buf[:i].decode(errors='replace').strip()
                    del buf[:i + 1]
                    if line:
                        self.events('msg', line)

            await client.start_notify(TX_UUID, on_notify)
            rx = client.services.get_characteristic(RX_UUID)
            chunk = max(20, rx.max_write_without_response_size)
            self.events('up', device.address)

            async def pump():
                async for text in self.writes():
                    data = text.rstrip('\r\n').encode() + b'\n'
                    for off in range(0, len(data), chunk):
                        await client.write_gatt_char(rx, data[off:off + chunk], response=False)

            pump_task = asyncio.create_task(pump())
            gone = asyncio.create_task(lost.wait())
            try:
                done, _ = await asyncio.wait({pump_task, gone}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in (pump_task, gone):
                    t.cancel()
            if gone in done:
                self.log('BLE link lost')
            else:
                pump_task.result()                      # surface write errors


def _looks_like_address(s: str) -> bool:
    return bool(re.fullmatch(r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}', s)
                or re.fullmatch(r'[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}', s))


class WsLink(AsyncLink):
    """A websocket upstream speaking the firmware protocol (an ESP32 stand-in)."""

    async def run(self):
        from websockets.asyncio.client import connect
        self.events('activity', 'connecting')
        async with connect(self.target, open_timeout=CONNECT_TIMEOUT_S, proxy=None) as ws:
            self.events('up', self.target)

            async def pump():
                async for text in self.writes():
                    await ws.send(text)

            pump_task = asyncio.create_task(pump())
            try:
                async for raw in ws:
                    self.events('msg', raw if isinstance(raw, str) else raw.decode(errors='replace'))
            finally:
                pump_task.cancel()


LINKS: dict[str, type[Link]] = {'ble': BleLink, 'ws': WsLink, 'wss': WsLink}


def default_target() -> str:
    """The upstream the hub uses when its option is the default: ``ESP32_URL``, else ``ble://$BLE_ADDRESS``
    or ``ble://$BLE_NAME``, else ble://gpio."""
    if os.environ.get('ESP32_URL'):
        return os.environ['ESP32_URL']
    return f"ble://{os.environ.get('BLE_ADDRESS') or os.environ.get('BLE_NAME') or 'gpio'}"


class _LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True, name='gpio-link')
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        def shutdown():
            for t in asyncio.all_tasks(self.loop):
                t.cancel()
            self.loop.call_later(0.2, self.loop.stop)
        try:
            self.loop.call_soon_threadsafe(shutdown)
        except RuntimeError:
            return
        self.thread.join(3)
        if not self.loop.is_running():
            self.loop.close()


# ======================================================================== sessions
class Session:
    """One client of a :class:`GpioBridge`: the firmware's JSON protocol (hardware/rc-control/src/main.cpp) as dicts.

    On open it receives the greeting the websocket did: ``bridge``, ``labels`` and (when the
    ESP32 is online) ``state``. ``recv`` returns the next reply / broadcast; ``request`` sends a command
    under a fresh id and returns its reply (that reply is not also queued for ``recv``). Closing releases
    the pins this session pressed with ``autorelease``. Thread-safe; usable as a context manager."""

    def __init__(self, bridge: 'GpioBridge', client_id: int, addr: str = '',
                 listener: Callable[[dict | None], None] | None = None, internal: bool = False):
        self.bridge, self.client, self.addr = bridge, client_id, addr
        self.listener, self.internal = listener, internal
        self.holds: set[str] = set()
        self.closed = False
        self._q: queue.Queue = queue.Queue()
        self._waiting: dict[str, dict] = {}
        self._expired: set[str] = set()
        self._wlock = threading.Lock()
        self._req_ids = itertools.count(1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f'<gpio.Session {self.client}{" " + self.addr if self.addr else ""}{" closed" if self.closed else ""}>'

    # -- bridge side
    def _deliver(self, msg: dict | None):
        """Called by the bridge (under its lock): a reply or broadcast for this client; None = closed."""
        if msg is not None:
            rid = msg.get('id')
            if isinstance(rid, str):
                with self._wlock:
                    w = self._waiting.get(rid)
                    late = w is None and rid in self._expired
                    self._expired.discard(rid)
                if w is not None:
                    w['reply'] = msg
                    w['event'].set()
                    return
                if late:
                    return                               # reply to a request() that already timed out
        if self.listener is not None:
            self.listener(msg)
        else:
            self._q.put(msg)

    # -- client side
    def send(self, msg: dict) -> None:
        """Send one command (``{'cmd': 'trigger', 'pin': 26, 'ms': 300, 'id': 1}``); replies arrive via recv."""
        if self.closed:
            raise ConnectionError('GPIO session closed')
        self.bridge._client_message(self, msg)

    def send_text(self, raw: str) -> None:
        """Send one JSON text frame (as a websocket client would): malformed JSON gets an error reply."""
        try:
            msg = json.loads(raw)
        except ValueError:
            return self.bridge._reply_error(self, _NOID, 'invalid JSON')
        self.send(msg)

    def recv(self, timeout: float | None = None) -> dict | None:
        """The next reply or broadcast, or None on timeout / once the session is closed and drained."""
        if self.closed and self._q.empty():
            return None
        try:
            return self._q.get(timeout=timeout) if timeout is None or timeout > 0 else self._q.get_nowait()
        except queue.Empty:
            return None

    def request(self, msg: dict, timeout: float = 5.0) -> dict:
        """Send ``msg`` under a fresh id and return its ``ack`` / ``error`` / ``state`` reply (raises
        TimeoutError after ``timeout`` s). Only for commands the bridge answers with an id."""
        rid = f'_req{next(self._req_ids)}'
        w = {'event': threading.Event(), 'reply': None}
        with self._wlock:
            self._waiting[rid] = w
        try:
            self.send({**msg, 'id': rid})
            if not w['event'].wait(timeout):
                with self._wlock:
                    self._expired.add(rid)
                raise TimeoutError(f"no reply from the GPIO bridge within {timeout:.1f} s for {msg}")
        finally:
            with self._wlock:
                self._waiting.pop(rid, None)
        reply = dict(w['reply'])
        if 'id' in msg:
            reply['id'] = msg['id']
        else:
            reply.pop('id', None)
        return reply

    def close(self):
        if self.closed:
            return
        self.bridge._close_session(self)


class RemoteSession:
    """A :class:`Session` over the hub's websocket (``ws://host:port/api/gpio/ws``), for scripts that run
    outside the hub process. Same interface; ``closed`` turns true when the connection drops."""

    def __init__(self, url: str, timeout: float = 10.0):
        from websockets.sync.client import connect
        self.url = url
        self.ws = connect(url, open_timeout=timeout, close_timeout=1, proxy=None)
        self.closed = False
        self._q: queue.Queue = queue.Queue()
        self._waiting: dict[str, dict] = {}
        self._expired: set[str] = set()
        self._wlock = threading.Lock()
        self._req_ids = itertools.count(1)
        self._reader = threading.Thread(target=self._read, daemon=True, name='gpio-remote')
        self._reader.start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f'<gpio.RemoteSession {self.url}{" closed" if self.closed else ""}>'

    def _read(self):
        try:
            for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                rid = msg.get('id') if isinstance(msg, dict) else None
                if isinstance(rid, str):
                    with self._wlock:
                        w = self._waiting.get(rid)
                        late = w is None and rid in self._expired
                        self._expired.discard(rid)
                    if w is not None:
                        w['reply'] = msg
                        w['event'].set()
                        continue
                    if late:
                        continue
                self._q.put(msg)
        except Exception:                                # noqa: BLE001 - connection dropped
            pass
        finally:
            self.closed = True
            self._q.put(None)
            with self._wlock:
                for w in self._waiting.values():
                    w['event'].set()

    def send(self, msg: dict) -> None:
        if self.closed:
            raise ConnectionError(f'GPIO bridge connection closed ({self.url})')
        try:
            self.ws.send(_dumps(msg))
        except Exception as ex:                          # websockets ConnectionClosed
            self.closed = True
            raise ConnectionError(f'GPIO bridge connection closed ({self.url}): {ex}') from None

    def send_text(self, raw: str) -> None:
        self.ws.send(raw)

    def recv(self, timeout: float | None = None) -> dict | None:
        if self.closed and self._q.empty():
            return None
        try:
            msg = self._q.get(timeout=timeout) if timeout is None or timeout > 0 else self._q.get_nowait()
        except queue.Empty:
            return None
        return msg

    def request(self, msg: dict, timeout: float = 5.0) -> dict:
        rid = f'_req{next(self._req_ids)}'
        w = {'event': threading.Event(), 'reply': None}
        with self._wlock:
            self._waiting[rid] = w
        try:
            self.send({**msg, 'id': rid})
            if not w['event'].wait(timeout):
                with self._wlock:
                    self._expired.add(rid)
                raise TimeoutError(f'no reply from the GPIO bridge within {timeout:.1f} s for {msg}')
            if w['reply'] is None:
                raise ConnectionError(f'GPIO bridge connection closed ({self.url})')
        finally:
            with self._wlock:
                self._waiting.pop(rid, None)
        reply = dict(w['reply'])
        if 'id' in msg:
            reply['id'] = msg['id']
        else:
            reply.pop('id', None)
        return reply

    def close(self):
        self.closed = True
        try:
            self.ws.close()
        except Exception:                                # noqa: BLE001
            pass


def bridge_url(host: str) -> str:
    """``'host:port'`` (or ``http://host:port``) -> ``ws://host:port/api/gpio/ws``; ws:// URLs pass through."""
    host = host.strip()
    if host.startswith(('ws://', 'wss://')):
        return host
    host = host.removeprefix('http://').removeprefix('https://').rstrip('/')
    return f'ws://{host}{WS_PATH}'


def describe(bridge: 'GpioBridge | str | None') -> str:
    """For log lines: where a bridge argument points."""
    if isinstance(bridge, GpioBridge):
        return f'in-process GPIO bridge ({bridge.target})'
    return bridge_url(bridge or DEFAULT_HOST)


def connect(bridge: 'GpioBridge | str | None' = None, timeout: float = 10.0) -> 'Session | RemoteSession':
    """A session on ``bridge``: a :class:`GpioBridge` (in-process), a ``'host:port'`` of a hub (websocket),
    or None for the local hub (localhost:8888)."""
    if isinstance(bridge, GpioBridge):
        return bridge.session()
    return RemoteSession(bridge_url(bridge or DEFAULT_HOST), timeout=timeout)


def health(bridge: 'GpioBridge | str | None' = None, timeout: float = 2.0) -> dict:
    """``status()`` of a bridge object, or GET /api/gpio/health of a hub."""
    if isinstance(bridge, GpioBridge):
        return bridge.status()
    import urllib.request
    host = (bridge or DEFAULT_HOST).strip().removeprefix('http://').rstrip('/')
    with urllib.request.urlopen(f'http://{host}/api/gpio/health', timeout=timeout) as r:
        return json.loads(r.read())


# ======================================================================== the bridge
class GpioBridge:
    """The shared ESP32 link (see the module docstring)."""

    def __init__(self, target: str = DEFAULT_TARGET, hardware_map_path=None, log=print,
                 link: Callable[..., Link] | None = None):
        self.target = target
        self.hardware_map_path = Path(hardware_map_path or hardware_map())
        self.log = log
        self._link_cls = link
        self._lock = threading.RLock()
        self._sessions: dict[int, Session] = {}
        self._client_ids = itertools.count(1)
        self._up_ids = itertools.count(1)
        self._pending: dict[int, dict] = {}
        self._link: Link | None = None
        self._open = False
        self._last_state: dict | None = None
        self._last_activity = 0.0
        self._pins: dict[int, dict] = {}
        self._reconnect_at: float | None = None
        self._closed = False
        self._started = False
        self._loop: _LoopThread | None = None
        self._timer: threading.Thread | None = None
        self._stop = threading.Event()
        labels = pin_labels(self.hardware_map_path)
        if labels is None:
            self.log(f"Couldn't read pin names from {self.hardware_map_path}")
        self.labels: dict[str, str] = {str(k): v for k, v in (labels or {}).items()}
        self._internal = Session(self, 0, internal=True)

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> 'GpioBridge':
        with self._lock:
            if self._started:
                return self
            self._started = True
        self._loop = _LoopThread()
        self._timer = threading.Thread(target=self._tick, daemon=True, name='gpio-timer')
        self._timer.start()
        self._connect()
        self.log(f'GPIO bridge: ESP32 at {self.target}')
        return self

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sessions = list(self._sessions.values())
        for s in sessions:
            s.close()
        self._stop.set()
        with self._lock:
            link, self._link = self._link, None
            self._open = False
            self._fail_pending('ESP32 disconnected')
        if link is not None:
            link.close()
        if self._timer is not None:
            self._timer.join(2)
        if self._loop is not None:
            self._loop.stop()

    @property
    def online(self) -> bool:
        return self._open

    esp32 = online

    def status(self) -> dict:
        return {'esp32': self._open, 'target': self.target}

    def info(self) -> dict:
        st = self.status()
        return {**st, 'sessions': len(self._sessions)}

    def labels_dict(self) -> dict:
        return dict(self.labels)

    # ---------------------------------------------------------------- sessions
    def session(self, addr: str = '', listener: Callable[[dict | None], None] | None = None) -> Session:
        """A new client. ``listener`` (called under the bridge lock, must not block) replaces the queue."""
        with self._lock:
            if self._closed:
                raise ConnectionError('GPIO bridge closed')
            s = Session(self, next(self._client_ids), addr, listener)
            self._sessions[s.client] = s
            s._deliver(self._bridge_msg())
            s._deliver(self._labels_msg())
            if self._last_state is not None:
                s._deliver(dict(self._last_state))
        return s

    def _close_session(self, s: Session):
        with self._lock:
            if s.closed:
                return
            s.closed = True
            self._sessions.pop(s.client, None)
            # Release anything this client was holding with autorelease.
            for pin in list(s.holds):
                n = _js_number(pin)
                target = pin if pin == 'all' or math.isnan(n) else (int(n) if n.is_integer() else n)
                self._send_upstream({'cmd': 'release', 'pin': target})
            s.holds.clear()
            s._deliver(None)

    def _publish(self, msg: dict):
        for s in list(self._sessions.values()):
            s._deliver(dict(msg))

    def _reply(self, s: Session, msg: dict):
        if not s.closed:
            s._deliver(msg)

    def _reply_error(self, s: Session, rid, error: str):
        self._reply(s, {'type': 'error', 'error': error} if rid is _NOID else {'type': 'error', 'id': rid, 'error': error})

    def _bridge_msg(self):
        return {'type': 'bridge', 'esp32': self._open, 'target': self.target}

    def _labels_msg(self):
        return {'type': 'labels', 'labels': dict(self.labels)}

    # ---------------------------------------------------------------- client commands
    def _client_message(self, s: Session, msg):
        if not isinstance(msg, dict):
            return self._reply_error(s, _NOID, 'expected a JSON object')
        rid = msg.get('id', _NOID)
        cmd = msg.get('cmd')
        if cmd == 'set_label':
            n = _js_number(msg['pin']) if 'pin' in msg else math.nan
            if math.isnan(n) or not n.is_integer() or (self._pins and int(n) not in self._pins):
                return self._reply_error(s, rid, 'unknown pin')
            label = msg.get('label')
            self.set_label(int(n), label.strip()[:40] if isinstance(label, str) else '')
            return
        with self._lock:
            if cmd == 'state' and msg.get('fresh') is True:
                # A fresh read goes to the firmware with its own request id, and the firmware echoes that id on
                # its state reply, so a pressed-state broadcast already queued can never satisfy a new read.
                if not self._send_upstream({'cmd': 'state', 'fresh': True}, s, rid):
                    self._reply_error(s, rid, 'ESP32 not connected')
                return
            if cmd == 'state' and self._open and self._last_state is not None:
                reply = dict(self._last_state)
                if rid is not _NOID:
                    reply['id'] = rid
                return self._reply(s, reply)
            if not self._send_upstream(msg, s, rid):
                return self._reply_error(s, rid, 'ESP32 not connected')
            self._track_holds(s, msg)

    @staticmethod
    def _track_holds(s: Session, msg: dict):
        pin = _js_str(msg.get('pin', _NOID)).lower()
        cmd = {'low': 'press', 'high': 'release'}.get(msg.get('cmd'), msg.get('cmd'))
        if cmd == 'press' and msg.get('autorelease') not in (None, False, 0, ''):
            s.holds.add(pin)
        elif cmd in ('release', 'trigger'):
            if pin == 'all':
                s.holds.clear()
            else:
                s.holds.discard(pin)

    def _send_upstream(self, msg: dict, s: Session | None = None, rid=_NOID) -> bool:
        if self._link is None or not self._open:
            return False
        if s is not None:
            up = next(self._up_ids)
            self._pending[up] = {'session': s, 'id': rid, 'sent_at': time.monotonic(), 'fresh': msg.get('fresh') is True}
            msg = {**msg, 'id': up}
        else:
            msg = {k: v for k, v in msg.items() if k != 'id'}
        self._link.send(_dumps(msg))
        return True

    def _fail_pending(self, error: str, older_than: float = 0.0):
        """Answer pending requests sent at least ``older_than`` s ago with ``error`` (default: all of them,
        rather than leaving them to the 5 s reply timeout)."""
        now = time.monotonic()
        for up, p in list(self._pending.items()):
            if now - p['sent_at'] < older_than:
                continue
            del self._pending[up]
            self._reply_error(p['session'], p['id'], error)

    # ---------------------------------------------------------------- upstream events
    def _connect(self):
        with self._lock:
            if self._closed:
                return
            self._last_activity = time.monotonic()
            self._reconnect_at = None
            cls = self._link_cls or LINKS.get(self.target.split('://', 1)[0] if '://' in self.target else 'ble')
            if cls is None:
                self.log(f'GPIO bridge: unknown target {self.target!r} (use ble://<name> or ws://host:port/ws)')
                return
            link = None

            def events(kind, data=None):
                self._link_event(link, kind, data)
            link = cls(self.target, events, self._loop.loop)
            self._link = link
        link.start()

    def _link_event(self, link: Link, kind: str, data):
        if kind == 'log':
            self.log(f'GPIO bridge: {data}')
            return
        with self._lock:
            if self._link is not link or link is None:
                return
            if kind == 'activity':
                self._last_activity = time.monotonic()
            elif kind == 'up':
                self._open = True
                self._last_activity = time.monotonic()
                self.log(f'ESP32 connected: {self.target}')
                self._publish(self._bridge_msg())
            elif kind == 'msg':
                self._upstream_message(data)
            elif kind == 'down':
                self._upstream_closed(link)

    def _upstream_message(self, text: str):
        self._last_activity = time.monotonic()
        try:
            msg = json.loads(text)
        except ValueError:
            return
        if not isinstance(msg, dict):
            return
        if msg.get('type') == 'state':
            # Replies to a state request carry its id; broadcasts and heartbeat replies don't.
            up = msg.get('id')
            state = {k: v for k, v in msg.items() if k != 'id'}
            if state != self._last_state:                    # heartbeats repeat the same state: fan out changes only
                if isinstance(state.get('pins'), list):
                    self._update_pins(state['pins'])
                self._publish(state)
            self._last_state = state
            p = self._pending.pop(up, None) if isinstance(up, int) and not isinstance(up, bool) else None
            if p:
                reply = {**state, **({'fresh': True, 'source': 'firmware'} if p['fresh'] else {})}
                if p['id'] is not _NOID:
                    reply['id'] = p['id']
                self._reply(p['session'], reply)
            return
        # ack / error: route back to whoever sent the request, restoring their id.
        up = msg.get('id')
        p = self._pending.pop(up, None) if isinstance(up, int) and not isinstance(up, bool) else None
        if not p:
            return                                           # e.g. the bridge's own auto-release commands
        reply = {k: v for k, v in msg.items() if k != 'id'}
        if p['id'] is not _NOID:
            reply['id'] = p['id']
        self._reply(p['session'], reply)

    def _update_pins(self, pins: list):
        self._pins = {p['pin']: p for p in pins if isinstance(p, dict) and 'pin' in p}

    def _upstream_closed(self, link: Link):
        if self._link is not link:
            return                                           # already handled (e.g. dropped by the watchdog)
        if self._open:
            self.log('ESP32 disconnected; retrying...')
        self._open = False
        self._link = None
        self._last_state = None
        self._fail_pending('ESP32 disconnected')
        self._publish(self._bridge_msg())
        if not self._closed:
            self._reconnect_at = time.monotonic() + RECONNECT_S

    def _tick(self):
        """Heartbeat + upstream watchdog, reply timeouts and reconnects."""
        next_beat = time.monotonic() + HEARTBEAT_S
        next_sec = time.monotonic() + 1.0
        while not self._stop.wait(0.1):
            now = time.monotonic()
            reconnect = False
            with self._lock:
                if self._reconnect_at is not None and now >= self._reconnect_at and self._link is None:
                    reconnect = True
                if now >= next_beat:
                    next_beat = now + HEARTBEAT_S
                    link = self._link
                    if link is not None:
                        if now - self._last_activity > UPSTREAM_TIMEOUT_S:
                            self.log('ESP32 stopped responding' if self._open else 'ESP32 connect timed out')
                            self._upstream_closed(link)
                            link.close()
                        elif self._open:
                            link.send(_dumps({'cmd': 'state'}))
                if now >= next_sec:
                    next_sec = now + 1.0
                    self._fail_pending('ESP32 did not reply', REPLY_TIMEOUT_S)
            if reconnect:
                self._connect()

    # ---------------------------------------------------------------- labels
    def set_label(self, pin: int, label: str):
        """Rename a joint pin; saved to its joint in the hardware map (digital_twin.sim.pins) and broadcast to every
        session. A pin no joint uses, or an empty name, is refused (logged) and the map's names are re-sent."""
        with self._lock:
            label = (label or '').strip()[:40]
            labels = {**self.labels, str(pin): label}
            try:
                self.labels = save_pin_labels(labels, self.hardware_map_path)
            except (OSError, ValueError) as ex:
                self.log(f"Couldn't write {self.hardware_map_path}: {ex}")
                self.labels = {str(k): v for k, v in (pin_labels(self.hardware_map_path) or {}).items()} or self.labels
            self._publish(self._labels_msg())

    # ---------------------------------------------------------------- in-process helpers
    def release_all(self, timeout: float = 2.0) -> dict:
        """``release all`` (the E-STOP): never raises. {'ok', 'sent', 'reply' | 'error', 'ms'}."""
        t0 = time.monotonic()
        sent = self._open
        try:
            reply = self._internal.request({'cmd': 'release', 'pin': 'all'}, timeout=timeout)
            ok = reply.get('type') == 'ack'
            out = {'ok': ok, 'sent': reply.get('error') != 'ESP32 not connected'}
            out.update({'reply': reply} if ok else {'error': str(reply.get('error') or reply)})
        except Exception as ex:                              # noqa: BLE001 - E-STOP must not raise
            out = {'ok': False, 'sent': sent, 'error': f'{type(ex).__name__}: {ex}'}
        out['ms'] = round((time.monotonic() - t0) * 1000, 1)
        return out

    def trigger(self, pin: int, ms: int, timeout: float = 3.0) -> dict:
        """One timed press; returns the ack. RuntimeError on an error reply, TimeoutError without one."""
        reply = self._internal.request({'cmd': 'trigger', 'pin': pin, 'ms': int(ms)}, timeout=timeout)
        if reply.get('type') == 'error':
            raise RuntimeError(f"GPIO {pin}: {reply.get('error')}")
        return reply

    def state(self, fresh: bool = True, timeout: float = 3.0) -> dict:
        """Current pin states (``fresh``: asked from the firmware, not the cache)."""
        reply = self._internal.request({'cmd': 'state', **({'fresh': True} if fresh else {})}, timeout=timeout)
        if reply.get('type') == 'error':
            raise RuntimeError(str(reply.get('error')))
        return reply
