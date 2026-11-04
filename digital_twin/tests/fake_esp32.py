"""A fake ESP32 for the GPIO bridge tests (digital_twin/tests/test_gpio_bridge.py, hub/tests/test_gpio.py).

The fake is a :class:`gpio.Link` that models the firmware (hardware/rc-control/src/main.cpp):
eight push-high pins, trigger / press / release / state, state broadcasts on change, request ids
echoed on replies. Importing this module registers it as the ``fake://`` link. One module (not a test file), so
both test files share the same class and its ``instances``.
"""
import json
import queue
import threading
import time

from digital_twin.hardware import gpio

PINS = [32, 33, 25, 26, 19, 21, 22, 23]


class FakeEsp32(gpio.Link):
    """In-process ESP32 stand-in. ``silent``: swallow commands (no replies); ``drop()``: lose the link."""
    instances: list['FakeEsp32'] = []

    def start(self):
        self.pressed = {p: False for p in PINS}
        self.timers: dict[int, threading.Timer] = {}
        self.rx: list[dict] = []
        self.silent = False
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.alive = True
        FakeEsp32.instances.append(self)
        threading.Thread(target=self._work, daemon=True).start()
        self.q.put(('up', None))

    def state(self):
        return {'type': 'state', 'pins': [{'pin': p, 'name': f'D{p}', 'mode': 'push-high', 'pressed': self.pressed[p]}
                                          for p in PINS]}

    def emit(self, msg):
        if self.alive:
            self.events('msg', json.dumps(msg))

    def _set(self, pin, on):
        with self.lock:
            t = self.timers.pop(pin, None)
            if t:
                t.cancel()
            if self.pressed[pin] == on:
                return
            self.pressed[pin] = on
        self.q.put(('broadcast', None))

    def _work(self):
        while True:
            kind, text = self.q.get()
            if kind == 'stop':
                return
            if kind == 'up':
                self.events('up', 'fake')
                self.emit(self.state())
                continue
            if kind == 'broadcast':
                self.emit(self.state())
                continue
            m = json.loads(text)
            self.rx.append(m)
            if self.silent:
                continue
            rid = m.get('id')

            def send(msg):
                if rid is not None:
                    msg['id'] = rid
                self.emit(msg)
            if m.get('cmd') == 'state':
                send(self.state()); continue
            cmd = {'low': 'press', 'high': 'release'}.get(m.get('cmd'), m.get('cmd'))
            if cmd not in ('trigger', 'press', 'release'):
                send({'type': 'error', 'error': 'unknown cmd (use trigger, press, release, state)'}); continue
            pin = m.get('pin')
            pins = PINS if pin == 'all' else [pin] if pin in PINS else \
                [int(pin[1:])] if isinstance(pin, str) and pin[1:].isdigit() and int(pin[1:]) in PINS else None
            if pins is None:
                send({'type': 'error', 'error': 'pin not controllable'}); continue
            ms = m.get('ms', 200)
            if not isinstance(ms, int) or not 1 <= ms <= 60000:
                send({'type': 'error', 'error': 'ms must be 1..60000'}); continue
            for p in pins:
                self._set(p, cmd != 'release')
                if cmd == 'trigger':
                    t = threading.Timer(ms / 1000, self._set, (p, False)); t.daemon = True
                    with self.lock:
                        self.timers[p] = t
                    t.start()
            send({'type': 'ack', 'cmd': m.get('cmd'), 'pin': pin})

    def send(self, text):
        self.q.put(('cmd', text))

    def close(self):
        self.alive = False
        self.q.put(('stop', None))

    def drop(self):
        """The ESP32 vanished: the link ends."""
        self.close()
        self.events('down', None)


gpio.LINKS['fake'] = FakeEsp32


def wait_for(cond, timeout=3.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def recv_until(session, pred, timeout=3.0):
    t_end = time.monotonic() + timeout
    seen = []
    while time.monotonic() < t_end:
        m = session.recv(max(0.01, t_end - time.monotonic()))
        if m is None:
            continue
        seen.append(m)
        if pred(m):
            return m
    raise AssertionError(f'no matching message; got {seen}')


def drain(session, quiet=0.15):
    out = []
    while (m := session.recv(quiet)) is not None:
        out.append(m)
    return out


def fake() -> FakeEsp32:
    return FakeEsp32.instances[-1]
