"""The inference event bus: JSON events fanned out to the page's Server-Sent Events streams."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque

from hub.apps.inference.util import clean


class EventBus:
    """Fan-out of JSON events to SSE subscribers (bounded queues; a slow client drops events)."""

    def __init__(self, backlog: int = 300):
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._seq = 0
        self.recent = deque(maxlen=backlog)       # non-sensor events, replayed to new subscribers

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=2000)
        with self._lock:
            self._subs.add(q)
            backlog = list(self.recent)
        for ev in backlog:
            q.put_nowait(ev)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    @property
    def n_subscribers(self) -> int:
        with self._lock:
            return len(self._subs)

    def publish(self, typ: str, data) -> dict:
        with self._lock:
            self._seq += 1
            ev = {'id': self._seq, 'type': typ, 't': time.time(), 'data': clean(data)}
            if typ != 'sensors':
                self.recent.append(ev)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass
        return ev
