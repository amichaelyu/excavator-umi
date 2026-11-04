"""GPIO bridge app: the hub holds the ESP32 link (digital_twin.hardware.gpio.GpioBridge) and serves its protocol.

    GET  /api/gpio/health                  {"esp32", "target"}
    GET  /api/gpio/labels                  {"<gpio>": "<name>", ...}
    WS  /api/gpio/ws                      the firmware protocol (hardware/rc-control/src/main.cpp), one Session per socket

The service is the GpioBridge itself, so other apps call it directly (``hub.service('gpio')``).
"""
from __future__ import annotations

import asyncio
import json

import anyio.to_thread
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from digital_twin.hardware.gpio import DEFAULT_TARGET, GpioBridge, default_target


def start(hub, target: str = DEFAULT_TARGET, hardware_map: str | None = None) -> GpioBridge:
    """``target``: ble://<name> (default ble://gpio; BLE_ADDRESS / BLE_NAME / ESP32_URL in the environment
    still apply when it is the default) or ws://host:port/ws (an ESP32 stand-in). ``hardware_map``: optional
    path (default configs/hardware/hardware_map.json; pin names live on its joints: ``joints.<j>.positive`` / ``negative``)."""
    from hub.app import rel
    if target == DEFAULT_TARGET:
        target = default_target()
    return GpioBridge(target, hardware_map_path=rel(hardware_map) if hardware_map else None).start()


def routes(get) -> APIRouter:
    r = APIRouter()

    @r.get('/health')
    def health():
        return get().status()

    @r.get('/labels')
    def labels():
        return get().labels_dict()

    @r.websocket('/ws')
    async def ws(websocket: WebSocket):
        try:
            bridge = get()
        except HTTPException:
            await websocket.close(code=1013)             # try again later: the bridge is stopped / starting
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()

        def listener(msg):                              # bridge thread -> this event loop; never blocks
            try:
                loop.call_soon_threadsafe(out.put_nowait, msg)
            except RuntimeError:                         # loop closed (hub shutting down)
                pass
        addr = websocket.client.host if websocket.client else ''
        try:
            session = bridge.session(addr=addr, listener=listener)
        except ConnectionError:
            await websocket.close(code=1013)
            return

        async def pump_out():
            while True:
                msg = await out.get()
                if msg is None:                          # session closed (bridge stopped)
                    await websocket.close(code=1001)
                    return
                await websocket.send_text(json.dumps(msg, separators=(',', ':'), ensure_ascii=False))

        sender = asyncio.create_task(pump_out())
        try:
            while True:
                recv = asyncio.create_task(websocket.receive())
                done, _ = await asyncio.wait({recv, sender}, return_when=asyncio.FIRST_COMPLETED)
                if sender in done:
                    recv.cancel()
                    break
                m = recv.result()
                if m['type'] == 'websocket.disconnect':
                    break
                text = m.get('text')
                if text is None and m.get('bytes') is not None:
                    text = m['bytes'].decode(errors='replace')
                if text is not None:
                    await anyio.to_thread.run_sync(session.send_text, text)
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            pass
        finally:
            sender.cancel()
            session.close()                              # quick (no I/O wait), and must run even when cancelled

    return r
