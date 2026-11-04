"""GPIO bridge app: the hub holds the ESP32 link (digital_twin.hardware.gpio.GpioBridge) and serves its protocol.

    GET  /api/gpio/health                  {"esp32", "target", "recording"}
    GET  /api/gpio/labels                  {"<gpio>": "<name>", ...}
    GET  /api/gpio/recordings              [{id, name, startedAt, durationMs, events, bytes, complete}]
    GET  /api/gpio/recordings/<id>         JSONL (?format=csv for CSV)
    POST /api/gpio/recordings/start        {"name": "..."} (optional)
    POST /api/gpio/recordings/stop
    WS   /api/gpio/ws                      the firmware protocol (rc-control/src/main.cpp), one Session per socket

The service is the GpioBridge itself, so other apps call it directly (``hub.service('gpio')``).
"""
from __future__ import annotations

import asyncio
import json

import anyio.to_thread
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from digital_twin.hardware.gpio import DEFAULT_TARGET, RECORDINGS, GpioBridge, default_target


def start(hub, target: str = DEFAULT_TARGET, hardware_map: str | None = None, recordings: str | None = None) -> GpioBridge:
    """``target``: ble://<name> (default ble://gpio; BLE_ADDRESS / BLE_NAME / ESP32_URL in the environment
    still apply when it is the default) or ws://host:port/ws (an ESP32 stand-in). ``hardware_map`` (pin names
    live on its joints: ``joints.<j>.positive`` / ``negative``) / ``recordings``: optional paths (default configs/hardware_map.json, rc-control/recordings/)."""
    from hub.app import rel
    if target == DEFAULT_TARGET:
        target = default_target()
    return GpioBridge(target, hardware_map_path=rel(hardware_map) if hardware_map else None,
                      recordings_dir=rel(recordings) if recordings else RECORDINGS).start()


def routes(get) -> APIRouter:
    r = APIRouter()

    @r.get('/health')
    def health():
        return get().status()

    @r.get('/labels')
    def labels():
        return get().labels_dict()

    @r.get('/recordings')
    def recordings():
        return get().recordings()

    @r.post('/recordings/start')
    async def recordings_start(request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        name = body.get('name') if isinstance(body, dict) else None
        bridge = get()
        return await anyio.to_thread.run_sync(bridge.start_recording, name)

    @r.post('/recordings/stop')
    def recordings_stop():
        return get().stop_recording()

    @r.get('/recordings/{rid}')
    def recording(rid: str, format: str = ''):
        bridge = get()
        if format == 'csv':
            body = bridge.recording_csv(rid)
            if body is None:
                raise HTTPException(404, 'not found')
            return Response(body, media_type='text/csv',
                            headers={'Content-Disposition': f'attachment; filename="{rid}.csv"'})
        path = bridge.recording_path(rid)
        if path is None:
            raise HTTPException(404, 'not found')
        bridge.recorder.flush()
        return FileResponse(path, media_type='application/x-ndjson', filename=f'{rid}.jsonl')

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
            session = bridge.session('ws', addr=addr, listener=listener)
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
