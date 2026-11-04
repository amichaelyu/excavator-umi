"""Helpers shared by several hub apps: JPEG encode / decode, the excavator Pi's default host list, the
live-camera fetch (twin viewer live pane, inference Pi camera) and single-range HTTP file responses (run
videos, replay media)."""
from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from fastapi.responses import Response, StreamingResponse

from digital_twin.hardware.pi import DEFAULT_PI_HOST  # noqa: F401  (the one definition; inference)

LIVE_MAX_BYTES = 8 << 20
RANGE_CHUNK = 256 * 1024


# ------------------------------------------------------------------ JPEG
def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)).save(buf, 'JPEG', quality=quality)
    return buf.getvalue()


def decode_jpeg(body: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(body)).convert('RGB'))


def extract_jpeg(buf: bytes) -> bytes | None:
    """First complete JPEG (SOI..EOI) in a byte buffer, e.g. one MJPEG multipart part."""
    s = buf.find(b'\xff\xd8')
    if s < 0:
        return None
    e = buf.find(b'\xff\xd9', s + 2)
    return None if e < 0 else buf[s:e + 2]


# ------------------------------------------------------------------ live camera
def fetch_live_frame(url: str, token: str = '', timeout: float = 2.0) -> tuple[int, bytes, str]:
    """(status, body, content_type) for the latest live JPEG. 204 = no new frame; 502 = error JSON."""
    headers = {'Accept': 'image/jpeg, multipart/x-mixed-replace, */*', 'User-Agent': 'hub.apps.twin_viewer.core'}
    if token:
        headers['Cookie'] = f'umi_access={token}'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
            if resp.status == 204:
                return 204, b'', ''
            ctype = resp.headers.get('Content-Type', '')
            if ctype.startswith('multipart/'):
                buf = b''
                while len(buf) < LIVE_MAX_BYTES:
                    chunk = resp.read1(65536) if hasattr(resp, 'read1') else resp.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    jpg = extract_jpeg(buf)
                    if jpg:
                        return 200, jpg, 'image/jpeg'
                raise ValueError('no JPEG found in the MJPEG stream')
            body = resp.read(LIVE_MAX_BYTES + 1)
            if len(body) > LIVE_MAX_BYTES:
                raise ValueError('live image too large')
            if not body.startswith(b'\xff\xd8'):
                raise ValueError(f'live source did not return a JPEG (Content-Type {ctype or "unknown"})')
            return 200, body, 'image/jpeg'
    except urllib.error.HTTPError as ex:
        hint = ' (check the dashboard token: --live-token-file or ?token=)' if ex.code in (401, 403) else ''
        msg = f'live source returned HTTP {ex.code}{hint}'
    except (urllib.error.URLError, OSError, ValueError) as ex:
        msg = f'live source unreachable: {getattr(ex, "reason", ex)}'
    return 502, json.dumps({'error': msg}).encode(), 'application/json'


# ------------------------------------------------------------------ HTTP Range
def parse_range(header: str | None, size: int):
    """Parse a single-range ``Range: bytes=`` header.

    Returns None (no/ignored range -> 200 full body), (start, end) inclusive, or raises
    ValueError for an unsatisfiable range (-> 416). Multi-range requests fall back to the first range.
    """
    if not header:
        return None
    m = re.match(r'^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*(?:,.*)?$', header)
    if not m or (m.group(1) == '' and m.group(2) == ''):
        return None
    a, b = m.group(1), m.group(2)
    if a == '':                                   # suffix: last N bytes
        n = int(b)
        if n == 0 or size == 0:
            raise ValueError('unsatisfiable')
        return max(0, size - n), size - 1
    start = int(a)
    end = size - 1 if b == '' else min(int(b), size - 1)
    if start >= size or end < start:
        raise ValueError('unsatisfiable')
    return start, end


def range_response(f: Path, range_header: str | None, head: bool, media_type: str, headers=None) -> Response:
    """``f`` with single-range support (videos seek): 200, 206 or 416; HEAD and empty files get no
    body; the file is streamed in RANGE_CHUNK pieces. ``headers`` are added to the 200 / 206 headers."""
    size = f.stat().st_size
    try:
        rng = parse_range(range_header, size)
    except ValueError:
        return Response(b'', 416, media_type='text/plain',
                        headers={'Content-Range': f'bytes */{size}', 'Accept-Ranges': 'bytes'})
    start, end = rng if rng else (0, size - 1)
    length = end - start + 1 if size else 0
    out = {'Content-Length': str(length), 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache', **(headers or {})}
    if rng:
        out['Content-Range'] = f'bytes {start}-{end}/{size}'
    status = 206 if rng else 200
    if head or not length:
        return Response(b'', status, headers=out, media_type=media_type)

    def chunks():
        with open(f, 'rb') as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(RANGE_CHUNK, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk
    return StreamingResponse(chunks(), status, headers=out, media_type=media_type)
