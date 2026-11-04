"""Local live preview for a USB MJPEG camera and BerryIMUv3.

Requires python3-smbus2, python3-pil and v4l-utils on Raspberry Pi OS. A smaller
JPEG is made for the live preview. IMU read times are host receipt times, not
hardware sample timestamps. The live preview itself is not a complete training-data
collection pipeline. Camera calibration (lens, camera/IMU) runs on a Mac from a saved
recording: python -m policy.processing.camera_calibration.

More LSM6DS-family IMUs (the excavator's boom and stick IMUs) are listed in
``imus.json`` in the data directory (see ``__main__``): ``[{"name": "bucket", "bus": 1, "address": "0x6a"}, ...]``.
The ``bucket`` entry (else the first) is the primary IMU: it alone feeds ``imu`` in
/api/status, /api/imu/recent and recordings. Every IMU, primary included, is
reported under ``imus`` in /api/status for the inference-time joint estimator, and
``/api/imu/recent?imu=<name>`` serves any IMU's full-rate history (arm calibration);
the extra IMUs are never recorded and never stop a recording.
"""
from collections import deque
import hmac
import io
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import select
import socket
from socketserver import TCPServer
import struct
import subprocess
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .capture import TimedFrames
from .download import RecordingDownload
from .time_alignment import TimeAlignment


class JPEGFrames:
    """Extract complete JPEGs across arbitrary pipe read boundaries.

Skip length-delimited segments, including embedded thumbnail bytes. In scan
data, byte stuffing and restart markers cannot be confused with end-of-image.
"""
    def __init__(self):
        self.buf = bytearray()
        self.pos = 2

    def feed(self, data):
        self.buf.extend(data)
        frames = []
        while True:
            if not self.buf.startswith(b"\xff\xd8"):
                start = self.buf.find(b"\xff\xd8")
                if start < 0:
                    self.buf[:] = self.buf[-1:] if self.buf.endswith(b"\xff") else b""
                    self.pos = 2
                    break
                del self.buf[:start]
                self.pos = 2
            marker = self.buf.find(b"\xff", self.pos)
            if marker < 0:
                self.pos = len(self.buf)
                break
            if marker + 1 >= len(self.buf):
                self.pos = marker
                break
            code = self.buf[marker + 1]
            if code == 0xD9:
                frames.append(bytes(self.buf[:marker + 2]))
                del self.buf[:marker + 2]
                self.pos = 2
                continue
            if code == 0xD8:  # Recover from an interrupted frame.
                del self.buf[:marker]
                self.pos = 2
                continue
            if code in (0x00, 0x01) or 0xD0 <= code <= 0xD7:
                self.pos = marker + 2
                continue
            if code == 0xFF:
                self.pos = marker + 1
                continue
            if marker + 4 > len(self.buf):
                self.pos = marker
                break
            length = int.from_bytes(self.buf[marker + 2:marker + 4], "big")
            if length < 2:
                del self.buf[:marker + 2]
                self.pos = 2
                continue
            end = marker + 2 + length
            if end > len(self.buf):
                self.pos = marker
                break
            self.pos = end
        if len(self.buf) > 8 * 1024 * 1024:
            self.buf.clear()
            self.pos = 2
            raise ValueError("Camera supplied an oversized or incomplete JPEG.")
        return frames


def rate(times):
    return (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0


PRIMARY_IMU = {'name': 'bucket', 'bus': 1, 'address': 0x6A, 'model': 'lsm6ds'}
# WHO_AM_I of LSM6DS parts sharing the registers and scale codes used here (CTRL1-3, STATUS, OUTX_L_G).
LSM6DS_IDS = {0x69: 'LSM6DS3', 0x6A: 'LSM6DSL', 0x6B: 'LSM6DSR/ISM330DHCX', 0x6C: 'LSM6DSO/LSM6DSOX'}
# WHO_AM_I of InvenSense parts sharing the MPU6050 registers used here (PWR_MGMT_1, CONFIG, ACCEL_XOUT_H).
MPU6050_IDS = {0x68: 'MPU6050', 0x70: 'MPU6500', 0x71: 'MPU9250', 0x73: 'MPU9255'}
IMU_ADDRESSES = {'lsm6ds': (0x6A, 0x6B), 'mpu6050': (0x68, 0x69)}


def load_imu_config(path):
    """IMU list from imus.json (missing file: the BerryIMUv3 on bus 1 only). Returns (primary, others)."""
    entries = [dict(PRIMARY_IMU)]
    if path is not None and Path(path).exists():
        raw = json.loads(Path(path).read_text())
        entries = []
        for item in raw:
            address = item['address']
            address = int(address, 16) if isinstance(address, str) else int(address)
            name = str(item['name'])
            if not name.isidentifier():
                raise ValueError(f'imus.json: invalid IMU name {name!r}')
            model = str(item.get('model', 'lsm6ds'))
            if model not in IMU_ADDRESSES:
                raise ValueError(f'imus.json: {name} model {model!r} is not one of {", ".join(IMU_ADDRESSES)}')
            if address not in IMU_ADDRESSES[model]:
                choices = ' or '.join(f'{a:#x}' for a in IMU_ADDRESSES[model])
                raise ValueError(f'imus.json: {name} address {address:#x} is not an {model} address ({choices})')
            entries.append({'name': name, 'bus': int(item['bus']), 'address': address, 'model': model})
        if not entries:
            raise ValueError('imus.json lists no IMUs')
    names = [e['name'] for e in entries]
    if len(set(names)) != len(names):
        raise ValueError('imus.json: IMU names must be unique')
    places = [(e['bus'], e['address']) for e in entries]
    if len(set(places)) != len(places):
        raise ValueError('imus.json: two IMUs share one bus and address')
    primary = next((e for e in entries if e['name'] == 'bucket'), entries[0])
    return primary, [e for e in entries if e is not primary]


IMU_GLITCH_RETRIES = 50              # consecutive failed I2C transfers (~0.15 s) before an IMU counts as lost
MEAN_SAMPLES = 12                     # ~58 ms at 208 Hz: the at-rest accel mean reported for joint estimation


def accel_mean(samples):
    """Mean accel [g] of the newest MEAN_SAMPLES readings (noise / sqrt(n) for an arm at rest), or None."""
    recent = list(samples)[-MEAN_SAMPLES:]
    if not recent:
        return None
    return [sum(s['accel'][i] for s in recent) / len(recent) for i in range(3)], len(recent)


def decode_imu(data):
    raw = struct.unpack("<6h", bytes(data))
    # LSM6DSL CTRL1_XL=0x58: +/-4 g; CTRL2_G=0x54: +/-500 deg/s.
    return {
        "accel": [v * 0.000122 for v in raw[3:]],
        "gyro": [v * 0.0175 for v in raw[:3]],
        "clipped": any(abs(v) >= 32700 for v in raw),
    }


def decode_mpu6050(data):
    raw = struct.unpack(">7h", bytes(data))
    # ACCEL_XOUT_H..GYRO_ZOUT_L, temperature in the middle. ACCEL_CONFIG=0x08: +/-4 g; GYRO_CONFIG=0x08: +/-500 deg/s.
    return {
        "accel": [v / 8192 for v in raw[:3]],
        "gyro": [v / 65.5 for v in raw[4:]],
        "clipped": any(abs(v) >= 32700 for v in raw[:3] + raw[4:]),
    }


class LSM6DS:
    """208 Hz, +/-4 g, +/-500 dps."""
    saved_registers = (0x10, 0x11, 0x12)

    def setup(self, bus, address, label):
        who = bus.read_byte_data(address, 0x0F)
        if who not in LSM6DS_IDS:
            raise RuntimeError(f"Expected an LSM6DS IMU ({label}); WHO_AM_I was {who:#04x}.")
        for reg, value in ((0x12, 0x44), (0x10, 0x58), (0x11, 0x54)):
            bus.write_byte_data(address, reg, value)

    decode = staticmethod(decode_imu)

    def ready(self, bus, address):
        return bus.read_byte_data(address, 0x1E) & 3 == 3

    def read(self, bus, address):
        return bus.read_i2c_block_data(address, 0x22, 12)


class MPU6050:
    """200 Hz (1 kHz / (1 + 4)), ~94 Hz low-pass, +/-4 g, +/-500 dps."""
    saved_registers = (0x6B, 0x19, 0x1A, 0x1B, 0x1C, 0x38)

    def setup(self, bus, address, label):
        who = bus.read_byte_data(address, 0x75)
        if who not in MPU6050_IDS:
            raise RuntimeError(f"Expected an MPU6050 IMU ({label}); WHO_AM_I was {who:#04x}.")
        # Wake on the gyro X clock, then rate, low-pass, ranges, and the data-ready flag.
        for reg, value in ((0x6B, 0x01), (0x19, 0x04), (0x1A, 0x02), (0x1B, 0x08), (0x1C, 0x08), (0x38, 0x01)):
            bus.write_byte_data(address, reg, value)

    decode = staticmethod(decode_mpu6050)

    def ready(self, bus, address):
        return bus.read_byte_data(address, 0x3A) & 1 == 1     # INT_STATUS.DATA_RDY, cleared by this read

    def read(self, bus, address):
        return bus.read_i2c_block_data(address, 0x3B, 14)


IMU_DRIVERS = {'lsm6ds': LSM6DS(), 'mpu6050': MPU6050()}


class State:
    def __init__(self, alignment=None):
        self.alignment = alignment or TimeAlignment()
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame = None
        self.sequence = 0
        self.frame_metadata = None
        self.recorder = None
        self.preview = None
        self.preview_error = ""
        self.preview_requested = float("-inf")
        self.camera_times = deque(maxlen=120)
        self.imu_times = deque(maxlen=208)
        self.imu_history = deque(maxlen=832)
        self.sample = None
        self.camera_error = "Starting camera…"
        self.imu_error = "Starting motion sensor…"
        self.started = time.monotonic()
        self.primary_imu = dict(PRIMARY_IMU)
        self.extra_imus = {}

    def configure_imus(self, primary, others):
        with self.condition:
            self.primary_imu = dict(primary)
            self.extra_imus = {e['name']: {'bus': e['bus'], 'address': e['address'], 'model': e.get('model', 'lsm6ds'),
                                           'times': deque(maxlen=208),
                                           'recent': deque(maxlen=MEAN_SAMPLES), 'history': deque(maxlen=832),
                                           'sample': None,
                                           'error': 'Starting motion sensor…'} for e in others}

    def extra_imu_error(self, name, message):
        with self.condition:
            self.extra_imus[name]['error'] = str(message)

    def error(self, device, message):
        with self.condition:
            setattr(self, device + "_error", str(message))
            self.condition.notify_all()
        if device in ('camera', 'imu') and self.recorder:
            label = 'Camera' if device == 'camera' else 'IMU'
            self.recorder.fail(f'{label} disconnected or failed: {message}. Recording stopped; captured data will be saved.')

    def frame_received(self, frame, metadata=None):
        with self.condition:
            self.frame = frame
            self.sequence += 1
            self.camera_times.append(time.monotonic())
            if metadata is not None:
                metadata = self.alignment.apply(dict(metadata, receipt_monotonic_ns=time.monotonic_ns()))
            self.frame_metadata = metadata
            self.camera_error = ""
            self.condition.notify_all()
        if self.recorder and metadata:
            self.recorder.put('frame', (frame, metadata))

    def sample_received(self, sample, name=None):
        if name is not None and name in self.extra_imus:
            with self.condition:
                entry = self.extra_imus[name]
                entry['sample'] = sample
                entry['recent'].append(sample)
                entry['history'].append(sample)
                entry['times'].append(time.monotonic())
                entry['error'] = ""
            return
        with self.condition:
            self.sample = sample
            self.imu_history.append(sample)
            self.imu_times.append(time.monotonic())
            self.imu_error = ""
        if self.recorder:
            self.recorder.put('imu', sample)

    def snapshot(self):
        with self.condition:
            now = time.monotonic()
            ca = now - self.camera_times[-1] if self.camera_times else None
            ia = now - self.imu_times[-1] if self.imu_times else None
            camera_live = ca is not None and ca < 2 and not self.camera_error
            imu_live = ia is not None and ia < 0.5 and not self.imu_error
            def entry_mean(samples, live):
                m = accel_mean(samples) if live else None
                return {"accel_mean": m[0], "mean_samples": m[1]} if m else {"accel_mean": None, "mean_samples": 0}
            imus = {self.primary_imu['name']: {
                "live": bool(imu_live), "age_s": ia, "hz": rate(self.imu_times) if imu_live else 0,
                "error": self.imu_error if not imu_live else "", "sample": self.sample if imu_live else None,
                **entry_mean(self.imu_history, imu_live),
                "bus": self.primary_imu['bus'], "address": self.primary_imu['address'], "primary": True}}
            for name, entry in self.extra_imus.items():
                age = now - entry['times'][-1] if entry['times'] else None
                live = age is not None and age < 0.5 and not entry['error']
                imus[name] = {"live": bool(live), "age_s": age, "hz": rate(entry['times']) if live else 0,
                              "error": entry['error'] if not live else "", "sample": entry['sample'] if live else None,
                              **entry_mean(entry['recent'], live),
                              "bus": entry['bus'], "address": entry['address'], "primary": False}
            return {
                "server_monotonic_s": now, "uptime_s": now - self.started,
                "time_alignment": self.alignment.snapshot(self.frame_metadata if camera_live else None),
                "camera": {"live": bool(camera_live), "age_s": ca,
                           "fps": rate(self.camera_times) if camera_live else 0,
                           "frames": self.sequence, "preview_fps_limit": 20,
                           "preview_size": [640, 360],
                           "timing": self.frame_metadata,
                           "error": self.camera_error if not camera_live else ""},
                "imu": {"live": bool(imu_live), "age_s": ia,
                        "hz": rate(self.imu_times) if imu_live else 0,
                        "error": self.imu_error if not imu_live else "",
                        "sample": self.sample if imu_live else None},
                "imus": imus,
            }


def preview_worker(state):
    """Encode at most one latest frame at a time, shared by all viewers."""
    try:
        from PIL import Image
    except ImportError:
        state.error("preview", "Preview tools are missing. Rerun the dashboard installer.")
        return
    sequence = -1
    while not state.stop.is_set():
        with state.condition:
            state.condition.wait_for(lambda: state.stop.is_set() or (
                time.monotonic() - state.preview_requested < 2
                and state.sequence != sequence and state.frame is not None), timeout=1)
            if state.stop.is_set():
                return
            if time.monotonic() - state.preview_requested >= 2 or state.sequence == sequence or state.frame is None:
                continue
            frame, sequence, received_at = state.frame, state.sequence, state.camera_times[-1]
        try:
            with Image.open(io.BytesIO(frame)) as image:
                image.draft("RGB", (640, 360))
                image.thumbnail((640, 360))
                output = io.BytesIO()
                image.convert("RGB").save(output, format="JPEG", quality=70)
                preview = (output.getvalue(), sequence, received_at, image.width, image.height)
            with state.condition:
                state.preview = preview
                state.preview_error = ""
        except Exception:
            state.error("preview", "Could not prepare the camera preview.")
        state.stop.wait(0.05)


def camera_worker(state, device):
    while not state.stop.is_set():
        process = None
        try:
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen([
                    str(Path(__file__).parent / 'capture'), device,
                ], stdout=subprocess.PIPE, stderr=errors, bufsize=0)
                parser = TimedFrames()
                last_frame = time.monotonic()
                while not state.stop.is_set():
                    ready, _, _ = select.select([process.stdout], [], [], 0.5)
                    if ready:
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            errors.seek(0)
                            detail = errors.read(2048).decode("utf-8", "replace").strip()
                            raise RuntimeError(detail or "Camera stopped. Check its USB cable and close other camera programs.")
                        for frame, metadata in parser.feed(chunk):
                            state.frame_received(frame, metadata)
                            last_frame = time.monotonic()
                    if time.monotonic() - last_frame > 5:
                        raise RuntimeError("No new camera frames. Check its USB cable and close other camera programs.")
        except FileNotFoundError:
            state.error("camera", "Timestamped capture tool is missing. Rerun the dashboard installer.")
        except Exception as error:
            state.error("camera", str(error))
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process.stdout.close()
        state.stop.wait(2)


def imu_worker(state, imu=None):
    """Read one LSM6DS or MPU6050 IMU (+/-4 g, +/-500 dps). ``imu`` is an imus.json entry; None = the primary."""
    imu = imu or state.primary_imu
    name, bus_number, address = imu['name'], imu['bus'], imu['address']
    driver = IMU_DRIVERS[imu.get('model', 'lsm6ds')]
    primary = name == state.primary_imu['name']
    report = (lambda message: state.error("imu", message)) if primary else \
        (lambda message: state.extra_imu_error(name, message))
    where = f"I²C bus {bus_number} address {address:#04x}"
    while not state.stop.is_set():
        bus = None
        saved = {}
        try:
            from smbus2 import SMBus           # releases the GIL during a transfer, so the IMU buses read in parallel
            bus = SMBus(bus_number)
            for reg in driver.saved_registers:
                saved[reg] = bus.read_byte_data(address, reg)
            driver.setup(bus, address, f"{name} at {where}")
            state.stop.wait(0.2)
            last_sample = time.monotonic()
            glitches = 0
            while not state.stop.is_set():
                try:
                    if glitches:                      # a noise burst can also reset the chip: re-apply its setup
                        driver.setup(bus, address, f"{name} at {where}")
                    ready = driver.ready(bus, address)
                    if ready:
                        before = time.monotonic_ns()
                        data = driver.read(bus, address)
                        after = time.monotonic_ns()
                    glitches = 0
                except OSError:
                    # Motor switching noise corrupts single transfers (EIO). Retry in place: dropping the
                    # IMU would stop any recording; a dead bus still fails after IMU_GLITCH_RETRIES.
                    glitches += 1
                    if glitches > IMU_GLITCH_RETRIES:
                        raise
                    state.stop.wait(0.002)
                    continue
                if ready:
                    sample = driver.decode(data)
                    sample["read_monotonic_ns"] = (before + after) // 2
                    sample["read_duration_ns"] = after - before
                    state.sample_received(sample, None if primary else name)
                    last_sample = time.monotonic()
                elif time.monotonic() - last_sample > 2:
                    raise RuntimeError(f"No fresh readings from the {name} IMU ({where}). Check its wiring with the Pi powered off.")
                state.stop.wait(0.001)
        except ImportError:
            report("IMU tools are missing. Install python3-smbus2 on the Pi.")
        except FileNotFoundError:
            report(f"I²C bus {bus_number} does not exist. Enable it in /boot/firmware/config.txt and reboot.")
        except PermissionError:
            report("The server needs access to the Pi's I²C device. Start it with the included installer.")
        except Exception as error:
            report(f"{name} IMU ({where}): {error}" if not primary else str(error))
        finally:
            if bus is not None:
                try:
                    for reg, value in saved.items():
                        bus.write_byte_data(address, reg, value)
                except OSError:
                    pass
                bus.close()
        state.stop.wait(2)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        # HTTPServer normally performs a reverse-DNS lookup here, which can
        # stall startup for tens of seconds on a local Wi-Fi network.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def __init__(self, address, state, token=""):
        self.state = state
        self.token = token
        self.assets = Path(__file__).parent / "static"
        self.stream_slots = threading.BoundedSemaphore(8)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(2)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, fmt, *args):
        pass  # Access tokens must not appear in query-string logs.

    def reply(self, status, data, kind="application/json", headers=None):
        if isinstance(data, str):
            data = data.encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if not self.server.token:
            return True
        try:
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            value = cookies["umi_access"].value if "umi_access" in cookies else ""
            return hmac.compare_digest(value, self.server.token)
        except Exception:
            return False

    def do_GET(self):
        try:
            self.get()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_POST(self):
        self.close_connection = True
        try:
            if not self.authorized():
                self.reply(401, '{"error":"Open the private dashboard link first."}')
                return
            path = urlsplit(self.path).path
            # This custom header requires a CORS preflight for other origins.
            if self.headers.get('X-UMI-Action') != 'recording' or self.headers.get('Content-Type') != 'application/json':
                self.reply(403, '{"error":"Use the dashboard recording controls."}')
                return
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 1024:
                self.reply(413, '{"error":"Request too large"}')
                return
            body = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object.')
            recorder = self.server.state.recorder
            if recorder is None:
                self.reply(409, '{"error":"Recording is available on the Pi only."}')
                return
            if path == '/api/recordings/start':
                export_video = body.get('export_video', True)
                if not isinstance(export_video, bool):
                    raise ValueError('export_video must be true or false.')
                label = body.get('label', '')
                if not isinstance(label, str) or len(label) > 80:
                    raise ValueError('Recording names must be 80 characters or fewer.')
                state = self.server.state.snapshot()
                timing = state['camera'].get('timing')
                if not (state['camera']['live'] and state['imu']['live'] and timing and timing['clock'] == 'CLOCK_MONOTONIC'):
                    raise ValueError('Both devices must be live, with monotonic camera timestamps.')
                result = recorder.start({'time_calibration': self.server.state.alignment.profile,
                    'imu_config': {'model': 'LSM6DSL', 'odr_hz': 208, 'gyro_range_dps': 500, 'accel_range_g': 4}},
                    duration=None, mode='task', label=label.strip(), export_video=export_video)
            elif path == '/api/recordings/stop':
                session = body.get('id')
                if not isinstance(session, str):
                    raise ValueError('Refresh the page to identify the recording to stop.')
                recorder.stop(session)
                result = recorder.snapshot()
            else:
                self.reply(404, '{"error":"Not found"}')
                return
            self.reply(200, json.dumps(result))
        except (ValueError, OSError) as error:
            self.reply(409, json.dumps({'error': str(error)}))

    def download(self, session, kind):
        recorder = self.server.state.recorder
        try:
            if recorder is None:
                raise ValueError('No recording available.')
            if kind == 'video':
                path = recorder.video(session)
                size, content_type, filename = path.stat().st_size, 'video/mp4', session+'.mp4'
            elif kind == 'data':
                archive = RecordingDownload(recorder.data_files(session))
                size, content_type, filename = archive.size, 'application/x-tar', session+'.tar'
            else:
                raise ValueError('Unknown download type.')
        except (ValueError, OSError) as error:
            self.reply(409, json.dumps({'error': str(error)}))
            return
        self.connection.settimeout(30)
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        if kind == 'data':
            archive.write(self.wfile)
        else:
            with path.open('rb') as source:
                while chunk := source.read(128*1024):
                    self.wfile.write(chunk)

    def get(self):
        url = urlsplit(self.path)
        token = parse_qs(url.query).get("token", [""])[0]
        if url.path == "/" and token and self.server.token and hmac.compare_digest(token, self.server.token):
            self.reply(303, b"", headers={
                "Location": "/", "Set-Cookie": f"umi_access={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000",
            })
            return
        if not self.authorized():
            self.reply(401, "Open the private dashboard link printed by the installer.", "text/plain; charset=utf-8")
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/record": ("index.html", "text/html; charset=utf-8"),
                  "/recording.js": ("recording.js", "text/javascript; charset=utf-8"),
                  "/recording.css": ("recording.css", "text/css; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/pi.js": ("pi.js", "text/javascript; charset=utf-8"),
                  "/latest-poll.js": ("latest-poll.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8"),
                  "/theme.css": ("theme.css", "text/css; charset=utf-8")}
        if url.path in assets:
            filename, kind = assets[url.path]
            self.reply(200, (self.server.assets / filename).read_bytes(), kind)
        elif url.path == "/api/status":
            self.reply(200, json.dumps(self.server.state.snapshot()))
        elif url.path == '/api/imu/recent':
            state = self.server.state
            name = parse_qs(url.query).get('imu', [''])[0]
            with state.condition:
                status = state.snapshot()
                if name and name != state.primary_imu['name']:
                    if name not in state.extra_imus:
                        self.reply(404, json.dumps({'error': f'no IMU named {name!r}'}))
                        return
                    live = status['imus'][name]['live']
                    samples = list(state.extra_imus[name]['history']) if live else []
                else:
                    live = status['imu']['live']
                    samples = list(state.imu_history) if live else []
            self.reply(200, json.dumps({'server_monotonic_s': status['server_monotonic_s'],
                'live': live, 'samples': samples}))
        elif url.path == '/api/recordings/status':
            recorder = self.server.state.recorder
            result = recorder.snapshot() if recorder else {'status': 'unavailable'}
            result['devices'] = self.server.state.snapshot()
            self.reply(200, json.dumps(result))
        elif url.path == '/api/recordings':
            recorder = self.server.state.recorder
            keys = ('id', 'status', 'label', 'recording_mode', 'elapsed_s', 'frames', 'imu_samples',
                    'error', 'warning', 'raw_data_saved', 'video_status', 'video_export_requested', 'bytes_written')
            entries = [{key: item.get(key) for key in keys} for item in recorder.list_saved()] if recorder else []
            self.reply(200, json.dumps(entries))
        elif url.path == '/api/recordings/download':
            query = parse_qs(url.query)
            self.download(query.get('id', [''])[0], query.get('kind', [''])[0])
        elif url.path == "/preview.jpg":
            state = self.server.state
            after = parse_qs(url.query).get("after", [""])[0]
            with state.condition:
                state.preview_requested = time.monotonic()
                state.condition.notify_all()
                preview, error = state.preview, state.preview_error
                live = state.snapshot()["camera"]["live"]
            if not live or error:
                self.reply(503, json.dumps({"error": error or "No fresh camera image available"}))
            elif preview is None or str(preview[1]) == after or time.monotonic() - preview[2] > 0.25:
                self.reply(204, b"")
            else:
                frame, sequence, received_at, width, height = preview
                self.reply(200, frame, "image/jpeg", {
                    "X-Frame-Sequence": str(sequence),
                    "X-Frame-Age-Ms": str(round((time.monotonic() - received_at) * 1000, 2)),
                    "X-Preview-Size": f"{width}×{height}",
                })
        elif url.path == "/snapshot.jpg":
            with self.server.state.condition:
                frame = self.server.state.frame
                live = self.server.state.snapshot()["camera"]["live"]
            if frame is None or not live:
                self.reply(503, '{"error":"No fresh camera image available"}')
            else:
                self.reply(200, frame, "image/jpeg", {"Content-Disposition": 'attachment; filename="bucket-camera.jpg"'})
        elif url.path in ("/events", "/stream.mjpg"):
            if not self.server.stream_slots.acquire(blocking=False):
                self.reply(503, '{"error":"Too many viewers; close another dashboard tab."}')
                return
            try:
                self.stream(url.path == "/events")
            finally:
                self.server.stream_slots.release()
        else:
            self.reply(404, '{"error":"Not found"}')

    def stream(self, events):
        self.connection.settimeout(5)
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if events else "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        state = self.server.state
        last_sequence = -1
        while not state.stop.is_set():
            if events:
                payload = "data: " + json.dumps(state.snapshot(), separators=(",", ":")) + "\n\n"
                self.wfile.write(payload.encode())
            else:
                with state.condition:
                    state.condition.wait_for(lambda: state.sequence != last_sequence or state.stop.is_set(), timeout=1)
                    frame, sequence = state.frame, state.sequence
                    live = state.snapshot()["camera"]["live"]
                if not live:
                    return  # Release the viewer slot when a camera disconnects.
                if frame is None or sequence == last_sequence:
                    state.stop.wait(0.1)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                last_sequence = sequence
            self.wfile.flush()
            state.stop.wait(0.05)

