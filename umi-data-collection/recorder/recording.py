"""Timestamped recordings. Capture threads never wait for disk or encoding."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import shutil
import tarfile
import threading
import time
import uuid

SESSION = re.compile(r'[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}')
DATA_FILES = ('metadata.json', 'frames.csv', 'imu.csv', 'video.mjpg')
# metadata.json (archived with the data) holds only what processing reads. The recorder's own state for the
# dashboard goes to status.json, which stays on the Pi.
TIME_CALIBRATION_KEYS = ('recording_id', 'recording_sha256', 'offset_ns', 'image_size', 'timestamp_source',
                         'sign_convention')
STATUS_KEYS = ('status', 'warning', 'elapsed_s', 'recording_mode', 'bytes_written', 'raw_data_saved',
               'video_status', 'video_export_requested')


class Recorder:
    def __init__(self, root, health=None):
        self.root, self.health = Path(root), health
        self.lock = threading.RLock()
        self.closing = threading.Event()
        self.session = None
        self.status = 'idle'
        self.error = self.warning = self.label = ''
        self.frames = self.samples = self.dropped = self.clipped = 0
        self.bytes_written = self.last_frame = 0
        self.started = self.ended = 0
        self.duration, self.mode = 45, 'calibration'
        self.export_video = False
        self.free_bytes = None
        self.raw_data_saved = False
        self.video_status, self.video_progress = 'none', 0
        self.queue = self.thread = None
        for entry in self.list_saved():
            if entry['status'] in ('recording', 'saving'):
                directory = self.root/entry['id']
                metadata, status = self._read(directory)
                metadata['error'] = entry.get('error') or 'Dashboard restarted before this recording finished. Review the saved data.'
                status.update(status='ready' if entry.get('raw_data_saved') else 'error',
                              video_status='ready' if (directory/'video.mp4').exists() else 'error')
                self._write_json(directory/'metadata.json', metadata)
                self._write_json(directory/'status.json', status)

    def snapshot(self):
        with self.lock:
            elapsed = (self.ended or time.monotonic())-self.started if self.started else 0
            if self.duration is not None:
                elapsed = min(self.duration, elapsed)
            return {'status': self.status, 'id': self.session, 'error': self.error, 'warning': self.warning,
                    'elapsed_s': max(0, elapsed), 'duration_s': self.duration, 'recording_mode': self.mode,
                    'label': self.label, 'frames': self.frames, 'imu_samples': self.samples,
                    'dropped': self.dropped, 'clipped_samples': self.clipped, 'bytes_written': self.bytes_written,
                    'free_bytes': self.free_bytes, 'raw_data_saved': self.raw_data_saved,
                    'video_status': self.video_status, 'video_progress': self.video_progress,
                    'video_export_requested': self.export_video}

    def start(self, metadata, duration=45, mode='calibration', label='', export_video=True):
        if not isinstance(export_video, bool):
            raise ValueError('export_video must be true or false.')
        with self.lock:
            if self.status in ('recording', 'saving'):
                raise ValueError('A recording is already running or being saved.')
            if self.closing.is_set():
                raise ValueError('The dashboard is shutting down.')
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.free_bytes = shutil.disk_usage(self.root).free
            if self.free_bytes < 1024**3:
                raise ValueError('At least 1 GB of free space is needed on the Pi.')
            if mode == 'task' and export_video:
                from .video_export import check_video_support
                check_video_support()
            self.session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
            self.started, self.ended = time.monotonic(), 0
            self.duration, self.mode, self.label = duration, mode, label
            self.export_video = bool(mode == 'task' and export_video)
            self.frames = self.samples = self.dropped = self.clipped = self.bytes_written = 0
            self.last_frame = 0
            self.error = self.warning = ''
            self.raw_data_saved = False
            self.video_status, self.video_progress = 'none', 0
            self.queue = queue.Queue(maxsize=2048)
            self.status = 'recording'
            profile = metadata.get('time_calibration')
            metadata = {'time_calibration': {k: profile[k] for k in TIME_CALIBRATION_KEYS if k in profile} if profile else None,
                        'imu_config': metadata.get('imu_config'),
                        'started_monotonic_s': self.started, 'duration_s': duration, 'format_version': 2}
            self.thread = threading.Thread(target=self._write, args=(metadata,), daemon=True)
            self.thread.start()
            return self.snapshot()

    def fail(self, message):
        with self.lock:
            if self.status == 'recording':
                self.error = message
                self.stop()

    def put(self, kind, payload):
        with self.lock:
            if self.status != 'recording':
                return
            if self.duration is not None and time.monotonic()-self.started >= self.duration:
                self.stop()
                return
            if kind == 'frame':
                meta = payload[1]
                timestamp = meta['camera_timestamp_ns']
                if self.mode == 'task' and (timestamp <= self.last_frame or meta.get('buffer_error')
                        or meta.get('clock') != 'CLOCK_MONOTONIC'):
                    self.fail('Camera timing changed or a damaged frame arrived. Recording stopped; captured data will be saved.')
                    return
                if 0 <= timestamp-self.last_frame < 45_000_000:
                    return
                self.last_frame = timestamp
            try:
                self.queue.put_nowait((kind, payload))
            except queue.Full:
                self.dropped += 1
                self.fail('The SD card could not keep up. Recording stopped; captured data will be saved with a dropped-data warning.')

    def stop(self, session=None):
        with self.lock:
            if session is not None and session != self.session:
                raise ValueError('This is a different recording. Refresh the page before stopping it.')
            if self.status == 'recording':
                self.ended, self.status = time.monotonic(), 'saving'

    def shutdown(self):
        self.closing.set()
        self.fail('The dashboard shut down during recording. Captured data was retained.')
        self.stop()

    def _directory(self, session):
        if not isinstance(session, str) or not SESSION.fullmatch(session):
            raise ValueError('Invalid recording ID.')
        directory = self.root/session
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('Recording not found.')
        return directory

    def entry(self, session):
        directory = self._directory(session)
        try:
            metadata, status = self._read(directory)
        except (OSError, ValueError):
            raise ValueError('Recording details are not available yet.')
        if not isinstance(metadata, dict) or not isinstance(status, dict):
            raise ValueError('Recording metadata is invalid.')
        metadata = dict(metadata, **status)
        metadata.setdefault('status', 'error')
        if (self.root/(session+'.tar')).is_file():
            metadata.update(status='ready', raw_data_saved=True)
        metadata['id'] = session
        return metadata

    def list_saved(self):
        entries = []
        if self.root.exists():
            for directory in sorted(self.root.iterdir(), reverse=True):
                if not SESSION.fullmatch(directory.name) or not directory.is_dir():
                    continue
                try:
                    entries.append(self.entry(directory.name))
                except ValueError:
                    continue
                if len(entries) >= 30:
                    break
        return entries

    def archive(self, session):
        self._directory(session)
        path = self.root/(session+'.tar')
        if path.is_symlink() or not path.is_file():
            raise ValueError('This recording is not ready to download.')
        return path

    def video(self, session):
        path = self._directory(session)/'video.mp4'
        if path.is_symlink() or not path.is_file():
            raise ValueError('The MP4 is not ready. Raw data remains saved on the Pi.')
        return path

    def data_files(self, session):
        metadata = self.entry(session)
        if metadata['status'] == 'recording' or (metadata['status'] == 'saving' and not metadata.get('raw_data_saved')):
            raise ValueError('Wait for the recording data to finish saving.')
        directory = self._directory(session)
        files = [directory/name for name in DATA_FILES]
        if not all(path.is_file() and not path.is_symlink() for path in files):
            raise ValueError('Some recording files are missing. The remaining files are retained on the Pi.')
        return files

    @staticmethod
    def _read(directory):
        """(metadata.json, status.json) of a recording directory. Older recordings have no status.json:
        their metadata.json still carries the dashboard state."""
        metadata = json.loads((directory/'metadata.json').read_text())
        try:
            status = json.loads((directory/'status.json').read_text())
        except FileNotFoundError:
            status = {}
        return metadata, status

    @staticmethod
    def _write_json(path, data):
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(path)

    def _save(self, directory, metadata):
        """metadata.json: what processing reads. status.json: the dashboard's state, kept on the Pi."""
        with self.lock:
            snapshot = self.snapshot()
            recorded = dict(metadata, id=self.session, label=self.label, frames=self.frames,
                            imu_samples=self.samples, dropped=self.dropped, clipped_samples=self.clipped,
                            error=self.error)
            status = {key: snapshot[key] for key in STATUS_KEYS}
        self._write_json(directory/'metadata.json', recorded)
        self._write_json(directory/'status.json', status)

    def _monitor(self):
        if self.health:
            state = self.health()
            for device, label in (('camera', 'Camera'), ('imu', 'IMU')):
                if not state[device]['live']:
                    self.fail(f'{label} disconnected or stopped sending data. Recording stopped; captured data will be saved.')
        self.free_bytes = shutil.disk_usage(self.root).free
        if self.free_bytes < max(512*1024**2, self.bytes_written):
            self.fail('Storage is nearly full. Recording stopped to leave room to save the video.')

    def _write(self, metadata):
        directory = self.root/self.session
        try:
            directory.mkdir(mode=0o700)
            self._save(directory, metadata)
            with (directory/'video.mjpg').open('wb') as video, (directory/'frames.csv').open('w', newline='') as cf, (directory/'imu.csv').open('w', newline='') as im:
                frames, imu = csv.writer(cf), csv.writer(im)
                frames.writerow(['camera_timestamp_ns', 'dequeue_monotonic_ns', 'receipt_monotonic_ns', 'sequence', 'flags', 'offset', 'size', 'width', 'height',
                                 'camera_on_imu_host_clock_ns', 'camera_to_imu_offset_ns', 'time_calibration_id'])
                imu.writerow(['read_monotonic_ns', 'read_duration_ns', 'gx_dps', 'gy_dps', 'gz_dps', 'ax_g', 'ay_g', 'az_g', 'clipped'])
                checked = flushed = 0
                while True:
                    now = time.monotonic()
                    if now-checked > .25:
                        if self.status == 'recording':
                            self._monitor()
                        checked = now
                    if now-flushed > 1:
                        for stream in (video, cf, im):
                            stream.flush()
                        self._save(directory, metadata)
                        flushed = now
                    with self.lock:
                        if self.duration is not None and now-self.started >= self.duration:
                            self.stop()
                        finished = self.status != 'recording' and self.queue.empty()
                    if finished:
                        break
                    try:
                        kind, payload = self.queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if kind == 'frame':
                        data, meta = payload
                        position = video.tell()
                        video.write(data)
                        frames.writerow([meta[k] for k in ('camera_timestamp_ns', 'dequeue_monotonic_ns', 'receipt_monotonic_ns', 'sequence', 'flags')] + [position, len(data), meta['width'], meta['height']] +
                                        [meta.get(k, '') for k in ('camera_on_imu_host_clock_ns', 'camera_to_imu_offset_ns', 'time_calibration_id')])
                        with self.lock:
                            self.frames += 1
                            self.bytes_written += len(data)
                    else:
                        imu.writerow([payload['read_monotonic_ns'], payload['read_duration_ns'], *payload['gyro'], *payload['accel'], int(payload['clipped'])])
                        with self.lock:
                            self.samples += 1
                            self.clipped += int(payload['clipped'])
                            if self.clipped:
                                self.warning = 'Some IMU readings exceeded the sensor range. These sections need review.'
                for stream in (video, cf, im):
                    stream.flush()
                    os.fsync(stream.fileno())
            self.raw_data_saved = True
            self._save(directory, metadata)
            if self.mode == 'calibration':
                partial = self.root/(self.session+'.tar.partial')
                with tarfile.open(partial, 'w') as archive:
                    for name in DATA_FILES:
                        archive.add(directory/name, arcname=name)
                partial.replace(self.root/(self.session+'.tar'))
            elif self.export_video:
                self.video_status = 'saving'
                self._save(directory, metadata)
                try:
                    from .video_export import export_video
                    export_video(directory, self._video_progress, self.closing)
                    self.video_status = 'ready'
                except Exception as error:
                    self.video_status = 'error'
                    self.warning = f'MP4 could not be finished: {error}. Original video and motion data are saved; download the raw data.'
            self.free_bytes = shutil.disk_usage(self.root).free
            with self.lock:
                self.status = 'ready'
                self._save(directory, metadata)
        except Exception as error:
            with self.lock:
                self.ended = self.ended or time.monotonic()
                self.error = f'Recording could not finish saving: {error}. Available files remain on the Pi.'
                self.status = 'error'
                try:
                    self._save(directory, metadata)
                except OSError:
                    pass

    def _video_progress(self, progress):
        with self.lock:
            self.video_progress = progress
