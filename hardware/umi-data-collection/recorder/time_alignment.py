"""Apply a validated software offset without changing raw capture timestamps."""
import hashlib
import json
import math
from pathlib import Path
import re
import socket

SIGN = 'matching_imu_host_time = raw_camera_monotonic_time + camera_to_imu_offset_s'
FIELDS = ('camera_on_imu_host_clock_ns', 'camera_to_imu_offset_ns', 'time_calibration_id')


class TimeAlignment:
    def __init__(self, path=None, hostname=None):
        self.profile = None
        self.error = ''
        if path is None or not Path(path).exists():
            return
        try:
            raw = Path(path).read_bytes()
            if len(raw) > 16384:
                raise ValueError('Calibration file is too large.')
            data = json.loads(raw)
            if data.get('status') != 'validated_software_offset' or data.get('reasons') != []:
                raise ValueError('The calibration has not passed validation.')
            offset = data['camera_to_imu_offset_s']
            if type(offset) not in (int, float) or not math.isfinite(offset) or abs(offset) > .3:
                raise ValueError('Invalid time correction.')
            deployment = data['deployment']
            if deployment['hostname'] != (hostname or socket.gethostname()):
                raise ValueError('This calibration belongs to a different Pi.')
            if data['sign_convention'] != SIGN:
                raise ValueError('Unknown time correction convention.')
            if not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}', data['recording_id']):
                raise ValueError('Invalid calibration recording ID.')
            if not re.fullmatch(r'[a-f0-9]{64}', data['recording_sha256']):
                raise ValueError('Missing calibration recording provenance.')
            size = data['camera_intrinsics']['image_size']
            if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
                raise ValueError('Invalid calibration image size.')
            source = deployment['timestamp_source']
            if source not in ('start_of_exposure', 'end_of_frame'):
                raise ValueError('Invalid camera timestamp source.')
            self.profile = {'recording_id': data['recording_id'], 'recording_sha256': data['recording_sha256'],
                'calibration_file_sha256': hashlib.sha256(raw).hexdigest(), 'offset_ns': round(offset * 1e9),
                'offset_s': offset, 'image_size': size, 'timestamp_source': source,
                'hostname': deployment['hostname'], 'sign_convention': SIGN,
                'method': 'software_camera_to_gyro_host_time', 'hardware_synchronized': False}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            self.error = f'Time correction not loaded: {error}'

    def apply(self, metadata):
        # Always derive from the raw timestamp, even if called twice.
        result = {k: v for k, v in metadata.items() if k not in FIELDS}
        profile = self.profile
        if (profile and result.get('clock') == 'CLOCK_MONOTONIC' and not result.get('buffer_error')
                and [result.get('width'), result.get('height')] == profile['image_size']
                and result.get('timestamp_source') == profile['timestamp_source']):
            result.update(camera_on_imu_host_clock_ns=result['camera_timestamp_ns'] + profile['offset_ns'],
                          camera_to_imu_offset_ns=profile['offset_ns'], time_calibration_id=profile['recording_id'])
        return result

    def snapshot(self, metadata=None):
        applied = bool(self.profile and metadata and metadata.get('time_calibration_id') == self.profile['recording_id'])
        return {'configured': self.profile is not None, 'applied': applied,
                'offset_ms': self.profile['offset_ns'] / 1e6 if self.profile else None,
                'recording_id': self.profile['recording_id'] if self.profile else None,
                'error': self.error or ('Waiting for a compatible camera frame.' if self.profile and not applied else '')}
