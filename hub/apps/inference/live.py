"""What the inference page shows live: the sensor poller (Pi status + bucket pitch, GPIO bridge health), the
job's joint estimate and the observation it last sent, the twin views (estimate, live twin state) and the
Pi camera with the CLOAK robot mask.

:class:`LiveViews` is a mixin of :class:`hub.apps.inference.core.Inference`: it uses the Inference state (``pi``, ``bridge``,
``bus``, ``hardware_map``, ``live_twin``, ``twin``, ``sensors``, ``q_est`` / ``q_source`` / ``q_est_t``,
``obs`` and the caches ``_twin_state_jpg`` / ``_masker`` / ``_cam_mask``, under ``_lock``),
and the poller also runs ``_deadman``.
"""
from __future__ import annotations

import math
import time

import numpy as np

from hub.apps._common import decode_jpeg, encode_jpeg
from hub.apps.inference.pi import BRIDGE_UNAVAILABLE, describe_bridge
from hub.apps.inference.twin import tint_mask
from hub.apps.inference.util import Refused
from digital_twin.policy import runner as pr


class LiveViews:
    """The live half of :class:`hub.apps.inference.core.Inference` (see the module docstring)."""

    # ---- the job's estimate and observation
    def set_estimate(self, q: dict, source: str, publish: bool = True):
        with self._lock:
            self.q_est, self.q_source = {k: float(v) for k, v in q.items()}, source
            self.q_est_t = time.monotonic()
        if publish:
            self.bus.publish('estimate', {'q': self.q_est, 'source': source})

    def set_observation(self, image: np.ndarray, job_id: str, state=None, cloak=None, mask=None):
        img, note = np.asarray(image), None
        if cloak is not None:                               # CLOAK policies: show what the model is told to ignore
            if cloak.get('needs_mask') and mask is not None and np.shape(mask) == img.shape[:2]:
                m = np.asarray(mask, bool)
                img = tint_mask(img, m)
                note = f'CLOAK on ({cloak.get("mode")}): robot mask {100 * m.mean():.1f}% of frame hidden'
            else:
                note = f'CLOAK off ({cloak.get("mode")}): robot visible to the model'
        try:
            jpg = encode_jpeg(img, 95)
        except Exception:
            return
        with self._lock:
            self.obs = (jpg, job_id, time.time(), None if state is None else np.asarray(state, float).tolist(), note)

    # ---- sensor poller
    def poll_pi(self) -> dict:
        try:
            st = self.pi.status()
        except Exception as ex:
            reason = getattr(ex, 'reason', None) or ex
            return {'ok': False, 'error': f'{type(ex).__name__}: {reason}', 'host': self.pi.base, 'link': self.pi.link()}
        self.live_twin.update(st.get('imus') or {'bucket': st.get('imu') or {}}, time.monotonic())
        cam, imu = st.get('camera') or {}, st.get('imu') or {}
        s = imu.get('sample') or {}
        out = {'ok': True, 'host': self.pi.base, 'link': self.pi.link(), 'rtt_ms': self.pi.status_ms,
               'uptime_s': st.get('uptime_s'),
               'camera': {'live': bool(cam.get('live')), 'fps': cam.get('fps'), 'age_s': cam.get('age_s'),
                          'frames': cam.get('frames'), 'error': cam.get('error') or '',
                          'proxy_fps': round(self.pi.proxy_fps(), 1)},
               'imu': {'live': bool(imu.get('live')), 'hz': imu.get('hz'), 'age_s': imu.get('age_s'),
                       'error': imu.get('error') or '', 'accel_g': s.get('accel'), 'gyro_dps': s.get('gyro'),
                       'clipped': s.get('clipped')}}
        if imu.get('live') and s.get('accel') is not None:
            p = self.live_twin.link_pitch('bucket')             # the arm IMU calibration (model convention)
            out['imu'].update(pitch_deg=None if p is None else math.degrees(p),
                              pitch_source='arm IMU calibration' if p is not None else 'no arm IMU calibration')
        return out

    def poll_bridge(self) -> dict:
        """The hub's GPIO bridge health (GpioBridge.status(): ESP32 link, target)."""
        b = self.bridge
        if b is None:
            return {'ok': False, 'host': 'hub gpio', 'error': BRIDGE_UNAVAILABLE}
        try:
            h = b.status()
        except Exception as ex:
            return {'ok': False, 'host': describe_bridge(b), 'error': f'{type(ex).__name__}: {ex}'}
        return {'ok': True, 'host': describe_bridge(b), 'esp32': bool(h.get('esp32')), 'target': h.get('target')}

    def snapshot_sensors(self) -> dict:
        with self._lock:
            est = dict(self.q_est) if self.q_est else None
        snap = dict(self.sensors)
        snap['twin_state'] = self.twin_state()
        if est:
            snap['estimate'] = {'q': est, 'pitch_deg': math.degrees(est.get('boom', 0) + est.get('bucket', 0)),
                                'source': self.q_source}
        return snap

    def refresh_sensors(self, pi=True, bridge=True) -> dict:
        if pi:
            self.sensors['pi'] = self.poll_pi()
        if bridge:
            self.sensors['bridge'] = self.poll_bridge()
        return self.snapshot_sensors()

    def _poll_loop(self):
        next_bridge = 0.0
        while not self._closing.is_set():
            t = time.monotonic()
            try:
                do_bridge = t >= next_bridge
                self.refresh_sensors(pi=True, bridge=do_bridge)
                if do_bridge:
                    next_bridge = t + self.bridge_period_s
                self.bus.publish('sensors', self.snapshot_sensors())
                self._deadman()
            except Exception as ex:                          # the poller must never die silently
                self.bus.publish('log', {'job': None, 'line': f'sensor poller error: {type(ex).__name__}: {ex}'})
            period = self.pi_period_s if self.sensors['pi'].get('ok') else 1.0
            self._closing.wait(max(0.02, period - (time.monotonic() - t)))

    # ---- views
    def twin_frame(self, camera='overview') -> bytes:
        with self._lock:
            q, src = self.q_est, self.q_source
        lines = []
        if q is None:
            try:
                q = pr.home_values(pr.load_hw(self.hardware_map))
            except Exception:
                q = {}
            lines.append('no estimate yet ! (shown at the homed pose)')
        else:
            lines.append(f"estimate boom {math.degrees(q.get('boom', 0)):+.1f} deg  bucket "
                         f"{math.degrees(q.get('bucket', 0)):+.1f} deg")
            lines.append(f'estimate: {src}')
        return self.twin.render(q, lines, camera)

    def twin_state(self) -> dict:
        with self._lock:
            q, t, src = (dict(self.q_est) if self.q_est else None), self.q_est_t, self.q_source
        return self.live_twin.snapshot(q, t, src)

    def twin_state_frame(self, camera='overview') -> tuple[bytes, dict]:
        st = self.twin_state()
        q = {n: j['rad'] for n, j in st['joints'].items()}
        key = tuple(round(v, 4) for v in q.values())
        last = self._twin_state_jpg
        if last is not None and last[0] == key and last[1] == camera:
            return last[2], st
        jpg = self.twin.render(q, (), camera)
        self._twin_state_jpg = (key, camera, jpg)
        return jpg, st

    def observation(self) -> tuple[bytes, str]:
        with self._lock:
            obs = self.obs
        if obs is not None:
            state = '' if obs[3] is None else ' state=' + ','.join(f'{v:.4f}' for v in obs[3])
            cloak = f' {obs[4]};' if len(obs) > 4 and obs[4] else ''
            return obs[0], f'sent; job {obs[1]};{cloak}{state}'
        status, body, _ctype = self.pi.frame()
        if status != 200:
            raise Refused(502, 'no observation sent yet and the live camera is unavailable')
        from policy.training.cloak.masking import letterbox
        img = decode_jpeg(body)
        img, _ = letterbox(img, np.zeros(img.shape[:2], bool))        # what the policy server sees
        return encode_jpeg(img, 90), 'preview (live camera, policy letterbox)'

    def camera_frame(self, mask: bool = False) -> tuple[int, bytes, str, str | None]:
        """The Pi camera frame. ``mask`` (a CLOAK checkpoint is selected): the twin-rendered robot mask at the
        live joint estimate tinted on it, as a CLOAK run sends it; the mask is re-rendered only when the pose
        or the mount changes. Returns (status, body, content type, CLOAK note or None)."""
        status, body, ctype = self.pi.frame()
        if status != 200 or not mask:
            return status, body, ctype, None
        try:
            img = decode_jpeg(body)
            q = {n: j['rad'] for n, j in self.twin_state()['joints'].items()}
            masker = self._preview_masker()
            key = (tuple(round(v, 3) for v in q.values()), img.shape[:2], id(masker))
            if self._cam_mask is None or self._cam_mask[0] != key:
                self._cam_mask = (key, masker.mask(q, img.shape))
            m = self._cam_mask[1]
            return 200, encode_jpeg(tint_mask(img, m), 90), 'image/jpeg', f'{100 * m.mean():.1f}% masked'
        except Exception as ex:                             # no twin / mask config: the plain frame
            return status, body, ctype, f'mask unavailable: {ex}'

    def _preview_masker(self):
        """The self-mask renderer for previews, rebuilt when configs/hardware/self_mask.json changes (mount updates)."""
        from digital_twin.sim.self_mask import DEFAULT_CONFIG, SelfMasker
        mt = DEFAULT_CONFIG.stat().st_mtime
        cur = self._masker
        if cur is None or cur[0] != mt:
            if cur is not None:
                try:
                    cur[1].close()
                except Exception:
                    pass
            cur = self._masker = (mt, SelfMasker())
        return cur[1]
