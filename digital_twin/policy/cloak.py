"""Bounded OpenPI transport for the 4-joint CLOAK π0.5 policy servers (policy/training/cloak/server.py).

Uses the official wire codec. No GPIO connection or hardware command. Callers pass the NATIVE
uncropped camera frame (1280×720 on the bucket camera) and a boolean robot-pixel mask with the same
H×W. The server letterboxes RGB and mask together to 224×224 (masking.letterbox: aspect-preserving
resize, zero padding, nearest-neighbour mask); ``infer`` applies that same letterbox before sending,
which the server's pass then leaves unchanged (letterbox of a 224×224 frame is the identity), so the
policy input is identical while a request is ~0.2 MB instead of ~2.8 MB (the uplink to the GPU box
is the slow part). Callers must not pad, resize or blank anything themselves.
Baseline servers zero the mask themselves; cloak/cloak_oov/noaug servers use it as given.
"""
from __future__ import annotations

import time
from contextlib import ExitStack
from urllib.parse import urlsplit

import numpy as np

from policy.configs.schema import POLICY_HZ

JOINTS = ('slew', 'boom', 'stick', 'bucket')
SCHEMA = 'actathon.cloak.policy.v1'
MAX_MESSAGE = 16 * 1024 * 1024  # websockets max_size on both ends
MASKED_MODES = ('cloak', 'cloak_oov', 'noaug')  # modes whose server forwards the pixel mask to the encoder
# Joint space of the labels (policy.configs.schema.LABEL_PROFILES): 'calibrated' = labels from the calibrated full4
# twin, whose ranges are the real machine's, so the runner passes joints through unchanged. Nothing else is served.
POLICY_SPACE = 'calibrated'


class CloakPolicy:
    def __init__(self, gateway_url, prompt, *, connect_timeout=5.0, infer_timeout=30.0):
        u = urlsplit(gateway_url)
        if u.scheme not in ('ws', 'wss') or not u.hostname or u.username or u.password:
            raise ValueError('CLOAK policy needs a ws:// or wss:// endpoint without embedded credentials')
        if not prompt or not prompt.strip():
            raise ValueError('A task prompt is required')
        self.url, self.prompt = gateway_url, prompt
        self.connect_timeout, self.infer_timeout = connect_timeout, infer_timeout
        self._ws = None
        self._stack = None
        self._metadata = None

    def metadata(self):
        if self._metadata is not None:
            return self._metadata
        from websockets.sync.client import connect
        from .vendor.openpi import msgpack_numpy as codec
        self._stack = ExitStack()
        try:
            self._ws = self._stack.enter_context(connect(
                self.url, compression=None, max_size=MAX_MESSAGE,
                open_timeout=self.connect_timeout, close_timeout=1))
            raw = self._ws.recv(timeout=self.connect_timeout)
            if not isinstance(raw, bytes):
                raise ValueError('CLOAK metadata was not a MessagePack response')
            md = codec.unpackb(raw)
            md = md if isinstance(md, dict) else {}
            c = md.get('action_contract') or {}
            if (md.get('schema') != SCHEMA or c.get('joints') != list(JOINTS) or c.get('units') != 'radians'
                    or c.get('type') != 'absolute next joint target' or c.get('sampling_hz') != POLICY_HZ
                    or md.get('action_horizon') != 10
                    or md.get('mode') not in ('baseline', *MASKED_MODES)
                    or c.get('policy_space') != POLICY_SPACE):
                raise ValueError('Endpoint is not the expected 4-joint CLOAK policy')
            mode = 'pretrained' if md.get('weights') == 'pretrained_initialization' else md['mode']
            self._metadata = {'camera_names': ['rgb'], 'action_keys': ['actions'], 'action_horizon': 10,
                              'action_dim': 4, 'joints': list(JOINTS), 'policy_backend': 'cloak',
                              'policy_space': POLICY_SPACE, 'mode': mode,
                              'needs_mask': md['mode'] in MASKED_MODES,  # pretrained follows its server's masking mode
                              'raw_text_required': False, 'server': md}
        except ConnectionRefusedError as ex:
            self.close()
            u = urlsplit(self.url)
            host = f'[{u.hostname}]' if ':' in u.hostname else u.hostname
            endpoint = f'{u.scheme}://{host}:{u.port or (443 if u.scheme == "wss" else 80)}'
            hint = ('Start the SSH port forward on the SAME computer running the policy dashboard, '
                    'keep it open, and check that its local port matches this gateway.'
                    if u.hostname in ('localhost', '127.0.0.1', '::1') else
                    'Check the gateway host/port and whether its policy server is listening.')
            raise ConnectionRefusedError(ex.errno, f'CLOAK policy connection refused at {endpoint}. {hint}') from ex
        except Exception:
            self.close()
            raise
        return self._metadata

    def infer(self, rgb, state, mask=None):
        md = self.metadata()
        from .vendor.openpi import msgpack_numpy as codec
        rgb, state = np.asarray(rgb), np.asarray(state, np.float32)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('CLOAK inference requires the native H×W×3 uint8 RGB camera frame')
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError('State must be 4 finite radians in [slew, boom, stick, bucket] order')
        if mask is None:
            if md['needs_mask']:
                raise ValueError(f'The {md["server"]["mode"]} policy requires the robot-pixel mask for this frame')
            mask = np.zeros(rgb.shape[:2], bool)
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
            raise ValueError('Mask must be boolean with the same H×W as the RGB frame (True = robot pixels)')
        from policy.training.cloak.masking import letterbox
        rgb, mask = letterbox(rgb, mask)            # the server's own preprocessing, done before the upload
        payload = codec.packb({'rgb': np.ascontiguousarray(rgb), 'embodiment_mask': np.ascontiguousarray(mask),
                               'state': state, 'prompt': self.prompt})
        if len(payload) > MAX_MESSAGE:
            raise ValueError(f'{rgb.shape[1]}×{rgb.shape[0]} frame makes a {len(payload)} B request, over the '
                             f'server\'s {MAX_MESSAGE} B limit; send the native camera frame, not an upscaled one')
        start = time.perf_counter()
        try:
            self._ws.send(payload)
            raw = self._ws.recv(timeout=self.infer_timeout)
            if not isinstance(raw, bytes):
                raise ValueError(f'CLOAK server error instead of actions: {str(raw)[:300]}')
            result = codec.unpackb(raw)
            actions = np.asarray(result.get('actions'), np.float32)
            if actions.shape != (10, 4) or not np.isfinite(actions).all():
                raise ValueError(f'Expected finite (10,4) joint targets, got {actions.shape}')
            timing = result.get('server_timing') or result.get('policy_timing') or {}
            return actions, time.perf_counter() - start, timing.get('infer_ms')
        except Exception:
            self.close()
            raise

    def reset(self):
        pass  # The CLOAK server is stateless between observations.

    def close(self):
        stack, self._stack = self._stack, None
        self._ws = self._metadata = None
        if stack is not None:
            stack.close()
