"""Mask the excavator out of its own camera at inference time.

The Pi camera rides on the bucket, so the machine itself (bucket lip, stick, boom, cab) is in
view and moves through the image as the joints move. :class:`SelfMasker` renders the MuJoCo twin
from that camera at the runner's joint estimate, keeps the pixels whose segmentation id is one of
the machine's visual meshes, warps the result through the camera's lens model and blanks those
pixels before the training preprocessing.

    from digital_twin.sim.self_mask import SelfMasker
    masker = SelfMasker()                                  # configs/hardware/self_mask.json
    masked, mask = masker.apply(rgb, {'boom': 0.3, 'bucket': -0.2})

Geometry comes from the twin's MJCF (``models/<profile>.xml``, built from the scanned link meshes;
there is no separate URDF). Joints are real-space radians, i.e. the ``full4_calibrated`` joint space
the runner's estimator reports; joints the model lacks are ignored and joints the estimate omits
(slew, and the stick unless tracked) sit at their rest values (kinematics.locked_joints()).

Rendering: a pinhole camera is added to the mount body (``mount.body``, default ``bucket``) at
``mount.pos_m`` / ``mount.rpy_deg``; with ``mount.frame`` ``world_home`` these are world axes with the
machine homed (see SelfMasker._mount_in_body), otherwise body axes. The segmentation image is rendered with a centred principal
point wide enough to cover the distorted field of view, then remapped per output pixel through
the calibrated intrinsics and distortion (``radtan`` = OpenCV, ``equidistant`` = OpenCV fisheye): the camera's
lens in configs/hardware/camera-calibration/camera.json (policy.data.camera), unless a config carries its own ``intrinsics``.
The remap table is computed once per frame size. The mask is computed at ``work_width`` and
dilated by ``dilate_px`` (output pixels) to absorb estimate and mount error.

All MuJoCo / GL work runs on one dedicated thread (GL contexts are thread-bound), so the masker
can be called from any thread. The mount is not measured yet (``"measured": false``); tune it in
configs/hardware/self_mask.json (``mount``).
"""
from __future__ import annotations

import copy
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from policy.data import camera

from ..paths import CONFIG, SIM, hardware_map

DEFAULT_CONFIG = CONFIG / 'self_mask.json'
CAMERA_LENS = 'configs/hardware/camera-calibration/camera.json'   # ``intrinsics.source`` of a lens taken from there (never saved into a config)
CAMERA_NAME = 'self_mask_camera'
MAX_HALF_FOV_DEG = 80.0           # pinhole render limit; distorted rays beyond this are left unmasked
MAX_RENDER_PX = 2048
# Default orientation in the mount body (columns = OpenCV camera x, y, z): optical axis along body
# -x (toward the cab, along the teeth), image-down along body -z, image-right along body +y.
R_BODY_CV_DEFAULT = np.array([[0.0, 0.0, -1.0],
                              [1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0]])
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])     # MuJoCo cameras look along -z with +y up


def camera_lens(path=None) -> dict:
    """The camera's lens (configs/hardware/camera-calibration/camera.json ``intrinsics``) as self-mask ``intrinsics``."""
    c = camera.intrinsics(path)
    return {'source': CAMERA_LENS, 'model': c['distortion_model'], 'image_size': c['image_size'],
            'matrix': c['matrix'], 'distortion': c['distortion']}


def load_config(path=None) -> dict:
    """A self-mask config, with the camera's lens as ``intrinsics`` unless it has its own."""
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    if 'intrinsics' not in cfg:
        cfg['intrinsics'] = camera_lens()
    return cfg


def mount_rotation(rpy_deg) -> np.ndarray:
    """Body <- OpenCV camera rotation for a mount ``rpy_deg`` offset (applied in the camera frame)."""
    from scipy.spatial.transform import Rotation
    return R_BODY_CV_DEFAULT @ Rotation.from_euler('xyz', np.asarray(rpy_deg, float), degrees=True).as_matrix()


def distort_normalized(x, y, dist, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted normalized coordinates -> distorted normalized coordinates (OpenCV models)."""
    D = np.r_[np.asarray(dist, np.float64), np.zeros(5)]
    if model == 'equidistant':
        r = np.hypot(x, y)
        th = np.arctan(r)
        thd = th * (1 + D[0] * th**2 + D[1] * th**4 + D[2] * th**6 + D[3] * th**8)
        s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        return x * s, y * s
    k1, k2, p1, p2, k3 = D[:5]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    return (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def undistort_normalized(u, v, K, dist, model: str, iters: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Distorted pixel coordinates -> undistorted normalized coordinates (x/z, y/z).

    Fixed-point inversion of :func:`distort_normalized` (OpenCV's scheme, more iterations). Pixels
    whose inverse does not re-distort to within 0.25 px (outside the lens model's monotonic range)
    come back as NaN."""
    K = np.asarray(K, np.float64)
    xd = (np.asarray(u, np.float64) - K[0, 2]) / K[0, 0]
    yd = (np.asarray(v, np.float64) - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    with np.errstate(all='ignore'):
        for _ in range(iters):
            dx, dy = distort_normalized(x, y, dist, model)
            x, y = x + (xd - dx), y + (yd - dy)
        dx, dy = distort_normalized(x, y, dist, model)
        bad = ~(np.hypot((dx - xd) * K[0, 0], (dy - yd) * K[1, 1]) < 0.25)
    x[bad] = np.nan
    y[bad] = np.nan
    return x, y


def rest_joints() -> dict[str, float]:
    """Rest values of the joints the hardware does not drive (= kinematics.locked_joints(); read
    here directly so this module does not import MuJoCo until it renders)."""
    try:
        cfg = json.loads(hardware_map().read_text()).get('locked_joints', {})
    except (OSError, ValueError):
        cfg = {}
    return {n: float((cfg.get(n) or {}).get('value_rad', 0.0)) for n in ('slew', 'stick')}


def home_joints() -> dict[str, float]:
    """The homed pose (hardware_map ``homing.joints`` home_value_rad, rest values for the rest): the
    pose a ``world_home`` mount is expressed at."""
    try:
        joints = json.loads(hardware_map().read_text())['homing']['joints']
        home = {n: float(h['home_value_rad']) for n, h in joints.items()}
    except (OSError, ValueError, KeyError):
        home = {}
    return {**rest_joints(), **home}


class SelfMasker:
    """Robot self-mask for one camera; see the module docstring."""

    def __init__(self, config: dict | None = None, config_path=None, profile: str | None = None):
        self.config_path = Path(config_path or DEFAULT_CONFIG)
        self.cfg = copy.deepcopy(config) if config is not None else load_config(self.config_path)
        if profile:
            self.cfg['profile'] = profile
        if 'intrinsics' not in self.cfg:
            self.cfg['intrinsics'] = camera_lens()
        intr = self.cfg['intrinsics']
        self.K0 = np.asarray(intr['matrix'], float)
        self.dist = np.asarray(intr.get('distortion', [0, 0, 0, 0, 0]), float)
        self.lens = intr.get('model', 'radtan')
        if self.lens not in ('radtan', 'equidistant', 'pinhole'):
            raise ValueError(f"intrinsics.model must be radtan, equidistant or pinhole, not {self.lens!r}")
        self.size0 = tuple(int(x) for x in intr['image_size'])            # (w, h)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='self-mask')
        self._gl = None                    # (mujoco, model, data, qadr, robot_lut, {(w, h): Renderer})
        self._rest = rest_joints()         # slew / stick where the hardware leaves them
        self._maps = {}                    # (w, h) -> (map_x, map_y, render_w, render_h, fovy_deg, f_render)
        self.last_ms = None

    # ---------------------------------------------------------------- config
    @property
    def mount(self) -> dict:
        return self.cfg['mount']

    def set_mount(self, pos_m=None, rpy_deg=None):
        if pos_m is not None:
            self.mount['pos_m'] = [float(x) for x in pos_m]
        if rpy_deg is not None:
            self.mount['rpy_deg'] = [float(x) for x in rpy_deg]
        if self._gl is not None:
            self._pool.submit(self._apply_mount).result(timeout=30)

    def set_options(self, dilate_px=None, fill_rgb=None):
        if dilate_px is not None:
            self.cfg['dilate_px'] = max(0, int(dilate_px))
        if fill_rgb is not None:
            self.cfg['fill_rgb'] = [int(np.clip(c, 0, 255)) for c in fill_rgb]

    def save(self, path=None, measured: bool | None = None) -> Path:
        """Write the config (atomic replace); a lens from configs/hardware/camera-calibration/camera.json stays there."""
        import os
        if measured is not None:
            self.cfg['measured'] = bool(measured)
        p = Path(path or self.config_path)
        cfg = {k: v for k, v in self.cfg.items() if not (k == 'intrinsics' and v.get('source') == CAMERA_LENS)}
        tmp = p.with_suffix(p.suffix + '.tmp')
        tmp.write_text(json.dumps(cfg, indent=2) + '\n')
        os.replace(tmp, p)
        return p

    def describe(self) -> str:
        m = self.mount
        return (f"self-mask: {self.cfg['profile']} twin, camera on {m['body']} at "
                f"{np.round(np.asarray(m['pos_m']) * 1000, 1).tolist()} mm, rpy {m['rpy_deg']} deg, "
                f"{self.lens} lens, dilate {self.cfg.get('dilate_px', 0)} px"
                + ('' if self.cfg.get('measured') else ' (mount NOT measured)'))

    # ---------------------------------------------------------------- GL thread
    def _init_gl(self):
        import mujoco
        spec = mujoco.MjSpec.from_file(str(SIM / 'models' / f"{self.cfg['profile']}.xml"))
        body = spec.body(self.mount['body'])
        if body is None:
            raise ValueError(f"mount body {self.mount['body']!r} is not in {self.cfg['profile']}")
        body.add_camera(name=CAMERA_NAME)
        spec.visual.global_.offwidth = MAX_RENDER_PX
        spec.visual.global_.offheight = MAX_RENDER_PX
        spec.visual.map.znear = 0.0005                    # the lens sits centimetres from the bucket
        m = spec.compile()
        d = mujoco.MjData(m)
        qadr = {m.joint(j).name: int(m.jnt_qposadr[j]) for j in range(m.njnt)}
        root = m.body('base').id
        anc = np.zeros(m.nbody, bool)
        for b in range(m.nbody):                          # bodies under the machine root
            k = b
            while k > 0 and k != root:
                k = m.body_parentid[k]
            anc[b] = k == root
        robot = anc[m.geom_bodyid] & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
        # A handheld rig may contain only a detached bucket. Hide absent links
        # from both segmentation and its depth buffer, using an explicit body list.
        mask_bodies = self.cfg.get('mask_bodies')
        if mask_bodies is not None:
            if not mask_bodies:
                raise ValueError('mask_bodies must name at least one geometry body')
            ids = [m.body(name).id for name in mask_bodies]
            robot &= np.isin(m.geom_bodyid, ids)
            if not robot.any():
                raise ValueError('mask_bodies contains no visual meshes')
            m.geom_group[~robot] = 5
            m.geom_group[robot] = 1
        lut = np.zeros(m.ngeom + 1, bool)
        lut[:m.ngeom] = robot
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        for g in np.unique(m.geom_group[robot]):
            opt.geomgroup[int(g)] = 1
        opt.sitegroup[:] = 0
        self._gl = {'mujoco': mujoco, 'm': m, 'd': d, 'qadr': qadr, 'lut': lut, 'opt': opt,
                    'cam': m.camera(CAMERA_NAME).id, 'renderers': {}}
        self._apply_mount()

    def _home_body_rotation(self) -> np.ndarray:
        """World <- mount body rotation with the machine at its homed pose."""
        g = self._gl
        self._pose(home_joints())
        return g['d'].xmat[g['m'].body(self.mount['body']).id].reshape(3, 3).copy()

    def _mount_in_body(self) -> tuple[np.ndarray, np.ndarray]:
        """(pos, body <- OpenCV camera rotation) of the mount in its body frame. ``mount.frame``
        ``world_home``: pos_m / rpy_deg are world axes with the machine homed (pos_m from the body
        origin; rpy_deg extrinsic about world x, y, z; 0 0 0 = level, looking along world -x, image
        right = world +y). Otherwise (``body``, the default) they are in the body frame."""
        pos = np.asarray(self.mount['pos_m'], float)
        if self.mount.get('frame') != 'world_home':
            return pos, mount_rotation(self.mount['rpy_deg'])
        from scipy.spatial.transform import Rotation
        Rh = self._home_body_rotation()
        R_world = Rotation.from_euler('xyz', self.mount['rpy_deg'], degrees=True).as_matrix() @ R_BODY_CV_DEFAULT
        return Rh.T @ pos, Rh.T @ R_world

    def _apply_mount(self):
        from scipy.spatial.transform import Rotation
        g = self._gl
        pos, R = self._mount_in_body()
        x, y, z, w = Rotation.from_matrix(R @ CV_TO_MJ).as_quat()
        g['m'].cam_pos[g['cam']] = pos
        g['m'].cam_quat[g['cam']] = [w, x, y, z]

    def _renderer(self, w, h, segmentation: bool):
        g = self._gl
        key = (w, h, segmentation)
        r = g['renderers'].get(key)
        if r is None:
            r = g['mujoco'].Renderer(g['m'], height=h, width=w)
            if segmentation:
                r.enable_segmentation_rendering()
            g['renderers'][key] = r
        return r

    def _remap(self, w: int, h: int):
        """Per output pixel of a w x h work image: where to sample the centred pinhole render."""
        hit = self._maps.get((w, h))
        if hit is not None:
            return hit
        sx, sy = w / self.size0[0], h / self.size0[1]
        K = self.K0 * np.array([[sx, 1, sx], [1, sy, sy], [1, 1, 1]])
        u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.0, np.arange(h, dtype=np.float64))
        if self.lens == 'pinhole':
            x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
        else:
            x, y = undistort_normalized(u, v, K, self.dist, self.lens)
        lim = math.tan(math.radians(MAX_HALF_FOV_DEG))
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < lim) & (np.abs(y) < lim)
        tx = float(np.max(np.abs(x[ok]))) * 1.02
        ty = float(np.max(np.abs(y[ok]))) * 1.02
        f = float(min(K[0, 0], K[1, 1]))
        f = min(f, MAX_RENDER_PX / (2 * tx), MAX_RENDER_PX / (2 * ty))
        rw, rh = int(math.ceil(2 * f * tx)), int(math.ceil(2 * f * ty))
        fovy = math.degrees(2 * math.atan(rh / 2 / f))
        mx = np.where(ok, f * x + rw / 2 - 0.5, -1).astype(np.float32)
        my = np.where(ok, f * y + rh / 2 - 0.5, -1).astype(np.float32)
        out = (mx, my, rw, rh, fovy)
        self._maps[(w, h)] = out
        return out

    def _pose(self, q: dict):
        g = self._gl
        m, d = g['m'], g['d']
        d.qpos[:] = m.qpos0
        for name, val in {**self._rest, **(q or {})}.items():
            a = g['qadr'].get(name)
            if a is not None and val is not None:
                d.qpos[a] = float(val)
        g['mujoco'].mj_kinematics(m, d)
        g['mujoco'].mj_camlight(m, d)

    def _render(self, q: dict, w: int, h: int, rgb: bool):
        import cv2
        if self._gl is None:
            self._init_gl()
        g = self._gl
        mx, my, rw, rh, fovy = self._remap(w, h)
        g['m'].cam_fovy[g['cam']] = fovy
        self._pose(q)
        r = self._renderer(rw, rh, segmentation=not rgb)
        r.update_scene(g['d'], camera=g['cam'], scene_option=g['opt'])
        img = r.render()
        if rgb:
            return cv2.remap(np.ascontiguousarray(img), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        objid, objtype = img[..., 0], img[..., 1]
        hit = (objtype == int(g['mujoco'].mjtObj.mjOBJ_GEOM)) & (objid >= 0)
        robot = np.zeros(hit.shape, np.uint8)
        robot[hit] = g['lut'][objid[hit]]
        return cv2.remap(robot, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ---------------------------------------------------------------- public
    def _work_size(self, w: int, h: int) -> tuple[int, int]:
        ww = min(w, int(self.cfg.get('work_width', 640)))
        return ww, max(1, round(h * ww / w))

    def mask(self, q: dict, shape) -> np.ndarray:
        """Boolean HxW mask (True = machine) for a frame of ``shape`` (h, w[, c]) at joints ``q``."""
        import cv2
        h, w = int(shape[0]), int(shape[1])
        ww, wh = self._work_size(w, h)
        t0 = time.perf_counter()
        m = self._pool.submit(self._render, dict(q or {}), ww, wh, False).result(timeout=30)
        if (ww, wh) != (w, h):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        k = int(self.cfg.get('dilate_px', 0))
        if k > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        self.last_ms = (time.perf_counter() - t0) * 1000
        return m.astype(bool)

    def apply(self, rgb, q: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (copy of ``rgb`` with the machine pixels filled, boolean mask)."""
        a = np.asarray(rgb)
        m = self.mask(q, a.shape)
        out = a.copy()
        out[m] = np.asarray(self.cfg.get('fill_rgb', [0, 0, 0]), a.dtype)[:out.shape[-1]] if out.ndim == 3 else 0
        return out, m

    def render_view(self, q: dict, shape) -> np.ndarray:
        """The twin as this camera sees it (RGB, lens-distorted), for checking the mount without a frame."""
        import cv2
        h, w = int(shape[0]), int(shape[1])
        ww, wh = self._work_size(w, h)
        img = self._pool.submit(self._render, dict(q or {}), ww, wh, True).result(timeout=30)
        return img if (ww, wh) == (w, h) else cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    def render_locator(self, q: dict, shape, frustum_m: float = 0.25, azimuth: float = 128.5,
                       elevation: float = -24.8, distance: float = 1.2) -> np.ndarray:
        """The whole twin from outside at joints ``q`` (default: the policy rollout app's ``overview``
        view; ``azimuth`` / ``elevation`` degrees and ``distance`` m orbit the bucket), with this camera
        drawn on it: a yellow frustum along its optical axis (pinhole, from the intrinsics), its OpenCV
        axes (red x right, green y down, blue z optical) and world axes at the origin (red x, green y,
        blue z). For checking where the mount is."""
        h, w = int(shape[0]), int(shape[1])
        return self._pool.submit(self._render_locator, dict(q or {}), w, h, frustum_m,
                                 float(azimuth), float(elevation), float(distance)).result(timeout=30)

    def _render_locator(self, q: dict, w: int, h: int, frustum_m: float, azimuth: float, elevation: float,
                        distance: float):
        import mujoco
        from .viz import add_frame, add_segment
        if self._gl is None:
            self._init_gl()
        g = self._gl
        m, d = g['m'], g['d']
        self._pose(q)
        key = ('locator', w, h)
        r = g['renderers'].get(key)
        if r is None:
            r = g['renderers'][key] = mujoco.Renderer(m, height=h, width=w)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1                              # the whole machine, not only the masked bodies
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = d.xpos[m.body(self.mount['body']).id]
        cam.azimuth, cam.elevation, cam.distance = azimuth, max(-89.0, min(89.0, elevation)), distance
        r.update_scene(d, camera=cam, scene_option=opt)
        add_frame(r.scene, np.zeros(3), np.eye(3), length=0.15, width=0.004)
        pos = d.cam_xpos[g['cam']].copy()
        R = d.cam_xmat[g['cam']].reshape(3, 3) @ CV_TO_MJ   # world <- OpenCV camera
        (fx, _, cx), (_, fy, cy), _ = self.K0
        iw, ih = self.size0
        corners = [pos + frustum_m * R @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
                   for u, v in ((0, 0), (iw, 0), (iw, ih), (0, ih))]
        yellow = (1, .85, .1, 1)
        for i, c in enumerate(corners):
            add_segment(r.scene, pos, c, yellow, 0.002)
            add_segment(r.scene, c, corners[(i + 1) % 4], yellow, 0.002)
        add_frame(r.scene, pos, R, length=0.06, width=0.003)
        return np.ascontiguousarray(r.render())

    def joints(self) -> dict[str, list[float]]:
        """Articulated joints of the twin -> [lo, hi] rad (welded joints are not in the model)."""
        def _j():
            if self._gl is None:
                self._init_gl()
            m = self._gl['m']
            return {m.joint(j).name: ([float(x) for x in m.jnt_range[j]] if m.jnt_limited[j] else [-np.pi, np.pi])
                    for j in range(m.njnt)}
        return self._pool.submit(_j).result(timeout=30)

    def close(self):
        def _close():
            if self._gl is not None:
                for r in self._gl['renderers'].values():
                    r.close()
                self._gl = None
        try:
            self._pool.submit(_close).result(timeout=10)
        except Exception:
            pass
        self._pool.shutdown(wait=False)


def overlay(rgb, mask, color=(255, 40, 40), alpha: float = 0.45) -> np.ndarray:
    """Tint the masked pixels and outline the mask edge (for viewers)."""
    import cv2
    out = np.asarray(rgb).copy()
    m = np.asarray(mask, bool)
    col = np.asarray(color, np.float32)
    out[m] = (out[m] * (1 - alpha) + col * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, tuple(int(c) for c in color), max(1, out.shape[1] // 640))
    return out
