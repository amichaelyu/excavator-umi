"""Kinematics for the excavator arm twin: FK, IK, workspace, joint limits.

There is one chain: ``q = [slew, boom, stick, bucket]`` (rad). The base spins (slew, about
world +z) and the arm has three pitch joints - boom (shoulder), stick (arm) and bucket
(wrist) - so a 3D point position *and* the bucket pitch are independently reachable.

The bucket orientation is ``Rz(slew) @ R_{-y}(pitch) @ R_rest`` with
``pitch = boom + (stick - stick_rest) + bucket`` (nose up positive). Pitch and ``R_rest`` refer
to the rest pose :func:`rest_q`: slew 0, boom 0, bucket 0 and the stick at its top hard stop
(the hardware map's ``locked_joints``, :func:`locked_joints`, where homing leaves it), so pitch 0 is the bucket
orientation the earlier two-joint twin called rest and ``actions.csv`` pitch targets keep their
meaning.

The real machine can drive all four joints (slew pins 19 / 21, stick 23 / 22). The pulse compiler's default
acts on boom and bucket only (:data:`HARDWARE_JOINTS`) and holds slew / stick at their current (default: rest)
values in the twin - see :func:`full_chain_q` and the ``locked`` arguments; the runner's 4-joint profile
(policy.runner ``pi05_tabletop_excavator``) drives all of :data:`JOINTS`.

:func:`ik` solves the slew in closed form from the target heading (the arm plane is offset
``y0`` from the slew axis), boom/stick/bucket from the three-link planar chain, then polishes
with MuJoCo's Jacobian. Position is the primary task and pitch is followed in its nullspace, so
a pitch request never costs position accuracy; without one the redundant pitch is chosen near
the middle of the joint ranges.

Profiles: ``full4`` (default; configs/hardware/robot.json priors, slew +-165 deg) and ``full4_calibrated``
(measured boom / bucket / stick ranges and speeds, continuous slew; configs/hardware/robot_full4_calibrated.json,
written by ``python -m digital_twin.sim.build --calibrated``).

Controlled point: the ``tooth_tip`` site (default) or the ``bucket`` body origin.
Orientation always refers to the bucket body frame (the site frame is identical).

Simulator-only; no robot commands are issued.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes
from scipy.spatial.transform import Rotation
from policy.data.local_config import ensure_local

from ..paths import CONFIG, SIM, hardware_map
from .pivots import fk as pivot_fk

JOINTS = ['slew', 'boom', 'stick', 'bucket']      # the twin's joints, in qpos / ctrl order
NQ = len(JOINTS)
CALIBRATION_JOINTS = JOINTS                       # same chain as the video / calibration tools
HARDWARE_JOINTS = ['boom', 'bucket']              # what the real machine drives (pins / policy)
PITCH_JOINTS = ['boom', 'stick', 'bucket']
DEFAULT_PROFILE = 'full4'


def load_locked_joints(path=None) -> dict[str, float]:
    """Rest values [rad] of the joints the hardware does not drive, from hardware_map.json
    ``locked_joints.<name>.value_rad`` (default: the map in use, digital_twin.paths.hardware_map()):
    slew (no control, reset by hand) and the stick (top hard stop, where homing leaves it)."""
    try:
        cfg = json.loads(Path(path or hardware_map()).read_text()).get('locked_joints', {})
    except (OSError, ValueError):
        cfg = {}
    return {n: float((cfg.get(n) or {}).get('value_rad', 0.0)) for n in ('slew', 'stick')}


@functools.lru_cache(maxsize=None)
def _locked_joints_at(path: str) -> tuple[tuple[str, float], ...]:
    return tuple(load_locked_joints(path).items())


def locked_joints() -> dict[str, float]:
    """Where the hardware leaves slew / stick ({name: rad}), from the hardware map in use; read once per
    map path (a fresh dict each call), again after :func:`reload`."""
    return dict(_locked_joints_at(str(hardware_map())))


def rest_q() -> np.ndarray:
    """The rest pose (pitch 0) over :data:`JOINTS`: slew / stick at :func:`locked_joints`, the rest 0."""
    locked = locked_joints()
    return np.array([locked.get(n, 0.0) for n in JOINTS])


def reload() -> None:
    """Forget the cached hardware-map reads: call after rewriting the map in place (the hub's calibration
    does). Arms built before keep the rest pose they were built with."""
    _locked_joints_at.cache_clear()


PLANE = [0, 2]                                    # world axes spanned by the arm plane at slew 0 (x, z)
CONTROL_HZ = 20.0
FRAMESKIP = 10
POINTS = ('tooth_tip', 'bucket')
TIP_OFFSET_IN_BUCKET = np.array([-0.028, 0.0, -0.049])  # tooth_tip site pos in the bucket frame
UNLIMITED_RANGE = (-np.pi, np.pi)                 # stands in for a continuous joint's range


# ------------------------------------------------------------------- config
PROFILE_CONFIGS = {'full4_calibrated': 'robot_full4_calibrated.json'}   # profile -> robot config (default robot.json)


def load_config(profile: str | None = None) -> dict:
    """robot.json, or the profile's own config (``full4_calibrated``: configs/hardware/robot_full4_calibrated.json,
    measured ranges / speeds written by ``python -m digital_twin.sim.build --calibrated``; per machine, created from
    its example when missing)."""
    return json.loads(ensure_local(CONFIG / PROFILE_CONFIGS.get(profile, 'robot.json')).read_text())


def joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) joint position limits [rad] from the MuJoCo model; a continuous joint
    (the calibrated slew) gets :data:`UNLIMITED_RANGE`."""
    ids = [model.joint(n).id for n in JOINTS]
    rng = np.array([model.jnt_range[i] if model.jnt_limited[i] else UNLIMITED_RANGE for i in ids], float)
    return rng[:, 0].copy(), rng[:, 1].copy()


def speed_limits(cfg: dict | None = None, joints=None) -> tuple[np.ndarray, np.ndarray]:
    """(positive-direction, negative-direction) joint speed limits [rad/s] from robot.json."""
    j = (cfg or load_config())['joints']
    joints = joints or JOINTS
    vp = np.array([j[n].get('speed_positive_rad_s', j[n]['speed_rad_s']) for n in joints])
    vn = np.array([j[n].get('speed_negative_rad_s', j[n]['speed_rad_s']) for n in joints])
    return vp, vn


def within_limits(q, lo, hi, tol: float = 1e-9) -> bool:
    q = np.asarray(q)
    return bool(np.all(q >= lo - tol) and np.all(q <= hi + tol))


def speed_ratio(Q: np.ndarray, dt: float, v_pos, v_neg) -> np.ndarray:
    """|dq/dt| / limit per tick and joint (> 1 means the limit is exceeded)."""
    dq = np.diff(np.asarray(Q), axis=0) / dt
    return np.abs(dq) / np.where(dq >= 0, v_pos, v_neg)


def min_move_time(q0, q1, v_pos, v_neg, peak_factor: float = 1.0) -> np.ndarray:
    """Per-joint minimum time to move q0 -> q1 when the profile's peak speed is
    ``peak_factor`` x the average speed (1.875 for a quintic, 1 for constant speed)."""
    d = np.asarray(q1, float) - np.asarray(q0, float)
    return peak_factor * np.abs(d) / np.where(d >= 0, v_pos, v_neg)


def full_chain_q(q_hw, held=None) -> np.ndarray:
    """[slew, boom, stick, bucket] from hardware joints [boom, bucket] (..., 2): slew and stick
    at ``held`` ({name: rad}, default their rest values)."""
    q_hw = np.asarray(q_hw, float)
    full = np.broadcast_to(rest_q(), q_hw.shape[:-1] + (NQ,)).copy()
    for n, v in (held or {}).items():
        full[..., JOINTS.index(n)] = v
    for i, n in enumerate(HARDWARE_JOINTS):
        full[..., JOINTS.index(n)] = q_hw[..., i]
    return full


def hardware_q(q) -> np.ndarray:
    """The hardware joints [boom, bucket] of full-chain q (..., 4)."""
    return np.asarray(q, float)[..., [JOINTS.index(n) for n in HARDWARE_JOINTS]]


# ------------------------------------------------------------------ rotations
def rot_pitch(phi) -> np.ndarray:
    """Rotation about -y by phi (x toward +z, i.e. nose up)."""
    return Rotation.from_rotvec(np.array([0.0, -1.0, 0.0]) * phi).as_matrix()


def rot_yaw(theta) -> np.ndarray:
    """Rotation about +z by theta (the slew)."""
    return Rotation.from_euler('z', theta).as_matrix()


def parse_rotation(target_rotation=None, *, pitch_deg=None, quat_wxyz=None, rotmat=None, rotvec=None):
    """Normalise a rotation spec. Returns (kind, value) with kind in
    {'none', 'pitch', 'matrix'}; 'pitch' value in rad, 'matrix' a 3x3 world<-bucket.

    ``target_rotation`` may be: None, a scalar (pitch in degrees), a length-4 quaternion
    (w, x, y, z), a 3x3 matrix, or a scipy Rotation. Keyword forms take precedence.
    """
    if pitch_deg is not None:
        return 'pitch', float(np.radians(pitch_deg))
    if quat_wxyz is not None:
        q = np.asarray(quat_wxyz, float)
        return 'matrix', Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
    if rotmat is not None:
        return 'matrix', Rotation.from_matrix(np.asarray(rotmat, float)).as_matrix()
    if rotvec is not None:
        return 'matrix', Rotation.from_rotvec(np.asarray(rotvec, float)).as_matrix()
    r = target_rotation
    if r is None:
        return 'none', None
    if isinstance(r, Rotation):
        return 'matrix', r.as_matrix()
    a = np.asarray(r, float)
    if a.ndim == 0 or a.size == 1:
        return 'pitch', float(np.radians(a.ravel()[0]))
    if a.shape == (4,):
        return parse_rotation(quat_wxyz=a)
    if a.shape == (3, 3):
        return parse_rotation(rotmat=a)
    raise ValueError('target_rotation: expected pitch (deg), quaternion wxyz, 3x3 matrix or Rotation')


def best_pitch_for(R_target: np.ndarray, slew: float = 0.0, R_rest: np.ndarray = np.eye(3)) -> tuple[float, float]:
    """Pitch phi minimising the angle between R_target and Rz(slew) R_{-y}(phi) R_rest.

    Returns (phi, residual angle [rad]) - the residual is the part of the target orientation
    (roll, or yaw away from the slew) that the arm cannot realise.
    """
    M = rot_yaw(slew).T @ R_target @ R_rest.T
    # maximise trace(Ry(th)^T M); R_{-y}(phi) = Ry(-phi)
    th = np.arctan2(M[0, 2] - M[2, 0], M[0, 0] + M[2, 2])
    phi = -th
    R_best = rot_yaw(slew) @ rot_pitch(phi) @ R_rest
    return float(phi), rotation_angle(R_best, R_target)


def rotation_angle(Ra, Rb) -> float:
    """Geodesic angle [rad] between two rotation matrices."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2
    return float(np.arccos(np.clip(c, -1, 1)))


# --------------------------------------------------------------------- FK
def load_model(profile: str = DEFAULT_PROFILE) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SIM / 'models' / f'{profile}.xml'))


def pitch_of(q):
    """Bucket pitch of q (..., 4): boom + (stick - stick_rest) + bucket."""
    q = np.asarray(q, float)
    return q[..., 1] + q[..., 2] - rest_q()[2] + q[..., 3]


def fk(model: mujoco.MjModel, data: mujoco.MjData, q) -> dict:
    """MuJoCo FK for q = [slew, boom, stick, bucket].

    Returns dict with ``tip_pos``, ``tip_rot`` (3x3 world<-site), ``tip_quat`` (wxyz),
    ``bucket_pos``, ``bucket_rot``, ``bucket_quat``, ``pitch`` and ``slew`` (rad).
    """
    adr = model.jnt_qposadr[[model.joint(n).id for n in JOINTS]]
    data.qpos[adr] = q
    mujoco.mj_kinematics(model, data)
    site = data.site('tooth_tip'); body = data.body('bucket')
    tip_rot = site.xmat.reshape(3, 3).copy()
    q = np.asarray(q, float)
    return {'tip_pos': site.xpos.copy(), 'tip_rot': tip_rot, 'tip_quat': _wxyz(tip_rot),
            'bucket_pos': body.xpos.copy(), 'bucket_rot': body.xmat.reshape(3, 3).copy(),
            'bucket_quat': body.xquat.copy(), 'pitch': float(pitch_of(q)), 'slew': float(q[0])}


def analytic_fk(q, cfg: dict | None = None, tip_offset=TIP_OFFSET_IN_BUCKET) -> dict:
    """Analytic FK of q = [slew, boom, stick, bucket] from the anatomical pivots in robot.json
    (the pivot chain of :func:`pivots.fk`); no MuJoCo involved."""
    q = np.asarray(q, float)
    rot, pos = pivot_fk(q, cfg or load_config())
    Rb, pb = rot[4], pos[4]
    return {'tip_pos': pb + Rb @ np.asarray(tip_offset), 'tip_rot': Rb, 'tip_quat': _wxyz(Rb),
            'bucket_pos': pb, 'bucket_rot': Rb, 'bucket_quat': _wxyz(Rb),
            'pitch': float(pitch_of(q)), 'slew': float(q[0])}


def _wxyz(R) -> np.ndarray:
    x = Rotation.from_matrix(R).as_quat()
    return np.r_[x[3], x[:3]]


class Arm:
    """MuJoCo arm model + data with FK / Jacobian for one controlled point; q = [slew, boom,
    stick, bucket]."""

    def __init__(self, profile: str = DEFAULT_PROFILE, point: str = 'tooth_tip', model: mujoco.MjModel | None = None):
        if point not in POINTS:
            raise ValueError(f'point must be one of {POINTS}')
        self.profile, self.point = profile, point
        self.model = model if model is not None else load_model(profile)
        self.data = mujoco.MjData(self.model)
        self.cfg = load_config(profile)
        self.joints = list(JOINTS)
        self.nq = NQ
        self.pitch_mask = np.array([n != 'slew' for n in JOINTS])
        self.jid = np.array([self.model.joint(n).id for n in JOINTS])
        self.qadr = self.model.jnt_qposadr[self.jid]
        self.dadr = self.model.jnt_dofadr[self.jid]
        self.lo, self.hi = joint_limits(self.model)
        self.v_pos, self.v_neg = speed_limits(self.cfg)
        self.site = self.model.site('tooth_tip').id
        self.bucket = self.model.body('bucket').id
        self.act = np.array([self.model.actuator(n).id for n in JOINTS])
        self._jacp = np.zeros((3, self.model.nv))
        self.q_rest = rest_q()
        self.fk(self.q_rest)
        self.R_rest = self.data.xmat[self.bucket].reshape(3, 3).copy()
        # The pitch joints rotate about the slew frame's -y axis: the controlled point keeps a
        # constant lateral offset y0 from the plane through the slew axis. R_plane / y0 are for
        # slew = 0; the plane turns about the slew axis (world +z) with the slew.
        slew = self.data.body('slew')
        self.R_plane = slew.xmat.reshape(3, 3).copy()
        self.slew_origin = slew.xpos.copy()
        self.y0 = float((self.R_plane.T @ (self.point_pos() - slew.xpos))[1])

    def point_pos(self) -> np.ndarray:
        if self.point == 'tooth_tip':
            return self.data.site_xpos[self.site].copy()
        return self.data.xpos[self.bucket].copy()

    def fk(self, q) -> np.ndarray:
        """Controlled-point position for q (kinematics only)."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.point_pos()

    def pose(self, q) -> dict:
        return fk(self.model, self.data, q)

    def bucket_rot(self) -> np.ndarray:
        return self.data.xmat[self.bucket].reshape(3, 3).copy()

    @staticmethod
    def pitch(q):
        """Bucket pitch about the arm-plane -y axis (+ = nose up), relative to the rest pose."""
        return pitch_of(q)

    @staticmethod
    def slew(q):
        return np.asarray(q)[..., 0]

    def rotation_for(self, pitch: float, slew: float = 0.0) -> np.ndarray:
        """Bucket rotation for a pitch and slew: Rz(slew) R_plane R_{-y}(pitch) R_rest."""
        return rot_yaw(slew) @ self.R_plane @ rot_pitch(pitch) @ self.R_rest

    def body_pitch(self) -> float:
        """Same quantity as :meth:`pitch`, measured from MuJoCo body frames (for verification)."""
        Rs = self.data.body('slew').xmat.reshape(3, 3)
        x = Rs.T @ (self.bucket_rot() @ self.R_rest.T)[:, 0]
        return float(np.arctan2(x[2], x[0]))

    def jac(self, q) -> np.ndarray:
        self.fk(q)
        mujoco.mj_comPos(self.model, self.data)
        if self.point == 'tooth_tip':
            mujoco.mj_jacSite(self.model, self.data, self._jacp, None, self.site)
        else:
            mujoco.mj_jacBody(self.model, self.data, self._jacp, None, self.bucket)
        return self._jacp[:, self.dadr].copy()

    def slew_for(self, p, behind: bool = False) -> tuple[float, float, float]:
        """(slew, in-plane distance u from the slew axis, unreachable radial gap) that puts the
        arm plane through world point p. The plane is offset y0 from the axis, so its heading is
        the target bearing minus asin(y0 / r). ``behind`` picks the other solution (u < 0: the
        point behind the slew axis in the arm frame, i.e. reaching over the cab)."""
        d = np.asarray(p, float)[:2] - self.slew_origin[:2]
        r = float(np.hypot(*d))
        gap = max(0.0, abs(self.y0) - r)
        u = np.sqrt(max(r * r - self.y0 * self.y0, 0.0)) * (-1 if behind else 1)
        s = np.arctan2(d[1], d[0]) - np.arctan2(self.y0, u)
        return float((s + np.pi) % (2 * np.pi) - np.pi), float(u), gap

    def slew_ok(self, s: float) -> bool:
        return bool(self.lo[0] - 1e-9 <= s <= self.hi[0] + 1e-9)

    def plane_for(self, p) -> tuple[float, float, float]:
        """(x_s, z, gap) of a world point: coordinates in the arm plane turned to face it (x_s
        measured as at slew = 0) and the unreachable radial gap left when p is closer to the
        slew axis than |y0|."""
        p = np.asarray(p, float)
        _, u, gap = self.slew_for(p)
        return float(self.slew_origin[0] + u), float(p[2]), gap


def _two_link(A0, L1, L2, xs, z, lo, hi):
    """Broadcasting two-link IK in the arm plane. Returns q (..., 2 branches, 2), valid (..., 2)."""
    xs, z = np.broadcast_arrays(np.asarray(xs, float), np.asarray(z, float))
    sh = (2,) + (1,) * xs.ndim
    W = np.stack([xs, z]) - A0.reshape(sh)
    a = L1 @ L2
    b = L1[1] * L2[0] - L1[0] * L2[1]
    K = (np.sum(W ** 2, 0) - L1 @ L1 - L2 @ L2) / 2
    c = K / np.hypot(a, b)
    reach = np.abs(c) <= 1
    acos = np.arccos(np.clip(c, -1, 1))
    qs, oks = [], []
    for sgn in (1, -1):
        q2 = np.arctan2(b, a) + sgn * acos
        V = L1.reshape(sh) + PlanarChain.rot(q2, L2.reshape(sh))
        q1 = np.arctan2(W[1], W[0]) - np.arctan2(V[1], V[0])
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        q2 = (q2 + np.pi) % (2 * np.pi) - np.pi
        q = np.stack([q1, q2], -1)
        ok = reach & np.all((q >= lo) & (q <= hi), -1)
        qs.append(q); oks.append(ok)
    return np.stack(qs, -2), np.stack(oks, -1)


class PlanarChain:
    """Closed-form boom / stick / bucket chain in the arm plane (x_s, z) at slew = 0, extracted
    from the MuJoCo model for the arm's controlled point: links L1 (boom pivot -> stick pivot),
    L2 (stick -> bucket pivot) and L3 (bucket pivot -> point). Used for pitch-feasibility maps and
    IK seeding; the IK itself is the numerical solver. Angles are the pitch joints [boom, stick,
    bucket]; ``lo`` / ``hi`` are theirs."""

    def __init__(self, arm: Arm, joint_margin: float = 0.0):
        arm.fk(np.zeros(NQ))
        d, m = arm.data, arm.model
        anchors = [d.xanchor[m.joint(n).id][PLANE].copy() for n in PITCH_JOINTS]
        pts = anchors + [arm.point_pos()[PLANE]]
        self.A0 = anchors[0]
        self.links = [b - a for a, b in zip(pts[:-1], pts[1:])]
        self.L1, self.L2, self.L3 = self.links
        self.offset = -arm.q_rest[2]              # pitch = sum of the chain's joints + offset
        self.lo = arm.lo[1:] + joint_margin
        self.hi = arm.hi[1:] - joint_margin

    @staticmethod
    def rot(th, v):
        """Rotate planar (x, z) vectors by th about the -y joint axis (x toward z)."""
        c, s = np.cos(th), np.sin(th)
        return np.stack([v[0] * c - v[1] * s, v[0] * s + v[1] * c])

    def fk(self, q):
        """In-plane point (N, 2) for pitch-joint angles q (N, 3)."""
        q = np.atleast_2d(np.asarray(q, float))
        p = np.repeat(self.A0[:, None], len(q), 1)
        th = np.zeros(len(q))
        for i, L in enumerate(self.links):
            th = th + q[:, i]
            p = p + self.rot(th, L[:, None])
        return p.T

    def ik(self, xs, z, phi):
        """Broadcasting IK for (point x_s, point z, bucket pitch phi): the wrist (bucket pivot) is
        fixed by the point and pitch, then boom / stick solve it as two links.
        Returns q (..., 2 branches, 3), valid (..., 2)."""
        xs, z, phi = np.broadcast_arrays(np.asarray(xs, float), np.asarray(z, float), np.asarray(phi, float))
        last = phi - self.offset                  # absolute angle of the last link
        wrist = np.stack([xs, z]) - self.rot(last, self.L3.reshape((2,) + (1,) * xs.ndim))
        q12, ok = _two_link(self.A0, self.L1, self.L2, wrist[0], wrist[1], self.lo[:2], self.hi[:2])
        q3 = last[..., None] - q12.sum(-1)
        q3 = (q3 + np.pi) % (2 * np.pi) - np.pi
        ok = ok & (q3 >= self.lo[2]) & (q3 <= self.hi[2])
        return np.concatenate([q12, q3[..., None]], -1), ok

    def pitch_range(self) -> tuple[float, float]:
        return float(self.lo.sum() + self.offset), float(self.hi.sum() + self.offset)


# ----------------------------------------------------------------- envelope
@dataclass
class Envelope:
    """Reachable point region: 2D occupancy in the arm plane (x_s, z) at slew = 0. The plane
    turns to face each point, so the 3D workspace is this region swept about the slew axis
    within ``slew_range``."""
    x0: float
    z0: float
    cell: float
    grid: np.ndarray        # (nx, nz) bool, eroded by margin
    y0: float
    samples: np.ndarray     # (M,2) raw FK samples (x_s, z)
    phi: np.ndarray | None = None      # (P,) bucket-pitch grid [rad]
    phi_ok: np.ndarray | None = None   # (nx, nz, P) pitch achievable at cell centre
    slew_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))   # slew axis (world)
    slew_range: tuple[float, float] = UNLIMITED_RANGE
    planar: bool = False    # slew held at 0: points are taken in the fixed plane y = y0 (lateral ignored)

    def cells(self, p):
        xs, z, _, _ = self.to_plane(p)
        i = np.clip(np.floor((xs - self.x0) / self.cell).astype(int), 0, self.grid.shape[0] - 1)
        k = np.clip(np.floor((z - self.z0) / self.cell).astype(int), 0, self.grid.shape[1] - 1)
        return i, k

    def to_plane(self, p):
        """(x_s, z, unreachable sideways part, slew in range) for world points (N, 3). Planar
        (slew held at 0): the sideways part is the offset from the fixed plane y = y0."""
        p = np.atleast_2d(p)
        if self.planar:
            return p[:, 0], p[:, 2], p[:, 1] - self.y0, np.ones(len(p), bool)
        d = p[:, :2] - self.slew_origin[:2]
        r = np.hypot(d[:, 0], d[:, 1])
        u = np.sqrt(np.maximum(r * r - self.y0 * self.y0, 0.0))
        s = np.arctan2(d[:, 1], d[:, 0]) - np.arctan2(self.y0, u)
        s = (s + np.pi) % (2 * np.pi) - np.pi
        lo, hi = self.slew_range
        return self.slew_origin[0] + u, p[:, 2], np.maximum(0.0, abs(self.y0) - r), (s >= lo) & (s <= hi)

    def lateral(self, p) -> np.ndarray:
        """Unreachable sideways part of world points: the gap inside the |y0| circle around the
        slew axis (zero elsewhere)."""
        return self.to_plane(p)[2]

    def contains(self, p) -> np.ndarray:
        """Reachability of world points (slew in range and in-plane point in the region)."""
        xs, z, _, ok = self.to_plane(p)
        i = np.floor((xs - self.x0) / self.cell).astype(int)
        k = np.floor((z - self.z0) / self.cell).astype(int)
        inside = ok & (i >= 0) & (k >= 0) & (i < self.grid.shape[0]) & (k < self.grid.shape[1])
        out = np.zeros(len(xs), bool)
        out[inside] = self.grid[i[inside], k[inside]]
        return out

    def pitch_range(self, p) -> tuple[float, float] | None:
        """(min, max) achievable pitch [rad] at point p, or None if unreachable. The achievable
        pitches can form more than one interval; this is the hull."""
        if self.phi_ok is None or not self.contains(p)[0]:
            return None
        i, k = self.cells(p)
        ok = self.phi_ok[i[0], k[0]]
        return (float(self.phi[ok].min()), float(self.phi[ok].max())) if ok.any() else None


def reachable_envelope(arm: Arm | None = None, n: int | None = None, cell: float = 0.003, joint_margin: float = 0.03,
                       margin: float = 0.006, z_min: float = 0.012, phi_step_deg: float = 1.0,
                       locked: dict | None = None, phi_margin_deg: float = 4.0) -> Envelope:
    """Sample the planar chain over the pitch joints' ranges (``n`` per free joint) and rasterise
    the point locus in the arm plane. Also computes, per cell, which bucket pitches are
    achievable: the three-link IK has a valid branch for it.

    ``locked`` = {joint: value} held fixed: a locked stick (as on the hardware) leaves the boom /
    bucket region, where a pitch counts as achievable within ``phi_margin_deg`` of an arm
    branch's pitch; a locked slew makes the envelope planar (the fixed plane at slew 0)."""
    arm = arm or Arm()
    locked = dict(locked or {})
    chain = PlanarChain(arm, joint_margin)
    free = [nm not in locked for nm in PITCH_JOINTS]
    n = n or (60 if all(free) else 160)
    axes = [np.linspace(lo, hi, n) if f else np.array([locked[nm]])
            for nm, f, lo, hi in zip(PITCH_JOINTS, free, chain.lo, chain.hi)]
    pts = chain.fk(np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3))
    pts_ok = pts[pts[:, 1] >= z_min]
    x0, z0 = pts_ok.min(0) - 2 * cell
    shape = np.ceil((pts_ok.max(0) + 2 * cell - [x0, z0]) / cell).astype(int) + 1
    grid = np.zeros(shape, bool)
    ij = np.floor((pts_ok - [x0, z0]) / cell).astype(int)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = binary_fill_holes(binary_closing(grid, iterations=2))
    grid[:, :max(0, int(np.ceil((z_min - z0) / cell)))] = False
    grid = binary_erosion(grid, iterations=max(1, int(round(margin / cell))))
    phi_lo, phi_hi = chain.pitch_range()
    phi = np.arange(phi_lo, phi_hi + 1e-9, np.radians(phi_step_deg))
    phi_ok = np.zeros((*shape, len(phi)), bool)
    xs = x0 + (np.arange(shape[0]) + .5) * cell
    zs = z0 + (np.arange(shape[1]) + .5) * cell
    if 'stick' in locked:
        # Boom + stick rigid: a two-link chain whose two arm branches fix the pitch; a pitch
        # counts as achievable within phi_margin_deg of a valid branch's pitch.
        q2 = float(locked['stick'])
        L1, L2 = chain.L1 + PlanarChain.rot(q2, chain.L2), PlanarChain.rot(q2, chain.L3)
        tol = np.radians(phi_margin_deg)
        for i, x in enumerate(xs):
            q, ok = _two_link(chain.A0, L1, L2, x, zs, chain.lo[[0, 2]], chain.hi[[0, 2]])   # (nz, 2, 2), (nz, 2)
            branch_phi = q.sum(-1) + q2 + chain.offset
            near = np.abs(phi[None, None, :] - branch_phi[:, :, None]) <= tol
            phi_ok[i] = (near & ok[:, :, None]).any(1)
    else:
        for i, x in enumerate(xs):
            _, ok = chain.ik(x, zs[:, None], phi[None, :])   # (nz, P, 2 branches)
            phi_ok[i] = ok.any(-1)
    phi_ok &= grid[:, :, None]
    slew_range = (float(arm.lo[0]), float(arm.hi[0]))
    return Envelope(float(x0), float(z0), cell, grid, arm.y0, pts, phi, phi_ok,
                    arm.slew_origin.copy(), slew_range, planar='slew' in locked)


# ---------------------------------------------------------------------- IK
def analytic_seeds(arm: Arm, target, phi=None, locked: dict | None = None) -> list[np.ndarray]:
    """Closed-form joint candidates for one target, valid branches only. The slew faces the
    target (also the reach-over-the-cab solution, if in range); without ``phi`` the redundant
    pitch is sampled across its range. With ``phi`` the branch whose pitch is closest comes
    first, otherwise candidates nearest the middle of the joint ranges.

    ``locked`` = {joint: value}: a locked slew keeps the arm plane at that heading; a locked
    stick turns boom + stick into one rigid link, leaving a two-link boom / bucket chain whose
    two arm branches fix the pitch."""
    target = np.asarray(target, float)
    locked = locked or {}
    chain = PlanarChain(arm)
    if 'slew' in locked:
        s = float(locked['slew'])
        d = rot_yaw(-s) @ (target - arm.slew_origin)
        headings = [(s, float(arm.slew_origin[0] + d[0]), float(target[2]))]
    else:
        headings = []
        for behind in (False, True):
            s, u, _ = arm.slew_for(target, behind)
            if arm.slew_ok(s):
                headings.append((s, float(arm.slew_origin[0] + u), float(target[2])))
    seeds = []
    for s, xs, z in headings:
        if 'stick' in locked:
            q2 = float(locked['stick'])
            L1 = chain.L1 + PlanarChain.rot(q2, chain.L2)
            L2 = PlanarChain.rot(q2, chain.L3)
            q, valid = _two_link(chain.A0, L1, L2, xs, z, chain.lo[[0, 2]], chain.hi[[0, 2]])   # (2, 2), (2,)
            seeds += [np.r_[s, q[b, 0], q2, q[b, 1]] for b in range(2) if valid[b]]
        else:
            phis = np.atleast_1d(phi) if phi is not None else np.linspace(*chain.pitch_range(), 37)
            q, valid = chain.ik(xs, z, phis)                  # (P, 2, 3), (P, 2)
            seeds += [np.r_[s, q[i, b]] for i, b in zip(*np.nonzero(valid))]
    if phi is not None:
        seeds.sort(key=lambda s: abs(pitch_of(s) - phi))
    else:
        mid = 0.5 * (arm.lo + arm.hi)
        seeds.sort(key=lambda s: np.linalg.norm(((s - mid) / (arm.hi - arm.lo))[1:]))
    return seeds


def analytic_seed(arm: Arm, env, target, phi=None) -> np.ndarray | None:
    s = analytic_seeds(arm, target, phi)
    return s[0] if s else None


def ik_step_solve(arm: Arm, target, phi_t, q0, damping=2e-3, iters=150, tol_pos=2e-5, tol_phi=2e-4,
                  hold_phi=False, free_mask=None, **_):
    """Damped least squares on the 3D point position (``mj_jacSite`` / ``mj_jacBody``) with
    active-set joint limits; the bucket pitch is followed exactly in the position task's
    nullspace, so it never costs position accuracy.
    ``hold_phi``: keep the start pitch as the pitch reference.
    ``free_mask``: joints allowed to move (others stay at their q0 value).
    Returns q, position residual (m), pitch residual (rad).
    """
    target = np.asarray(target, float)
    free0 = np.ones(NQ, bool) if free_mask is None else np.asarray(free_mask, bool)
    q = np.clip(np.asarray(q0, float).copy(), arm.lo, arm.hi)
    phi_goal = pitch_of(q) if hold_phi else phi_t
    Jphi = arm.pitch_mask.astype(float)
    for _ in range(iters):
        Jp = arm.jac(q)
        ep = target - arm.point_pos()
        eph = phi_goal - pitch_of(q)
        if np.linalg.norm(ep) < tol_pos and abs(eph) < tol_phi:
            break
        free = free0.copy()
        for _ in range(NQ):
            Jf = Jp[:, free]
            Jpinv = Jf.T @ np.linalg.inv(Jf @ Jf.T + damping ** 2 * np.eye(3))
            g = Jphi[free]
            dqf = Jpinv @ ep + (np.eye(free.sum()) - Jpinv @ Jf) @ (g * eph / max(g @ g, 1e-9))
            dq = np.zeros(NQ); dq[free] = dqf
            blocked = ((q <= arm.lo + 1e-9) & (dq < 0)) | ((q >= arm.hi - 1e-9) & (dq > 0))
            if not (blocked & free).any():
                break
            free &= ~blocked
        step = np.max(np.abs(dq))
        if step > 0.25:
            dq *= 0.25 / step
        q_new = np.clip(q + dq, arm.lo, arm.hi)
        if np.allclose(q_new, q, atol=1e-10):
            break
        q = q_new
    p = arm.fk(q)
    return q, float(np.linalg.norm(target - p)), float(phi_goal - pitch_of(q))


def solve_ik(arm: Arm, targets, phi_targets=None, q_init=None, w_pitch=0.03, reach_tol=2e-3, seed=0,
             locked: dict | None = None):
    """Solve a sequence of targets, warm-started from the previous solution. Unreachable
    samples keep the least-squares (nearest-reachable) solution; callers flag them.
    ``phi_targets`` (rad) are met wherever the position allows; ``w_pitch`` (m per rad) weighs
    their residual when choosing between seeds. ``locked``: {joint name: value} held fixed
    (e.g. slew / stick on hardware that cannot move them)."""
    free = np.array([n not in (locked or {}) for n in JOINTS])
    lock_idx = [JOINTS.index(n) for n in (locked or {})]
    lock_val = [float(v) for v in (locked or {}).values()]

    def fix(q):
        q = np.array(q, float); q[lock_idx] = lock_val
        return q

    targets = np.atleast_2d(targets)
    n = len(targets)
    rng = np.random.default_rng(seed)
    Q = np.zeros((n, NQ)); pos_res = np.zeros(n); phi_res = np.zeros(n)
    hold = phi_targets is None
    phis = np.zeros(n) if hold else np.atleast_1d(phi_targets)
    mid = 0.5 * (arm.lo + arm.hi)
    for k in range(n):
        if k == 0:
            seeds = [arm.q_rest, mid] + [rng.uniform(arm.lo * .7, arm.hi * .7) for _ in range(12)]
            seeds = analytic_seeds(arm, targets[0], None if hold else phis[0], locked)[:8] + seeds
            if q_init is not None:
                seeds.insert(0, np.asarray(q_init, float))
        else:
            seeds = [Q[k - 1]]
        best = None
        for s in seeds:
            # Without a pitch request each seed keeps its own pitch (hold_phi).
            sol = ik_step_solve(arm, targets[k], phis[k], fix(s), hold_phi=hold, free_mask=free)
            score = sol[1] + (0 if hold else w_pitch * abs(sol[2]))
            if best is None or score < best[0] - 1e-7:   # seeds come ordered by preference
                best = (score, sol)
        if k > 0 and best[1][1] > reach_tol:
            # Warm start stuck: retry from restarts, prefer the one nearest the previous q.
            extra = analytic_seeds(arm, targets[k], None if hold else phis[k], locked)[:8]
            for s in extra + [mid, arm.q_rest] + [rng.uniform(arm.lo * .7, arm.hi * .7) for _ in range(6)]:
                sol = ik_step_solve(arm, targets[k], phis[k], fix(s), hold_phi=hold, free_mask=free)
                score = sol[1] + (0 if hold else w_pitch * abs(sol[2])) + 1e-3 * np.linalg.norm(sol[0] - Q[k - 1])
                if score < best[0] - 1e-4:
                    best = (score, sol)
        Q[k], pos_res[k], phi_res[k] = best[1]
    return Q, pos_res, phi_res


@dataclass
class IKResult:
    q: np.ndarray                    # [slew, boom, stick, bucket] rad
    reachable: bool                  # position, lateral gap and pitch met within tolerance
    position_residual_m: float
    lateral_error_m: float           # radial gap inside the |y0| circle around the slew axis
    pitch_target_deg: float | None
    pitch_residual_deg: float | None
    orientation_error_deg: float | None      # achieved vs requested full rotation
    unachievable_orientation_deg: float | None  # best possible at this position (roll / off-slew yaw)
    point: str = 'tooth_tip'
    achieved_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    achieved_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    rotation_kind: str = 'none'
    joints: list = field(default_factory=lambda: list(JOINTS))

    def as_dict(self) -> dict:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        d['q_deg'] = np.degrees(self.q).tolist()
        return d


def ik(target_pos, target_rotation=None, q0=None, *, arm: Arm | None = None, point: str = 'tooth_tip',
       profile: str = DEFAULT_PROFILE, pitch_deg=None, quat_wxyz=None, rotmat=None, rotvec=None,
       w_pitch: float = 0.03, pos_tol: float = 1e-3, pitch_tol_deg: float = 0.5, lateral_tol: float | None = None,
       locked: dict | None = None) -> IKResult:
    """Solve [slew, boom, stick, bucket] for a point position and bucket orientation.

    ``target_rotation`` (or the keyword forms): bucket pitch in degrees (nose-up positive,
    0 = rest), quaternion wxyz, 3x3 matrix or scipy Rotation of the bucket frame in world. A
    full rotation is projected onto the achievable set ``Rz(slew) R_{-y}(pitch) R_rest`` at the
    slew facing the target; the dropped roll / yaw is reported in
    ``unachievable_orientation_deg``. ``None`` = position only. ``locked`` = {joint: value}
    joints held fixed. ``reachable`` means position within ``pos_tol``, lateral gap within
    ``lateral_tol`` (default ``pos_tol``) and pitch within ``pitch_tol_deg``.
    """
    arm = arm or Arm(profile, point=point)
    target_pos = np.asarray(target_pos, float)
    kind, val = parse_rotation(target_rotation, pitch_deg=pitch_deg, quat_wxyz=quat_wxyz, rotmat=rotmat, rotvec=rotvec)
    _, _, lateral = arm.plane_for(target_pos)
    R_target, unach = None, None
    if kind == 'pitch':
        phi = val
    elif kind == 'matrix':
        R_target = val
        slew = (locked or {}).get('slew', arm.slew_for(target_pos)[0])
        phi, u = best_pitch_for(R_target, slew, arm.R_rest)
        unach = float(np.degrees(u))
    else:
        phi = None
    if phi is not None:
        phi = _wrap_into(phi, *PlanarChain(arm).pitch_range())
    Q, pos_res, phi_res = solve_ik(arm, target_pos[None], None if phi is None else np.array([phi]),
                                   q_init=q0, w_pitch=w_pitch, locked=locked)
    q = Q[0]
    p = arm.fk(q); R = arm.bucket_rot()
    pitch_ok = phi is None or abs(np.degrees(phi_res[0])) <= pitch_tol_deg
    lateral_ok = abs(lateral) <= (pos_tol if lateral_tol is None else lateral_tol)
    return IKResult(q=q, reachable=bool(pos_res[0] <= pos_tol and lateral_ok and pitch_ok),
                    position_residual_m=float(pos_res[0]), lateral_error_m=float(lateral),
                    pitch_target_deg=None if phi is None else float(np.degrees(phi)),
                    pitch_residual_deg=None if phi is None else float(np.degrees(phi_res[0])),
                    orientation_error_deg=None if R_target is None else float(np.degrees(rotation_angle(R, R_target))),
                    unachievable_orientation_deg=unach, point=arm.point, achieved_pos=p, achieved_rot=R,
                    rotation_kind=kind)


def _wrap_into(phi, lo, hi):
    """Shift phi by multiples of 2*pi into [lo, hi] if possible."""
    for k in (0, -1, 1, -2, 2):
        v = phi + 2 * np.pi * k
        if lo <= v <= hi:
            return float(v)
    return float(phi)


BUCKET_ANGLE_WEIGHT = 0.2                         # m per rad: 1 deg of bucket angle ~ 3.5 mm of tip position
SMOOTH_WEIGHT = 0.0                               # m per rad of per-tick joint velocity change (track_ik)
RAMP_S = 0.1                                      # s from rest to full joint speed (track_ik acceleration limit)


def track_ik(arm: Arm, targets, phi_targets, q0, dt: float, w_pitch: float = BUCKET_ANGLE_WEIGHT,
             w_smooth: float = SMOOTH_WEIGHT, ramp_s: float = RAMP_S, locked: dict | None = None,
             iters: int = 8, damping: float = 2e-3) -> np.ndarray:
    """Speed-limited, smoothed IK along a target sequence: each tick starts from the previous
    commanded q and may move each joint only as far as its speed limit allows in ``dt`` (and stays
    within the joint limits). Within that box it minimises

        |tip error|^2 + w_pitch^2 |pitch error|^2 + w_smooth^2 |q - (2 q[k-1] - q[k-2])|^2

    (damped least squares with active-set bounds). The pitch term trades tip position for bucket
    angle at ``w_pitch`` metres per radian when the arm cannot keep up, instead of clipping each
    joint on its own. The smoothness term penalises leaving the constant-velocity prediction, i.e.
    joint acceleration (``w_smooth`` metres per radian of per-tick velocity change); 0 turns it
    off. ``ramp_s`` also limits acceleration hard: a joint needs at least that long to go from
    rest to full speed (a trapezoidal velocity profile), which removes the bang-bang jumps when a
    joint runs into its speed limit; 0 turns it off. ``locked`` joints stay at their q0 value.
    Returns the commanded q (n, 4)."""
    targets = np.atleast_2d(np.asarray(targets, float))
    phis = np.atleast_1d(np.asarray(phi_targets, float))
    free = np.array([n not in (locked or {}) for n in JOINTS])
    Jphi = arm.pitch_mask.astype(float)
    q = np.asarray(q0, float).copy()
    out = np.zeros((len(targets), NQ))
    for k, (p, phi) in enumerate(zip(targets, phis)):
        lo = np.maximum(arm.lo, q - arm.v_neg * dt) if k else arm.lo
        hi = np.minimum(arm.hi, q + arm.v_pos * dt) if k else arm.hi
        # constant-velocity prediction from the last two commands (none for the first two ticks)
        q_pred = 2 * out[k - 1] - out[k - 2] if k >= 2 else None
        if q_pred is not None and ramp_s > 0:
            a_dt2 = np.maximum(arm.v_pos, arm.v_neg) / ramp_s * dt * dt
            alo, ahi = np.maximum(lo, q_pred - a_dt2), np.minimum(hi, q_pred + a_dt2)
            ok = alo <= ahi                         # near a joint limit the ramp may not fit: drop it there
            lo, hi = np.where(ok, alo, lo), np.where(ok, ahi, hi)
        lo = np.where(free, lo, q); hi = np.where(free, hi, q)
        ws = w_smooth if q_pred is not None else 0.0
        for _ in range(iters):
            J = np.vstack([arm.jac(q), w_pitch * Jphi, ws * np.eye(NQ)])
            e = np.r_[p - arm.point_pos(), w_pitch * (phi - pitch_of(q)),
                      ws * (q_pred - q) if ws else np.zeros(NQ)]
            act = free.copy()
            for _ in range(NQ):
                Ja = J[:, act]
                dq = np.zeros(NQ)
                dq[act] = np.linalg.solve(Ja.T @ Ja + damping ** 2 * np.eye(act.sum()), Ja.T @ e)
                blocked = ((q <= lo + 1e-12) & (dq < 0)) | ((q >= hi - 1e-12) & (dq > 0))
                if not (blocked & act).any():
                    break
                act &= ~blocked
            q_new = np.clip(q + dq, lo, hi)
            if np.allclose(q_new, q, atol=1e-9):
                break
            q = q_new
        out[k] = q
    return out


def rate_limit(arm: Arm, Q: np.ndarray, dt: float):
    """Clip per-tick joint motion to the configured speeds. Returns limited q, mask and a report."""
    out = Q.copy()
    dq_raw = np.diff(Q, axis=0) / dt
    ratio = speed_ratio(Q, dt, arm.v_pos, arm.v_neg)
    for k in range(1, len(Q)):
        d = np.clip(Q[k] - out[k - 1], -arm.v_neg * dt, arm.v_pos * dt)
        out[k] = np.clip(out[k - 1] + d, arm.lo, arm.hi)
    limited = np.vstack([np.zeros((1, NQ), bool), ratio > 1 + 1e-9])
    rep = {'violating_samples_per_joint': {n: int(limited[:, i].sum()) for i, n in enumerate(JOINTS)},
           'violating_ticks_any_joint': int(limited.any(1).sum()),
           'max_speed_ratio_per_joint': {n: float(ratio[:, i].max()) if len(ratio) else 0.0 for i, n in enumerate(JOINTS)},
           'ik_peak_speed_rad_s': {n: float(np.abs(dq_raw[:, i]).max()) if len(dq_raw) else 0.0 for i, n in enumerate(JOINTS)},
           'speed_limit_pos_rad_s': dict(zip(JOINTS, arm.v_pos.tolist())),
           'speed_limit_neg_rad_s': dict(zip(JOINTS, arm.v_neg.tolist()))}
    return out, limited, rep


# -------------------------------------------------------------- propagation
def propagate_kinematic(arm: Arm, Q: np.ndarray):
    """Set qpos to each action, mj_forward; returns (tip positions, pitches, bucket body positions)."""
    m, d = arm.model, arm.data
    tips, pitches, bodies = [], [], []
    for q in Q:
        d.qpos[arm.qadr] = q; d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        tips.append(arm.point_pos())
        pitches.append(arm.body_pitch())
        bodies.append(d.xpos[arm.bucket].copy())
    return np.array(tips), np.array(pitches), np.array(bodies)


def propagate_dynamic(arm: Arm, Q: np.ndarray, frameskip: int = FRAMESKIP):
    """Reset at Q[0], then per 20 Hz tick: ctrl = Q[k], step physics frameskip x timestep (as env.py).
    State is sampled after each tick."""
    m, d = arm.model, arm.data
    mujoco.mj_resetData(m, d)
    d.qpos[arm.qadr] = Q[0]; d.ctrl[arm.act] = Q[0]
    mujoco.mj_forward(m, d)
    lim = m.actuator_ctrllimited[arm.act].astype(bool)
    lo = np.where(lim, m.actuator_ctrlrange[arm.act, 0], -np.inf)
    hi = np.where(lim, m.actuator_ctrlrange[arm.act, 1], np.inf)
    ctrl, qpos, qvel, tips, times, ncon, rots = [], [], [], [], [], [], []
    unstable = False
    for q in Q:
        d.ctrl[arm.act] = np.clip(q, lo, hi)
        mujoco.mj_step(m, d, nstep=frameskip)
        if not np.isfinite(d.qpos).all() or np.abs(d.qvel).max() > 20:
            unstable = True
        ctrl.append(d.ctrl[arm.act].copy()); qpos.append(d.qpos[arm.qadr].copy())
        qvel.append(d.qvel[arm.dadr].copy()); tips.append(arm.point_pos()); times.append(d.time)
        ncon.append(int(d.ncon)); rots.append(arm.bucket_rot())
    return {'ctrl': np.array(ctrl), 'qpos': np.array(qpos), 'qvel': np.array(qvel),
            'tip': np.array(tips), 'time': np.array(times), 'unstable': unstable, 'ncon': np.array(ncon),
            'rot': np.array(rots)}
