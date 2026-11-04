"""IMU batch smoother of SLAM and KLT factors (factor-graph MAP estimate; smoothing counterpart of an ESKF+RTS).

State per kept camera frame k (IMU body frame, gravity-aligned world, +z up):
    R_k = R_wb (left-perturbed: R_k = Exp(d_k) R0_k), p_k, v_k
Global: gyro bias bg, accel bias ba (constant over the <=12 s episode, with priors),
        log-scale per SLAM segment (only when camera-only SLAM segments are used).

Factors:
  * IMU preintegration between consecutive frames (Forster et al. 2017, first-order bias
    correction), whitened with propagated noise covariance.
  * Two-view visual factors from the KLT front end: relative camera rotation, and (when the
    derotated parallax is large enough) the unit translation direction. These carry no scale;
    metric scale comes only from the accelerometer.
  * Optional SLAM segment factors: relative pose between frames of one SLAM segment, with a
    per-segment scale (camera-only) or scale fixed to 1 (metric ORB-SLAM3 inertial maps).
  * Zero-velocity updates where the IMU shows the bucket at rest.
  * Bias priors (datasheet offsets; gyro bias prior centred on the calibration estimate).
  * Gauge prior: first position = 0, first yaw fixed.

The camera pose is T_w_c = T_w_b * T_b_c with R_b_c from calibration and t_b_c assumed
(default 0, documented; the camera-to-IMU lever arm was not measured).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot

from .recording import G

GRAV = np.array([0.0, 0.0, -G])


def skew(v):
    v = np.asarray(v)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(w):
    return Rot.from_rotvec(np.atleast_2d(w)).as_matrix()


def log_so3(R):
    return Rot.from_matrix(R).as_rotvec()


def right_jacobian(w):
    th = np.linalg.norm(w)
    K = skew(w)
    if th < 1e-6:
        return np.eye(3) - 0.5 * K
    return np.eye(3) - (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * K @ K


@dataclass
class ImuNoise:
    """Continuous-time noise densities.

    Datasheet (ST LSM6DSL, DocID028475 Rev 7, Table 3, typical, high-performance mode):
      gyro rate noise density 4 mdps/sqrt(Hz) = 7.0e-5 rad/s/sqrt(Hz)
      accel noise density at +-4 g: 90 ug/sqrt(Hz) = 8.8e-4 m/s^2/sqrt(Hz)
      gyro zero-rate level +-3 dps (bias prior); accel zero-g offset +-40 mg (bias prior)
    The datasheet has no bias-instability / random-walk figures. The defaults below inflate the
    white-noise densities (x10 gyro, x10 accel) for host I2C timestamp jitter, handheld/soil
    vibration and unmodelled rolling shutter; the inflation is an assumption, not a measurement.
    The bias random walks (ORB-SLAM3 settings only; the smoother keeps biases constant) are
    conventional values. The one default for the smoother and the ORB-SLAM3 settings.
    """
    gyro_nd: float = 7.0e-5 * 10
    accel_nd: float = 8.8e-4 * 10
    gyro_bias_prior_sd: float = np.deg2rad(0.3)   # around the calibrated gyro bias (camera.json)
    accel_bias_prior_sd: float = 0.04 * G
    gyro_walk: float = 1e-4                       # rad/s^2/sqrt(Hz)
    accel_walk: float = 1e-3                      # m/s^3/sqrt(Hz)


@dataclass
class Preint:
    dt: float
    dR: np.ndarray
    dv: np.ndarray
    dp: np.ndarray
    JRg: np.ndarray
    Jvg: np.ndarray
    Jva: np.ndarray
    Jpg: np.ndarray
    Jpa: np.ndarray
    cov: np.ndarray
    bg0: np.ndarray
    ba0: np.ndarray


def preintegrate(t, gyro, acc, ta, tb, bg, ba, noise: ImuNoise) -> Preint:
    inner = (t > ta) & (t < tb)
    nodes = np.r_[ta, t[inner], tb]
    g = np.column_stack([np.interp(nodes, t, gyro[:, i]) for i in range(3)])
    a = np.column_stack([np.interp(nodes, t, acc[:, i]) for i in range(3)])
    dR = np.eye(3); dv = np.zeros(3); dp = np.zeros(3)
    JRg = np.zeros((3, 3)); Jvg = np.zeros((3, 3)); Jva = np.zeros((3, 3)); Jpg = np.zeros((3, 3)); Jpa = np.zeros((3, 3))
    cov = np.zeros((9, 9))
    for j in range(len(nodes) - 1):
        h = nodes[j + 1] - nodes[j]
        if h <= 0:
            continue
        w = 0.5 * (g[j] + g[j + 1]) - bg
        f = 0.5 * (a[j] + a[j + 1]) - ba
        dRj = exp_so3(w * h)[0]
        Jr = right_jacobian(w * h)
        fx = skew(f)
        A = np.eye(9)
        A[0:3, 0:3] = dRj.T
        A[3:6, 0:3] = -dR @ fx * h
        A[6:9, 0:3] = -0.5 * dR @ fx * h * h
        A[6:9, 3:6] = np.eye(3) * h
        Bg = np.zeros((9, 3)); Bg[0:3] = Jr * h
        Ba = np.zeros((9, 3)); Ba[3:6] = dR * h; Ba[6:9] = 0.5 * dR * h * h
        cov = A @ cov @ A.T + Bg @ Bg.T * (noise.gyro_nd**2 / h) + Ba @ Ba.T * (noise.accel_nd**2 / h)
        # bias Jacobians (order matters: use previous dR / JRg)
        Jpa = Jpa + Jva * h - 0.5 * dR * h * h
        Jpg = Jpg + Jvg * h - 0.5 * dR @ fx @ JRg * h * h
        Jva = Jva - dR * h
        Jvg = Jvg - dR @ fx @ JRg * h
        JRg = dRj.T @ JRg - Jr * h
        dp = dp + dv * h + 0.5 * dR @ f * h * h
        dv = dv + dR @ f * h
        dR = dR @ dRj
    return Preint(tb - ta, dR, dv, dp, JRg, Jvg, Jva, Jpg, Jpa, cov + np.eye(9) * 1e-12, bg.copy(), ba.copy())


def integrate_imu_orientation(t, gyro, bg, times):
    """Gyro-only orientation R_b0_b(t) at requested times (body frame of the first time)."""
    Rs = [np.eye(3)]
    for k in range(len(times) - 1):
        inner = (t > times[k]) & (t < times[k + 1])
        nodes = np.r_[times[k], t[inner], times[k + 1]]
        gg = np.column_stack([np.interp(nodes, t, gyro[:, i]) for i in range(3)]) - bg
        R = Rs[-1]
        for j in range(len(nodes) - 1):
            R = R @ exp_so3(0.5 * (gg[j] + gg[j + 1]) * (nodes[j + 1] - nodes[j]))[0]
        Rs.append(R)
    return np.array(Rs)


def detect_stationary(t, gyro, acc, bg, times, win=0.3, gyro_thr_dps=8.0, acc_thr=0.5):
    """Boolean per requested time: IMU window centred there looks like rest."""
    wn = np.linalg.norm(gyro - bg, axis=1)
    an = np.linalg.norm(acc, axis=1)
    out = np.zeros(len(times), bool)
    for k, tk in enumerate(times):
        m = (t >= tk - win / 2) & (t <= tk + win / 2)
        if m.sum() < 20:
            continue
        out[k] = (np.max(wn[m]) < np.deg2rad(gyro_thr_dps)) and (np.std(an[m]) < acc_thr / 3) \
            and abs(np.mean(an[m]) - G) < acc_thr * 2
    return out


@dataclass
class VisualPairs:
    a: np.ndarray
    b: np.ndarray
    R_ab: np.ndarray          # measured camera rotation T_a_b rotation (M,3,3)
    t_dir: np.ndarray         # measured unit translation of b in camera a frame (M,3)
    use_dir: np.ndarray       # bool
    sd_rot: np.ndarray        # rad
    sd_dir: np.ndarray        # unitless (rad)


@dataclass
class SlamPairs:
    a: np.ndarray
    b: np.ndarray
    R_ab: np.ndarray
    t_ab: np.ndarray          # SLAM-frame units
    seg: np.ndarray           # segment index into log-scale parameters
    sd_rot: float = np.deg2rad(1.0)
    sd_t_rel: float = 0.10    # relative translation sd (fraction of predicted displacement)
    sd_t_abs: float = 0.005   # metres


@dataclass
class Problem:
    times: np.ndarray                 # frame times on the IMU clock [s]
    imu_t: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    R_bc: np.ndarray                  # camera-to-body rotation
    t_bc: np.ndarray                  # camera position in body frame (lever arm) [m]
    bg0: np.ndarray
    noise: ImuNoise = field(default_factory=ImuNoise)
    vis: VisualPairs = None
    slam: SlamPairs = None
    n_seg: int = 0
    seg_scale_fixed: bool = False
    seg_scale_init: np.ndarray = None
    stationary: np.ndarray = None
    zupt_sd: float = 0.02
    seg_scale_prior_sd: np.ndarray = None   # prior sd on log-scale per segment (centred on log 1)


class Smoother:
    def __init__(self, P: Problem):
        self.P = P
        N = len(P.times)
        self.N = N
        # Initial orientation: gyro integration, then gravity from the episode-mean specific force.
        Rb0 = integrate_imu_orientation(P.imu_t, P.gyro, P.bg0, P.times)
        fw = []
        for k in range(N - 1):
            m = (P.imu_t >= P.times[k]) & (P.imu_t < P.times[k + 1])
            if m.any():
                fw.append(Rb0[k] @ P.acc[m].mean(axis=0))
        up = np.mean(fw, axis=0); up /= np.linalg.norm(up)
        Rup = self._align(up, np.array([0, 0, 1.0]))
        self.R0 = np.einsum('ij,njk->nik', Rup, Rb0)
        self.bg = P.bg0.copy(); self.ba = np.zeros(3)
        self.p = np.zeros((N, 3)); self.v = np.zeros((N, 3))
        self.logs = np.log(P.seg_scale_init) if P.seg_scale_init is not None else np.zeros(P.n_seg)
        self.preint = None
        self.rot_only = False
        self.robust = True
        self.imu_inflation = 1.0
        self.speed_prior_sd = None      # homotopy: low-speed pseudo-measurement during early stages only
        self.init_mode = 'dead_reckoning'
        self.init_gravity_dir_body0 = Rup.T @ np.array([0, 0, 1.0])

    @staticmethod
    def _align(a, b):
        v = np.cross(a, b); c = np.dot(a, b)
        if np.linalg.norm(v) < 1e-9:
            return np.eye(3)
        return Rot.from_rotvec(v / np.linalg.norm(v) * np.arccos(np.clip(c, -1, 1))).as_matrix()

    # --- parameter packing -------------------------------------------------------------
    def _unpack(self, x):
        N = self.N
        d = x[:3 * N].reshape(N, 3); p = x[3 * N:6 * N].reshape(N, 3); v = x[6 * N:9 * N].reshape(N, 3)
        bg = x[9 * N:9 * N + 3]; ba = x[9 * N + 3:9 * N + 6]; logs = x[9 * N + 6:]
        R = np.einsum('nij,njk->nik', exp_so3(d), self.R0)
        return R, p, v, bg, ba, logs

    def _pack(self):
        return np.r_[np.zeros(3 * self.N), self.p.ravel(), self.v.ravel(), self.bg, self.ba, self.logs]

    def _build_preint(self):
        P = self.P
        self.preint = [preintegrate(P.imu_t, P.gyro, P.acc, P.times[k], P.times[k + 1], self.bg, self.ba, P.noise)
                       for k in range(self.N - 1)]
        pi = self.preint
        self.pi_dt = np.array([q.dt for q in pi])
        self.pi_dR = np.array([q.dR for q in pi]); self.pi_dv = np.array([q.dv for q in pi]); self.pi_dp = np.array([q.dp for q in pi])
        self.pi_JRg = np.array([q.JRg for q in pi]); self.pi_Jvg = np.array([q.Jvg for q in pi]); self.pi_Jva = np.array([q.Jva for q in pi])
        self.pi_Jpg = np.array([q.Jpg for q in pi]); self.pi_Jpa = np.array([q.Jpa for q in pi])
        self.pi_L = np.array([np.linalg.cholesky(np.linalg.inv(q.cov)).T for q in pi])
        self.pi_b0 = (self.bg.copy(), self.ba.copy())

    # --- residuals -----------------------------------------------------------------------
    def residuals(self, x, parts=False):
        P = self.P; N = self.N
        R, p, v, bg, ba, logs = self._unpack(x)
        out = {}
        # IMU
        dbg = bg - self.pi_b0[0]; dba = ba - self.pi_b0[1]
        Ri, Rj = R[:-1], R[1:]
        dt = self.pi_dt[:, None]
        dRc = np.einsum('nij,njk->nik', self.pi_dR, exp_so3(np.einsum('nij,j->ni', self.pi_JRg, dbg)))
        dvc = self.pi_dv + np.einsum('nij,j->ni', self.pi_Jvg, dbg) + np.einsum('nij,j->ni', self.pi_Jva, dba)
        dpc = self.pi_dp + np.einsum('nij,j->ni', self.pi_Jpg, dbg) + np.einsum('nij,j->ni', self.pi_Jpa, dba)
        rR = log_so3(np.einsum('nji,nkj,nkl->nil', dRc, Ri, Rj))
        rv = np.einsum('nji,nj->ni', Ri, v[1:] - v[:-1] - GRAV * dt) - dvc
        rp = np.einsum('nji,nj->ni', Ri, p[1:] - p[:-1] - v[:-1] * dt - 0.5 * GRAV * dt**2) - dpc
        if self.rot_only:
            rv = rv * 0; rp = rp * 0
        r = np.concatenate([rR, rv, rp], axis=1)
        out['imu'] = np.einsum('nij,nj->ni', self.pi_L, r).ravel() / self.imu_inflation
        # camera poses
        Rc = np.einsum('nij,jk->nik', R, P.R_bc)
        c = p + np.einsum('nij,j->ni', R, P.t_bc)
        if P.vis is not None and len(P.vis.a):
            V = P.vis
            Rpred = np.einsum('nji,njk->nik', Rc[V.a], Rc[V.b])
            er = log_so3(np.einsum('nji,njk->nik', V.R_ab, Rpred)) / V.sd_rot[:, None]
            out['vis_rot'] = self._rb(er, 3.0).ravel()
            d = np.einsum('nji,nj->ni', Rc[V.a], c[V.b] - c[V.a])
            dn = d / np.sqrt(np.sum(d**2, axis=1, keepdims=True) + 0.002**2)
            ed = (dn - V.t_dir) / V.sd_dir[:, None]
            ed = np.where(V.use_dir[:, None] & (not self.rot_only), ed, 0.0)
            out['vis_dir'] = self._rb(ed, 2.0).ravel()
        if P.slam is not None and len(P.slam.a):
            S = P.slam
            Rpred = np.einsum('nji,njk->nik', Rc[S.a], Rc[S.b])
            er = log_so3(np.einsum('nji,njk->nik', S.R_ab, Rpred)) / S.sd_rot
            out['slam_rot'] = self._rb(er, 3.0).ravel()
            d = np.einsum('nji,nj->ni', Rc[S.a], c[S.b] - c[S.a])
            s = np.exp(logs[S.seg])[:, None]
            sd = S.sd_t_abs + S.sd_t_rel * np.linalg.norm(s * S.t_ab, axis=1, keepdims=True)
            et = (d - s * S.t_ab) / sd
            out['slam_t'] = self._rb(et * (not self.rot_only), 3.0).ravel()
        if P.stationary is not None and P.stationary.any():
            out['zupt'] = (v[P.stationary] / P.zupt_sd).ravel() * (not self.rot_only)
        sp = self.speed_prior_sd
        out['speed_prior'] = (v / sp).ravel() if sp else np.zeros(3 * N)
        n = P.noise
        out['bias'] = np.r_[(bg - P.bg0) / n.gyro_bias_prior_sd, ba / n.accel_bias_prior_sd]
        if P.n_seg:
            sd = P.seg_scale_prior_sd if P.seg_scale_prior_sd is not None else np.full(P.n_seg, 3.0)
            out['scale'] = logs / (0.01 if P.seg_scale_fixed else sd)
        out['gauge'] = np.r_[p[0] / 1e-4, x[2] / 1e-4]
        if parts:
            return out
        return np.concatenate(list(out.values()))

    def _rb(self, r, c):
        return _robust(r, c) if self.robust else r

    def _sparsity(self, x0):
        """Jacobian sparsity: each block touches its two frames' 9-parameter states + globals."""
        P = self.P; N = self.N
        blocks = []  # (n_rows, frames list, uses_bias, seg or None)
        for k in range(N - 1):
            blocks.append((9, [k, k + 1], True, None))
        if P.vis is not None and len(P.vis.a):
            for a, b in zip(P.vis.a, P.vis.b):
                blocks.append((3, [a, b], False, None))
            rows_dir = [(3, [a, b], False, None) for a, b in zip(P.vis.a, P.vis.b)]
            blocks += rows_dir
        if P.slam is not None and len(P.slam.a):
            for a, b in zip(P.slam.a, P.slam.b):
                blocks.append((3, [a, b], False, None))
            for a, b, s in zip(P.slam.a, P.slam.b, P.slam.seg):
                blocks.append((3, [a, b], False, int(s)))
        if P.stationary is not None:
            for k in np.flatnonzero(P.stationary):
                blocks.append((3, [k], False, None))
        for k in range(N):
            blocks.append((3, [k], False, None))   # speed prior (zero unless the homotopy is active)
        blocks.append((6, [], True, None))
        if P.n_seg:
            blocks.append((P.n_seg, [], False, 'all'))
        blocks.append((4, [0], False, None))
        nrow = sum(b[0] for b in blocks)
        J = lil_matrix((nrow, len(x0)), dtype=int)
        r = 0
        for n, frames, bias, seg in blocks:
            for f in frames:
                for off in (0, 3 * N, 6 * N):
                    J[r:r + n, off + 3 * f: off + 3 * f + 3] = 1
            if bias:
                J[r:r + n, 9 * N:9 * N + 6] = 1
            if seg == 'all':
                J[r:r + n, 9 * N + 6:] = 1
            elif seg is not None:
                J[r:r + n, 9 * N + 6 + seg] = 1
            r += n
        return J.tocsr()

    def _lm(self, x0, free, iters=30, verbose=0, lam=1e-3):
        """Sparse Levenberg-Marquardt over the parameter subset `free` (bool mask)."""
        from scipy.optimize._numdiff import approx_derivative
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import diags
        x = x0.copy()
        sub = self.sparsity[:, free]

        def f_free(z):
            xx = x.copy(); xx[free] = z
            return self.residuals(xx)
        r = f_free(x[free]); cost = 0.5 * r @ r
        for it in range(iters):
            J = approx_derivative(f_free, x[free], method='2-point', sparsity=sub, rel_step=None,
                                  abs_step=1e-7, f0=r)
            J = J.tocsr()
            H = (J.T @ J).tocsc(); g = J.T @ r
            dH = H.diagonal()
            improved = False
            for _ in range(10):
                A = H + diags(lam * np.maximum(dH, 1e-9))
                dx = -spsolve(A.tocsc(), g)
                z = x[free] + dx
                r2 = f_free(z); c2 = 0.5 * r2 @ r2
                if np.isfinite(c2) and c2 < cost:
                    x[free] = z; rel = (cost - c2) / cost; r, cost = r2, c2
                    lam = max(lam / 3, 1e-7); improved = True
                    break
                lam *= 5
            if verbose:
                print(f'    LM {it}: cost {cost:.4g} lam {lam:.1e}', flush=True)
            if not improved or rel < 1e-7:
                break
        return x

    def _relinearize(self, x):
        R, p, v, bg, ba, logs = self._unpack(x)
        self.R0, self.p, self.v, self.bg, self.ba, self.logs = R, p, v, bg, ba, logs
        if np.linalg.norm(bg - self.pi_b0[0]) > np.deg2rad(0.1) or np.linalg.norm(ba - self.pi_b0[1]) > 0.02:
            self._build_preint()

    def _init_translation(self):
        """IMU dead reckoning from v0 = 0 with the current rotations (initial guess only)."""
        P = self.P
        if self.init_mode == 'zero':
            self.p[:] = 0; self.v[:] = 0
            return
        v = np.zeros(3); p = np.zeros(3)
        self.p[0] = p; self.v[0] = v
        for k in range(self.N - 1):
            q = self.preint[k]
            Ri = self.R0[k]
            p = p + v * q.dt + 0.5 * GRAV * q.dt**2 + Ri @ q.dp
            v = v + GRAV * q.dt + Ri @ q.dv
            self.p[k + 1] = p; self.v[k + 1] = v
        # remove linear velocity drift so the guess stays bounded (vision/IMU refine it)
        T = P.times - P.times[0]
        drift = self.v[-1] / max(T[-1], 1e-3)
        self.v -= drift[None, :] * T[:, None]
        self.p -= 0.5 * drift[None, :] * T[:, None] ** 2

    def warm_start(self, other: 'Smoother'):
        """Initialise from another solution of the same episode (e.g. for sensitivity runs)."""
        self.R0, self.p, self.v = other.R0.copy(), other.p.copy(), other.v.copy()
        self.bg, self.ba = other.bg.copy(), other.ba.copy()
        if len(other.logs) == len(self.logs):
            self.logs = other.logs.copy()
        self._warm = True
        return self

    def solve(self, outer=2, continuation=(4.0, 2.0, 1.0), verbose=0):
        """Staged Gauss-Newton/LM with graduated IMU-noise continuation.

        A: rotations + gyro bias from rotation-only residuals. B: translation, velocity, accel bias,
        segment scales with rotations fixed (L2 then robust). C: joint refinement while the IMU noise
        inflation is reduced to 1 (graduated optimisation avoids poor local minima in scale).
        """
        self._build_preint()
        x0 = self._pack()
        self.sparsity = self._sparsity(x0)
        N = self.N
        idx = np.arange(len(x0))
        rot = idx < 3 * N
        pos = (idx >= 3 * N) & (idx < 6 * N); vel = (idx >= 6 * N) & (idx < 9 * N)
        bgm = (idx >= 9 * N) & (idx < 9 * N + 3); bam = (idx >= 9 * N + 3) & (idx < 9 * N + 6)
        scl = idx >= 9 * N + 6
        allp = np.ones(len(x0), bool)
        if not getattr(self, '_warm', False):
            self.rot_only = True
            x = self._lm(self._pack(), rot | bgm, iters=20, verbose=verbose)
            self.rot_only = False
            self._relinearize(x); self._init_translation()
            self.imu_inflation = continuation[0]
            self.robust = False
            x = self._lm(self._pack(), pos | vel | bam | scl, iters=30, verbose=verbose)
            self.robust = True
            x = self._lm(x, pos | vel | bam | scl, iters=30, verbose=verbose)
            self._relinearize(x)
            steps = continuation
        else:
            steps = (1.0,)
        for f in steps:
            self.imu_inflation = f
            for it in range(outer if f == 1.0 else 1):
                x = self._lm(self._pack(), allp, iters=25, verbose=verbose)
                self._relinearize(x)
        if self.speed_prior_sd:
            # Release the homotopy prior and re-converge on the true objective.
            self.speed_prior_sd = None
            for it in range(outer):
                x = self._lm(self._pack(), allp, iters=25, verbose=verbose)
                self._relinearize(x)
        self.imu_inflation = 1.0
        self.x = self._pack()
        self.cost = 0.5 * float(np.sum(self.residuals(self.x) ** 2))
        return self

    # --- outputs -------------------------------------------------------------------------
    def camera_poses(self):
        Rc = np.einsum('nij,jk->nik', self.R0, self.P.R_bc)
        c = self.p + np.einsum('nij,j->ni', self.R0, self.P.t_bc)
        return Rc, c

    def imu_rate(self):
        """Body states at IMU sample times (between first and last frame), IMU-propagated from
        each smoothed frame state with the endpoint mismatch distributed linearly (continuous)."""
        P = self.P
        ts, Rs, ps, frame_k = [], [], [], []
        for k in range(self.N - 1):
            ta, tb = P.times[k], P.times[k + 1]
            m = (P.imu_t >= ta) & (P.imu_t < tb)
            nodes = np.r_[ta, P.imu_t[m], tb]
            g = np.column_stack([np.interp(nodes, P.imu_t, P.gyro[:, i]) for i in range(3)]) - self.bg
            a = np.column_stack([np.interp(nodes, P.imu_t, P.acc[:, i]) for i in range(3)]) - self.ba
            R = self.R0[k].copy(); p = self.p[k].copy(); v = self.v[k].copy()
            Rl, pl = [R], [p]
            for j in range(len(nodes) - 1):
                h = nodes[j + 1] - nodes[j]
                acc_w = R @ (0.5 * (a[j] + a[j + 1])) + GRAV
                p = p + v * h + 0.5 * acc_w * h * h
                v = v + acc_w * h
                R = R @ exp_so3(0.5 * (g[j] + g[j + 1]) * h)[0]
                Rl.append(R); pl.append(p)
            # Rotation: gyro propagation with the endpoint mismatch distributed linearly.
            # Position: cubic Hermite through the smoothed frame positions/velocities (C1-continuous).
            eR = log_so3(self.R0[k + 1] @ Rl[-1].T)
            h = tb - ta
            p0, p1, v0, v1 = self.p[k], self.p[k + 1], self.v[k], self.v[k + 1]
            for j in range(len(nodes) - 1):
                s = (nodes[j] - ta) / h
                h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
                ts.append(nodes[j]); Rs.append(exp_so3(eR * s)[0] @ Rl[j])
                ps.append(h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1); frame_k.append(k)
        ts.append(P.times[-1]); Rs.append(self.R0[-1]); ps.append(self.p[-1]); frame_k.append(self.N - 1)
        Rs = np.array(Rs); ps = np.array(ps)
        Rc = np.einsum('nij,jk->nik', Rs, P.R_bc)
        c = ps + np.einsum('nij,j->ni', Rs, P.t_bc)
        return np.array(ts), Rc, c, np.array(frame_k)

    def diagnostics(self):
        parts = self.residuals(self.x, parts=True)
        P = self.P
        out = {k: float(np.sqrt(np.mean(v**2))) if len(v) else None for k, v in parts.items()}
        # gravity alignment residual at stationary frames: measured specific force vs predicted
        ang = []
        if P.stationary is not None:
            for k in np.flatnonzero(P.stationary):
                m = np.abs(P.imu_t - P.times[k]) < 0.05
                if m.any():
                    f = P.acc[m].mean(axis=0) - self.ba
                    pred = self.R0[k].T @ -GRAV
                    ang.append(np.degrees(np.arccos(np.clip(f @ pred / np.linalg.norm(f) / np.linalg.norm(pred), -1, 1))))
        # Gravity/tilt consistency over all frames: angle between the measured specific force (bias-corrected,
        # averaged over +-50 ms) and the smoother's predicted one R^T (a - g), a from central velocity differences.
        ang_all = []
        for k in range(1, self.N - 1):
            m = np.abs(P.imu_t - P.times[k]) < 0.05
            if not m.any():
                continue
            a = (self.v[k + 1] - self.v[k - 1]) / (P.times[k + 1] - P.times[k - 1])
            pred = self.R0[k].T @ (a - GRAV)
            f = P.acc[m].mean(axis=0) - self.ba
            ang_all.append(np.degrees(np.arccos(np.clip(f @ pred / np.linalg.norm(f) / np.linalg.norm(pred), -1, 1))))
        return {'whitened_rms': out,
                'gravity_residual_deg_stationary_mean': float(np.mean(ang)) if ang else None,
                'gravity_residual_deg_stationary_max': float(np.max(ang)) if ang else None,
                'gravity_residual_deg_all_frames_median': float(np.median(ang_all)) if ang_all else None,
                'gravity_residual_deg_all_frames_p95': float(np.percentile(ang_all, 95)) if ang_all else None,
                'stationary_frames': int(P.stationary.sum()) if P.stationary is not None else 0,
                'gyro_bias_dps': np.degrees(self.bg).tolist(), 'accel_bias_mps2': self.ba.tolist(),
                'segment_scales': np.exp(self.logs).tolist()}


def _robust(r, c):
    """Cauchy-robustify residual blocks (rows) while keeping least-squares form."""
    s = np.sum(r**2, axis=1, keepdims=True)
    rho = c**2 * np.log1p(s / c**2)
    w = np.sqrt(np.where(s > 1e-12, rho / np.maximum(s, 1e-12), 1.0))
    return r * w
