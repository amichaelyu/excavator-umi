"""Output writers: trajectory CSVs, plots and overlay video."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from policy.data.camera import FRAME_SHAPE

HEADER = 't_s,px,py,pz,qw,qx,qy,qz,status\n'


def output_frame(R_wc0: np.ndarray, p_wc0: np.ndarray):
    """World frame for outputs: gravity-aligned (+z up), origin at the first pose, yaw chosen so the
    first camera's optical axis (or, if it points nearly vertically, its image-up axis) projects
    onto +x. Returns (R_out_w, p0) with x_out = R_out_w (x_w - p0)."""
    z = R_wc0[:, 2]
    h = np.array([z[0], z[1], 0.0])
    if np.linalg.norm(h) < 0.2:
        up_img = -R_wc0[:, 1]
        h = np.array([up_img[0], up_img[1], 0.0])
    yaw = np.arctan2(h[1], h[0])
    return Rot.from_euler('z', -yaw).as_matrix(), p_wc0.copy()


def write_csv(path: Path, t, R_wc, p_wc, status):
    # Preserve unavailable poses as NaN rows, keeping the original video timeline.
    R_wc = np.asarray(R_wc)
    valid = np.isfinite(R_wc).all(axis=(1, 2))
    q = np.full((len(t), 4), np.nan)
    if valid.any():
        q[valid] = Rot.from_matrix(R_wc[valid]).as_quat()  # x y z w
    q = q * np.sign(q[:, 3:4] + 1e-12)
    with open(path, 'w') as f:
        f.write(HEADER)
        for i in range(len(t)):
            f.write(f'{t[i]:.6f},{p_wc[i,0]:.5f},{p_wc[i,1]:.5f},{p_wc[i,2]:.5f},'
                    f'{q[i,3]:.6f},{q[i,0]:.6f},{q[i,1]:.6f},{q[i,2]:.6f},{status[i]}\n')


COLORS = {'tracked': '#2a7de1', 'imu_only': '#e8590c', 'interpolated': '#868e96'}


def plot(path: Path, t, p, status, title, R_wc=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color='#adb5bd', lw=1)
    for s, c in COLORS.items():
        m = np.array([x == s for x in status])
        if m.any():
            ax.scatter(p[m, 0], p[m, 1], p[m, 2], s=6, color=c, label=s)
    finite = np.isfinite(p).all(axis=1)
    if R_wc is not None:
        for i in range(0, len(t), max(1, len(t) // 12)):
            if not finite[i] or not np.isfinite(R_wc[i]).all():
                continue
            a = R_wc[i][:, 2] * 0.05
            ax.quiver(p[i, 0], p[i, 1], p[i, 2], a[0], a[1], a[2], color='k', lw=0.8)
    valid_p = p[finite]
    if len(valid_p):
        ax.scatter(*valid_p[0], color='g', s=40, marker='o', label='start')
        ax.scatter(*valid_p[-1], color='r', s=40, marker='s', label='end')
        lim = np.ptp(valid_p, axis=0).max() / 2 + 0.02
        c = (valid_p.max(0) + valid_p.min(0)) / 2
    else:
        lim, c = 0.5, np.zeros(3)
        ax.text2D(0.2, 0.5, 'No initialized pose estimates', transform=ax.transAxes)
    ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[1] - lim, c[1] + lim); ax.set_zlim(c[2] - lim, c[2] + lim)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z (up) [m]')
    if len(valid_p):
        ax.legend(loc='upper left', fontsize=8)
    ax.set_title('camera position (black: optical axis)')
    for k, name in enumerate('xyz'):
        a2 = fig.add_subplot(3, 2, 2 * (k + 1))
        a2.plot(t, p[:, k], color='#495057', lw=1)
        for s, col in COLORS.items():
            m = np.array([x == s for x in status])
            if m.any():
                a2.scatter(t[m], p[m, k], s=4, color=col)
        a2.set_ylabel(f'{name} [m]'); a2.grid(alpha=0.3)
        if len(t) > 1:
            a2.set_xlim(t[0], t[-1])
        if not finite.all():
            a2.fill_between(t, 0, 1, where=~finite, transform=a2.get_xaxis_transform(),
                            color='#e9ecef', alpha=0.8, label='pose unavailable')
        if k == 2:
            a2.set_xlabel('t [s]')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def overlay_video(path: Path, rec, tr, status, orb_state, t_s, p, fps=20, width=640,
                  tracker_name='ORB-SLAM3(VI)', feature_name='KLT'):
    tmp = path.with_suffix('.tmp.mp4')
    frame_h, frame_w = FRAME_SHAPE
    h = int(round(width * frame_h / frame_w))
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, h))
    s = width / frame_w
    names = {-1: 'n/a', 0: 'no img', 1: 'not init', 2: 'OK', 3: 'recently lost', 4: 'lost', 5: 'OK_KLT'}
    if not vw.isOpened():
        raise RuntimeError(f'Could not open video writer: {tmp}')
    for k, i in enumerate(tr.frame_idx):
        img = cv2.resize(rec.image(i, gray=False), (width, h), interpolation=cv2.INTER_AREA)
        ids, pts = tr.obs[k]
        for q in pts:
            cv2.circle(img, (int(q[0] * s), int(q[1] * s)), 2, (0, 255, 0), -1)
        col = {'tracked': (225, 125, 42), 'imu_only': (12, 89, 232)}.get(status[k], (150, 150, 150))
        cv2.rectangle(img, (0, 0), (width, 44), (0, 0, 0), -1)
        cv2.putText(img, f't={t_s[k]:5.2f}s  {status[k]}  {feature_name}={len(pts)}  {tracker_name}={names.get(int(orb_state[k]), "?")}',
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        pose_text = (f'p=({p[k,0]:+.3f},{p[k,1]:+.3f},{p[k,2]:+.3f}) m'
                     if np.isfinite(p[k]).all() else 'Pose unavailable: estimator has not initialized')
        cv2.putText(img, pose_text, (6, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    ff = shutil.which('ffmpeg')
    if ff:
        subprocess.run([ff, '-y', '-loglevel', 'error', '-i', str(tmp), '-c:v', 'libx264', '-crf', '30',
                        '-preset', 'veryfast', '-pix_fmt', 'yuv420p', str(path)], check=True)
        tmp.unlink()
    else:
        tmp.rename(path)
