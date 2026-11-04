"""Shared offscreen-rendering helpers (MuJoCo Renderer + scene overlays + HUD + H.264 writer)."""
from __future__ import annotations

import mujoco
import numpy as np


def add_segment(scene, a, b, rgba, width=0.0016):
    """Append a capsule from a to b to an mjvScene (no-op when the scene is full)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def add_frame(scene, pos, R, length=0.035, width=0.0018, alpha=1.0):
    """Draw an RGB (x, y, z) axes marker for a pose."""
    pos = np.asarray(pos, float); R = np.asarray(R, float)
    for i, c in enumerate([(1, .15, .15), (.15, 1, .15), (.25, .45, 1)]):
        add_segment(scene, pos, pos + length * R[:, i], (*c, alpha), width)


def make_camera(model, name='overview_close', lookat=(0.15, -0.02, 0.15), distance=0.9):
    """'overview_close' = free camera with the XML 'overview' viewing direction, nearer the arm;
    any other name selects that fixed XML camera."""
    cam = mujoco.MjvCamera()
    if name == 'overview_close':
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat; cam.distance = distance; cam.azimuth = 128.5; cam.elevation = -24.8
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = model.camera(name).id
    return cam


def hud(frame, lines, warn_token='UNREACHABLE'):
    """Darkened top band with white text lines (lines containing warn_token in red)."""
    import cv2
    h = 12 + 22 * len(lines)
    frame[:h] = (frame[:h] * 0.35).astype(np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (12, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (255, 120, 120) if warn_token and warn_token in s else (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def open_writer(path, fps=20):
    import imageio.v2 as imageio
    return imageio.get_writer(str(path), fps=fps, codec='libx264', quality=7,
                              pixelformat='yuv420p', macro_block_size=16)
