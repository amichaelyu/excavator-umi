"""Interactive MuJoCo viewer playback of precomputed arm trajectories (shared by the replay and
goto_pose entry points).

On macOS ``mujoco.viewer.launch_passive`` only works under the ``mjpython`` launcher; when
started from plain ``python`` we print the equivalent ``mjpython`` command instead.
"""
from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path

import numpy as np

from .viz import add_frame, add_segment


def needs_mjpython() -> bool:
    if sys.platform != 'darwin':
        return False
    import mujoco.viewer as mv
    return not isinstance(getattr(mv, '_MJPYTHON', None), getattr(mv, '_MjPythonBase', ()))


def mjpython_command(module: str, argv=None) -> str:
    """Exact command to rerun this invocation under mjpython (venv launcher if present)."""
    exe = Path(sys.executable).with_name('mjpython')
    launcher = str(exe) if exe.exists() else 'mjpython'
    args = list(sys.argv[1:] if argv is None else argv)
    return ' '.join([shlex.quote(launcher), '-m', module] + [shlex.quote(a) for a in args])


def guard(module: str, argv=None) -> bool:
    """True if the viewer can open here; otherwise print the mjpython command and return False."""
    if needs_mjpython():
        print('The interactive MuJoCo viewer needs the mjpython launcher on macOS. Run:\n  '
              + mjpython_command(module, argv), file=sys.stderr)
        return False
    return True


def play(model, qadr, t, qpos, *, target_path=None, target_frames=None, active_target=None,
         trail=None, time_scale=1.0, loop=True, max_seconds=None, title='', mocap=None):
    """Play qpos frames (N, len(qadr)) at times t (N,) in a passive viewer, in real time x time_scale.

    target_path: (M, 3) polyline drawn green; target_frames: list of (pos, R) axes markers;
    active_target: (N,) index of the highlighted frame per sample; trail: (N, 3) achieved tip
    drawn as a trailing orange line; mocap: (N, 3) positions for the mocap 'target' body.
    With ``loop`` the motion repeats until the window is closed; otherwise it plays once and
    holds the last frame until the window is closed;
    ``max_seconds`` closes it automatically (smoke tests).
    """
    import mujoco.viewer
    data = mujoco.MjData(model)
    t = np.asarray(t, float) - float(t[0])
    started = time.monotonic()
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = [.15, -.02, .15]; v.cam.distance = .9; v.cam.azimuth = 128.5; v.cam.elevation = -24.8
        if title:
            print(f'[viewer] {title}: {len(t)} frames, {t[-1] * time_scale:.1f} s per pass'
                  + (' (looping; close the window to exit)' if loop else ''))
        while v.is_running():
            t0 = time.monotonic()
            for k in range(len(t)):
                if not v.is_running() or (max_seconds and time.monotonic() - started > max_seconds):
                    return
                data.qpos[qadr] = qpos[k]
                if mocap is not None and model.nmocap:
                    data.mocap_pos[0] = mocap[k]
                mujoco.mj_forward(model, data)
                with v.lock():
                    scn = v.user_scn
                    scn.ngeom = 0
                    if target_path is not None:
                        for j in range(len(target_path) - 1):
                            add_segment(scn, target_path[j], target_path[j + 1], (.2, 1, .45, .8), .0012)
                    for i, (p, R) in enumerate(target_frames or []):
                        on = active_target is not None and i == active_target[k]
                        add_frame(scn, p, R, .04 if on else .025, .002 if on else .0012, 1. if on else .45)
                    if trail is not None:
                        for j in range(max(0, k - 80), k):
                            add_segment(scn, trail[j], trail[j + 1], (1, .6, .1, 1), .0015)
                v.sync()
                wait = t0 + t[k] * time_scale - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            if not loop:
                # hold the last frame until the window is closed
                while v.is_running() and not (max_seconds and time.monotonic() - started > max_seconds):
                    v.sync(); time.sleep(0.05)
                return
            time.sleep(0.5)
