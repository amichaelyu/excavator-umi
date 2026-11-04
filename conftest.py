"""Tests never read this machine's configs/hub/local.json (policy.data.local_config): every host falls back to its
placeholder default unless a test points $UMI_EXCAVATOR_LOCAL_CONFIG at its own file. Set before any test module
imports the hub, whose Pi defaults are read at import."""
import os
import sys

os.environ['UMI_EXCAVATOR_LOCAL_CONFIG'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'hub',
                                                        'local.tests-absent.json')

# Headless Linux (CI, cluster nodes): MuJoCo's default GLFW backend aborts without a display; render with EGL, as the
# preprocess image does (MUJOCO_GL=egl). Must be set before the first `import mujoco`.
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
