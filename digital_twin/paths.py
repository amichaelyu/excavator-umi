"""Filesystem locations of the digital twin, shared by every subpackage.

The scan-based digital twin of the E568 RC excavator: the MuJoCo model and the tooling that runs π0.5 on the real
machine. hub/ and policy/ both import it; it imports neither (only the leaf layers policy.configs and policy.data,
see policy/tests/test_layering.py). Layers (each imports only from the ones above it):

    paths      this module
    sim        the MuJoCo twin core: kinematics, model build (models/, assets/links), replay / IK planning,
               self-mask, pin map and pulse model, joint config
    hardware   the real machine: GPIO bridge + pulse execution, joint-state estimation, calibration
    plan       offline: target poses -> joint plans (goto_pose)
    policy     the CLOAK π0.5 client, checkpoint registry and the closed-loop runner

The twin's configs (robot incl. camera mount, and self-mask JSON) are in configs/hardware/ (CONFIG); its models/ and assets/ in
digital_twin/sim/ (SIM, MODELS). HUB_CONFIG (configs/hub/) holds the hub's registry (apps.json, local.json) and
local_checkpoints.json (the inference dashboard's Lambda endpoints).

The hardware map (pin names, pins, signs, homing, welded joints, calibration) is configs/hardware/hardware_map.json: one file the
hub writes when it calibrates (hardware.calibrate) and the sim and every ``python -m policy`` stage read. It is per
machine (gitignored), like configs/hardware/camera-calibration/camera.json and configs/hardware/robot_full4_calibrated.json; each is created from its
tracked ``*.example.json`` the first time it is read (policy.data.local_config.ensure_local).
:func:`hardware_map` names the file in use:

1. ``$UMI_EXCAVATOR_HARDWARE_MAP`` when set;
2. else :data:`DEFAULT_HARDWARE_MAP` (configs/hardware/hardware_map.json, created from the example when missing).

:func:`set_hardware_map` sets that variable, so the subprocesses a process starts read the same map. Cached
reads (``kinematics.locked_joints``) are keyed by the path; after rewriting the file in place call
``kinematics.reload()``.
"""
import os
from pathlib import Path

from policy.data.local_config import ensure_local

TWIN = Path(__file__).resolve().parent          # digital_twin/
REPO = TWIN.parent
SIM = TWIN / 'sim'                              # digital_twin/sim/: models/, assets/, outputs/
CONFIG = REPO / 'configs' / 'hardware'            # configs/hardware/: robot, self-mask, hardware map, camera
MODELS = SIM / 'models'
ARTIFACTS = REPO / 'artifacts'
DEFAULT_HARDWARE_MAP = CONFIG / 'hardware_map.json'
HARDWARE_MAP_ENV = 'UMI_EXCAVATOR_HARDWARE_MAP'
HUB_CONFIG = REPO / 'configs' / 'hub'           # configs/hub/: hub registry and checkpoint endpoints
LOCAL_CHECKPOINTS = HUB_CONFIG / 'local_checkpoints.json'


def hardware_map() -> Path:
    """The hardware map in use: ``$UMI_EXCAVATOR_HARDWARE_MAP``, else configs/hardware/hardware_map.json (created from
    configs/hardware/hardware_map.example.json when missing)."""
    path = os.environ.get(HARDWARE_MAP_ENV)
    return Path(path) if path else ensure_local(DEFAULT_HARDWARE_MAP)


def set_hardware_map(path) -> Path:
    """Read ``path`` as the hardware map from now on, in this process and the subprocesses it starts (None:
    back to configs/hardware/hardware_map.json). Returns the map now in use."""
    if path is None:
        os.environ.pop(HARDWARE_MAP_ENV, None)
    else:
        os.environ[HARDWARE_MAP_ENV] = str(Path(path).resolve())
    return hardware_map()


HARDWARE_MAP = hardware_map()                   # at import: configs/hardware/hardware_map.json unless $UMI_EXCAVATOR_HARDWARE_MAP
