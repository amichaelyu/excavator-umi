"""The ESP32 pins of the hardware map: each joint's two pins with their names.

Every GPIO number appears once in configs/hardware/hardware_map.json, on the joint it drives::

    "boom": {"positive": {"pin": 33, "label": "boom up"}, "negative": {"pin": 32, "label": "boom down"}, "sign": 1}

Everything else refers to a pin by joint and direction (``positive`` / ``negative``, or +1 / -1): the homing
pins (``homing.joints.<j>.home_direction``), the calibration checks (``checks.joints.<j>.drives``) and the
stick coupling channels (``coupling.channels``). The helpers here are the only readers of the raw pin keys:
:func:`pin` / :func:`entry_pin` (joint + direction -> GPIO), :func:`labels` / :func:`label` (GPIO -> name).
The GPIO page renames pins through :func:`save_pin_labels`, which rewrites only the joints' labels and
leaves the rest of the map (calibration, homing) as it is on disk.

:func:`read` loads a map.
"""
from __future__ import annotations

import json
from pathlib import Path

from policy.data.local_config import check_writable

from ..paths import hardware_map

DIRECTIONS = ('positive', 'negative')


def side(direction) -> str:
    """``'positive'`` / ``'negative'`` from +1 / -1 (any sign) or the name itself."""
    if isinstance(direction, str):
        if direction not in DIRECTIONS:
            raise ValueError(f'direction must be positive/negative, not {direction!r}')
        return direction
    return 'positive' if direction > 0 else 'negative'


def joint_entry(hw: dict, joint: str) -> dict | None:
    """A joint's map entry in ``joints`` (None when absent)."""
    return (hw.get('joints') or {}).get(joint)


def entry_pin(entry: dict, direction) -> int:
    """The GPIO of a joint entry's ``direction`` pin."""
    return int(entry[side(direction)]['pin'])


def pin(hw: dict, joint: str, direction) -> int:
    """The GPIO that drives ``joint`` in ``direction`` (+1 / -1 or positive / negative)."""
    entry = joint_entry(hw, joint)
    if entry is None:
        raise KeyError(f'hardware map: no joint {joint!r}')
    return entry_pin(entry, direction)


def joint_drives(hw: dict):
    """``(joint, direction, {'pin', 'label'})`` for every pin of every joint."""
    for name, j in (hw.get('joints') or {}).items():
        if isinstance(j, dict):
            for d in DIRECTIONS:
                if isinstance(j.get(d), dict):
                    yield name, d, j[d]


def labels(hw: dict) -> dict[int, str]:
    """``{gpio: name}`` of every joint pin of an already loaded map (by GPIO)."""
    return dict(sorted((int(p['pin']), str(p.get('label') or '')) for _, _, p in joint_drives(hw) if 'pin' in p))


def label(hw: dict, gpio) -> str:
    """The name of GPIO ``gpio`` in ``hw`` (empty string when no joint uses it)."""
    return labels(hw).get(int(gpio), '')


def read(path=None) -> dict:
    """The hardware map at ``path`` (default: the one in use)."""
    return json.loads(Path(path or hardware_map()).read_text())


# ------------------------------------------------------------------ the GPIO page
def pin_labels(path=None) -> dict[int, str] | None:
    """``{gpio: name}`` from the hardware map file in use, or None when it can't be read."""
    try:
        return labels(read(path))
    except (OSError, ValueError):
        return None


def save_pin_labels(new: dict, path=None) -> dict[str, str]:
    """Rename the joints' pins to ``new`` ``{gpio: name}``, re-reading the file right before the atomic write so
    other keys keep what is on disk. Every joint pin needs a name and only joint pins can be named (ValueError
    otherwise; nothing is written). Returns ``{"<gpio>": name}`` as written."""
    path = check_writable(path or hardware_map())
    hw = read(path)
    new = {int(k): str(v).strip() for k, v in new.items()}
    drives = {int(p['pin']): p for _, _, p in joint_drives(hw)}
    if unknown := sorted(set(new) - set(drives)):
        raise ValueError(f'GPIO {", ".join(map(str, unknown))}: not a joint pin in the hardware map, so it has no name')
    if unnamed := sorted(g for g in drives if not new.get(g)):
        raise ValueError(f'GPIO {", ".join(map(str, unnamed))}: a joint pin needs a name')
    for g, p in drives.items():
        p['label'] = new[g]
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(hw, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)
    return {str(k): v for k, v in labels(hw).items()}
