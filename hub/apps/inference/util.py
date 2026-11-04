"""Small helpers every inference module uses: the request refusal, JSON-safe values, JSON files, UTC stamps."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np


class Refused(Exception):
    """A request the dashboard refuses (HTTP status + message)."""

    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


def clean(o):
    """JSON-safe copy: numpy -> python, non-finite floats -> None, paths/dates -> str."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        return clean(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, '__dataclass_fields__'):
        import dataclasses
        return clean(dataclasses.asdict(o))
    return str(o)


def dumps(o) -> str:
    return json.dumps(clean(o), allow_nan=False)


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
