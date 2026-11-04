"""Run configs (one YAML per trained policy) and their schema: what `python -m policy <stage> <config>` runs."""
from .schema import ROOT, Config, load

__all__ = ["ROOT", "Config", "load"]
