"""Puts umi-data-collection/ on sys.path so the tests import the ``recorder`` package
(umi-data-collection has hyphens, so it cannot be a package itself)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
