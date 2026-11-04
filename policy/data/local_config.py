"""Per-machine hosts: the ``hosts`` block of the gitignored configs/hub/local.json.

Copy configs/hub/local.example.json to configs/hub/local.json and fill in your machines (the hub also saves the
settings changed on its page there, under ``apps``):

    {"hosts": {"pi":     {"urls": ["http://raspberrypi.local:8000"],
                          "token_file": "hardware/umi-data-collection/.devices/bucket/access-token"},
               "lambda": {"host": "ubuntu@203.0.113.10", "root": "~/umi-excavator"}}}

    pi.urls         the excavator Pi's dashboard, in preference order (e.g. a wired link-local address first,
                    then its mDNS name or Wi-Fi address); every Pi reader (hub apps, policy runner, calibration)
                    starts from these
    pi.token_file   the dashboard's access token (repo-relative or ``~``; hardware/umi-data-collection/deploy/deploy.sh
                    saves it there)
    lambda.host     the GPU box of ``python -m policy`` stages when the config has no lambda.host and no --host
    lambda.root     the checkout on that box

A missing file or key falls back to the defaults below. ``$UMI_EXCAVATOR_LOCAL_CONFIG`` names another file.
Standard library only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / 'configs' / 'hub' / 'local.json'
ENV_VAR = 'UMI_EXCAVATOR_LOCAL_CONFIG'
DEFAULT_PI_URLS = ('http://raspberrypi.local:8000',)
DEFAULT_PI_TOKEN_FILE = 'hardware/umi-data-collection/.devices/bucket/access-token'
DEFAULT_LAMBDA_ROOT = '~/umi-excavator'


# Per-machine configs (gitignored) whose tracked ``<name>.example.json`` is the template: the machine's own wiring
# and calibration (pin map, camera, calibrated robot model). configs/hub/local*.json are read as above instead.
LOCAL_CONFIGS = ('configs/hardware/hardware_map.json', 'configs/hardware/camera-calibration/camera.json', 'configs/hardware/robot_full4_calibrated.json')
EXAMPLES_ENV = 'UMI_EXCAVATOR_EXAMPLE_CONFIGS'   # set (the tests do, conftest.py): read the examples, never this machine's


def ensure_local(file, examples: bool = True) -> Path:
    """``file``, created from its ``<name>.example.json`` the first time it is needed so a fresh checkout runs
    (edit the copy for your machine; calibration writes into it). Returns ``file``, or the example itself when
    ``$UMI_EXCAVATOR_EXAMPLE_CONFIGS`` is set and ``examples`` (False: always the local file, e.g. to ship it)."""
    file = Path(file)
    example = file.with_name(f'{file.stem}.example{file.suffix}')
    if examples and os.environ.get(EXAMPLES_ENV) and example.is_file():
        return example
    if not file.exists() and example.is_file():
        tmp = file.with_name(f'.{file.name}.{os.getpid()}.tmp')   # atomic: a concurrent reader never sees half a file
        tmp.write_bytes(example.read_bytes())
        os.replace(tmp, file)
        print(f'{file.name}: created from {example.name}; edit it for your machine', file=sys.stderr)
    return file


def check_writable(file) -> Path:
    """``file`` unless it is a tracked ``*.example.json`` template (ValueError): writers (calibration, pin names, the
    camera fit) only ever change this machine's copy."""
    file = Path(file)
    if file.name.endswith('.example.json'):
        raise ValueError(f'{file.name} is the tracked template; write the local {file.name.replace(".example", "")}')
    return file


def path(env: dict | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env[ENV_VAR]).expanduser() if env.get(ENV_VAR) else LOCAL


def hosts(file: Path | None = None) -> dict:
    """The ``hosts`` block of the local config ({} when the file or the block is missing)."""
    file = path() if file is None else Path(file)
    if not file.is_file():
        return {}
    block = json.loads(file.read_text()).get('hosts') or {}
    if not isinstance(block, dict):
        raise ValueError(f'{file}: "hosts" must be an object')
    return block


def _section(name: str, file: Path | None) -> dict:
    section = hosts(file).get(name) or {}
    if not isinstance(section, dict):
        raise ValueError(f'{path() if file is None else file}: hosts.{name} must be an object')
    return section


def pi_urls(file: Path | None = None) -> list[str]:
    """hosts.pi.urls (a list or a comma-separated string), else DEFAULT_PI_URLS."""
    urls = _section('pi', file).get('urls') or DEFAULT_PI_URLS
    if isinstance(urls, str):
        urls = urls.split(',')
    return [u.strip() for u in urls if u and u.strip()]


def pi_token_file(file: Path | None = None) -> Path:
    """hosts.pi.token_file (``~`` expanded, relative to the repository root), else DEFAULT_PI_TOKEN_FILE."""
    p = Path(_section('pi', file).get('token_file') or DEFAULT_PI_TOKEN_FILE).expanduser()
    return p if p.is_absolute() else REPO / p


def lambda_host(file: Path | None = None) -> str | None:
    """hosts.lambda.host (``user@address``), or None."""
    return _section('lambda', file).get('host') or None


def lambda_root(file: Path | None = None) -> str:
    """hosts.lambda.root, else DEFAULT_LAMBDA_ROOT."""
    return _section('lambda', file).get('root') or DEFAULT_LAMBDA_ROOT
