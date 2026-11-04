"""A Lambda GPU box reached over SSH: sync code and data, build the training (or processing) image, run containers.

The box needs Docker with the NVIDIA runtime (Lambda Stack ships both) and the ``ubuntu`` user in the
``docker`` group. The image holds only the runtime (CUDA JAX, the pinned CLOAK backend); the synced
repository is bind-mounted at ``/workspace``, so code changes need a sync, not a rebuild.

Secrets (the W&B key of a run that tracks, FORWARD_ENV) go from this shell's environment to a container as
``docker run -e NAME``: ``ssh`` reads the values from its stdin into the remote shell's environment, so they are
never on a command line (here or on the box), in the printed commands, or in a file.
"""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from policy.configs import ROOT

DOCKERFILE = ROOT / 'policy' / 'training' / 'docker' / 'train.Dockerfile'
IMAGE = 'umi-excavator-train'
# Tracked code the container needs; datasets travel separately (push_data), results come back with pull.
CODE = ('policy/configs', 'configs/policy', 'configs/sim', 'configs/hardware_map.json', 'configs/camera.json',
        'policy/data', 'digital_twin/paths.py', 'digital_twin/sim', 'policy/training', 'policy/eval', 'pyproject.toml',
        'uv.lock', '.python-version')
# The ORB-SLAM3 reconstruction image (policy/processing/docker/Dockerfile; context: the repository root).
PROCESSING_DOCKERFILE = 'policy/processing/docker/Dockerfile'
PROCESSING_IMAGE = 'umi-excavator-processing'
PROCESSING_CODE = ('policy/data', 'policy/processing', 'configs/camera.json', 'pyproject.toml',
                   'uv.lock', '.python-version')
# What that image compiles or installs. Its Python code is overridden by the repository mounted at /workspace.
PROCESSING_INPUTS = ('policy/processing/docker/Dockerfile', 'policy/processing/docker/Dockerfile.dockerignore',
                     'policy/processing/slam/build_probe.sh', 'policy/processing/slam/orbslam_probe.cc',
                     'pyproject.toml', 'uv.lock', '.python-version')


# Environment a run that tracks with W&B (policy.training.cloak.wandb_run) takes from the launching shell.
FORWARD_ENV = ('WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_MODE', 'WANDB_BASE_URL')


def forwarded_env(environ=None):
    """{name: value} of the FORWARD_ENV variables set (non-empty) in ``environ`` (default: this process)."""
    environ = os.environ if environ is None else environ
    return {name: environ[name] for name in FORWARD_ENV if environ.get(name)}


def image_tag():
    """Content hash of everything baked into the image."""
    from policy.training.cloak import UPSTREAM_REVISION
    h = hashlib.sha256(UPSTREAM_REVISION.encode())
    for path in (DOCKERFILE, ROOT / 'pyproject.toml', ROOT / 'uv.lock', ROOT / '.python-version'):
        h.update(path.read_bytes())
    return f'{IMAGE}:{h.hexdigest()[:12]}'


def processing_image_tag():
    """Content hash of the processing image's build inputs."""
    h = hashlib.sha256()
    for name in PROCESSING_INPUTS:
        path = ROOT / name
        for f in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if f.is_file() and '__pycache__' not in f.parts:
                h.update(str(f.relative_to(ROOT)).encode() + b'\0' + f.read_bytes())
    return f'{PROCESSING_IMAGE}:{h.hexdigest()[:12]}'


def processing_build_command(tag):
    """Shell command, run from the repository root, that builds the processing image unless ``tag`` exists."""
    return (f'docker image inspect {tag} >/dev/null 2>&1 || docker build -f {PROCESSING_DOCKERFILE} '
            f'-t {tag} -t {PROCESSING_IMAGE}:latest .')


class Host:
    def __init__(self, address, root='~/umi-excavator', dry_run=False):
        self.address, self.root, self.dry_run = address, root.rstrip('/'), dry_run

    def _run(self, cmd, check=True, capture=False, stdin=None):
        print('+ ' + shlex.join(cmd), file=sys.stderr, flush=True)
        if self.dry_run:
            return ''
        result = subprocess.run(cmd, check=check, text=True, input=stdin, stdout=subprocess.PIPE if capture else None)
        return result.stdout if capture else result.returncode

    def ssh(self, command, check=True, capture=False, env=None):
        """Run a shell command on the box from the synced repository root. ``env``: {name: value} exported in the
        remote shell first, sent over stdin (not on the command line)."""
        env = env or {}
        for name, value in env.items():
            if not re.fullmatch(r'[A-Z_][A-Z0-9_]*', name) or '\n' in value:
                raise ValueError(f'cannot forward {name!r}: a variable name and a one-line value are required')
        if env:
            print(f'# exported on the box from this shell (via stdin): {", ".join(env)}', file=sys.stderr, flush=True)
        reads = ''.join(f'IFS= read -r {name}; export {name}; ' for name in env)
        return self._run(['ssh', self.address, f'{reads}mkdir -p {self.root} && cd {self.root} && {command}'],
                         check=check, capture=capture, stdin=''.join(f'{v}\n' for v in env.values()) if env else None)

    def push_code(self, code=CODE):
        files = subprocess.run(['jj', 'file', 'list', *code], cwd=ROOT, check=True, text=True,
                               capture_output=True).stdout
        cmd = ['rsync', '-az', '--files-from=-', './', f'{self.address}:{self.root}/']
        print('+ ' + shlex.join(cmd) + f'  # {len(files.splitlines())} tracked files', file=sys.stderr, flush=True)
        if not self.dry_run:
            subprocess.run(cmd, cwd=ROOT, input=files, text=True, check=True)

    def push_data(self, path, dest=None, only=None):
        """rsync a directory or file to ``dest`` (default: the same path relative to the repository root).
        ``only``: of a directory, just these top-level entries."""
        path = Path(path)
        dest = Path(dest or path)
        if not self.dry_run and not (ROOT / path).exists():
            raise FileNotFoundError(f'{path} is not built locally')
        self.ssh(f'mkdir -p {shlex.quote(str(dest.parent))}')
        local = ROOT / path
        is_file = local.is_file() or (not local.exists() and bool(path.suffix))   # dry run: guess
        slash = '' if is_file else '/'
        select = [*(f'--include=/{name}/***' for name in only), '--exclude=*'] if only is not None and not is_file else []
        self._run(['rsync', '-az', '--partial', *select, f'{ROOT / path}{slash}', f'{self.address}:{self.root}/{dest}{slash}'])

    def pull(self, path, exclude=('checkpoints', 'bundle-*', '.jax-cache')):
        path = Path(path)
        if not self.dry_run:
            (ROOT / path).mkdir(parents=True, exist_ok=True)
        self._run(['rsync', '-az', *[f'--exclude={e}' for e in exclude],
                   f'{self.address}:{self.root}/{path}/', f'{ROOT / path}/'])

    def ensure_image(self):
        """Build the training image on the box unless this exact tag is already there."""
        from policy.training.cloak import UPSTREAM_REVISION
        tag = image_tag()
        if self.ssh(f'docker image inspect {tag} >/dev/null 2>&1', check=False) != 0:
            self.ssh(f'docker build -f policy/training/docker/train.Dockerfile --build-arg CLOAK_REV={UPSTREAM_REVISION} '
                     f'-t {tag} -t {IMAGE}:latest .')
        return tag

    def ensure_processing_image(self):
        """Build the processing image on the box unless this exact tag is already there."""
        tag = processing_image_tag()
        self.ssh(processing_build_command(tag))
        return tag

    def docker_run(self, name, args, gpus=(), ports=(), detach=True, replace=False, image=None, env=None):
        """``docker run`` with the repository at /workspace: the training image (+ the OpenPI cache), or
        ``image`` (e.g. the processing image) working in /workspace. ``env``: {name: value} passed into the
        container (``-e NAME``; the value reaches the box through ``ssh``'s stdin)."""
        tag = image or image_tag()
        flags = ['--name', name, '--ipc=host', '--user', '"$(id -u):$(id -g)"', '-v', '"$PWD":/workspace']
        flags += ['-w', '/workspace'] if image else ['-v', '"$HOME/.cache/openpi":/cache/openpi']
        if gpus:
            flags += ['--gpus', shlex.quote(f'"device={",".join(map(str, gpus))}"'),
                      '-e', 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, range(len(gpus))))]
        for port in ports:
            flags += ['-p', f'127.0.0.1:{port}:{port}']
        for key in env or {}:
            flags += ['-e', key]
        flags += ['-d'] if detach else ['--rm']
        prefix = f'docker rm -f {name} >/dev/null 2>&1; ' if replace else ''
        cache = '' if image else 'mkdir -p "$HOME/.cache/openpi" && '
        return self.ssh(prefix + cache + 'docker run ' + ' '.join(flags)
                        + f' {tag} ' + shlex.join(args), **({'env': env} if env else {}))
