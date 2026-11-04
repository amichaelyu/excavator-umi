"""Locate the recordings root and refuse Git LFS pointer files.

Recordings live at ``<root>/<id>/`` (``metadata.json``, ``recording.tar``, ``video.mp4``). A config names its roots
(``source.recordings[].data_root``); tools without a config resolve one here:

    explicit ``--data-root`` > ``$UMI_DATA_DIR`` > ``<repo>/dataset``

(compose.yaml and policy/processing/docker/run.sh mount the same ``${UMI_DATA_DIR:-dataset}`` at /data.)

jj does not run Git LFS smudge filters, so a recording checked out through Git LFS can be a *pointer* text file;
:func:`require_real_file` refuses it.

Standard library only (no cv2, no numpy): the hub's data review app imports it as plainly as
policy.processing.slam.recording does.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV_VAR = 'UMI_DATA_DIR'
REPO_DEFAULT = REPO / 'dataset'
LFS_MAGIC = b'version https://git-lfs.github.com/spec/v1'


class LfsPointerError(RuntimeError):
    pass


def resolve_data_root(cli: str | os.PathLike | None = None, env: dict | None = None,
                      repo_default: Path | None = None) -> Path:
    env = os.environ if env is None else env
    if cli:
        return Path(cli).expanduser()
    if env.get(ENV_VAR):
        return Path(env[ENV_VAR]).expanduser()
    return repo_default or REPO_DEFAULT


def is_lfs_pointer(path: str | os.PathLike) -> bool:
    try:
        with open(path, 'rb') as f:
            return f.read(len(LFS_MAGIC)) == LFS_MAGIC
    except OSError:
        return False


def require_real_file(path: str | os.PathLike) -> Path:
    """Return ``path`` unchanged, or raise LfsPointerError if it is an LFS pointer."""
    path = Path(path)
    if is_lfs_pointer(path):
        raise LfsPointerError(
            f'{path} is a Git LFS pointer, not the real file (jj does not run LFS smudge filters). '
            f'Either run `git lfs pull` outside jj\'s view, or point {ENV_VAR} (or --data-root) at a directory '
            f'holding the real files.')
    return path
