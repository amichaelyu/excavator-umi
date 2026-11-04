"""Fetch public pi05_base weights into the OpenPI cache, in parallel (runs inside the training image).

``openpi.shared.download`` fetches gs:// files one at a time at about 1 MB/s anonymously; gcsfs pulls the
same files concurrently in seconds. The layout matches what ``maybe_download`` looks for, so training
and ``serve-base`` find the cached copy.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil

SOURCE = 'openpi-assets/checkpoints/pi05_base'


def fetch(cache=None):
    cache = Path(cache or os.environ.get('OPENPI_DATA_HOME', '~/.cache/openpi')).expanduser()
    target = cache / SOURCE
    if (target / 'params' / '_METADATA').is_file():
        return target
    import gcsfs
    partial = target.with_name(target.name + '.partial')
    shutil.rmtree(partial, ignore_errors=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    gcsfs.GCSFileSystem(token='anon').get(f'{SOURCE}/', f'{partial}/', recursive=True)
    if not (partial / 'params' / '_METADATA').is_file():
        raise RuntimeError(f'Incomplete download of gs://{SOURCE}')
    partial.rename(target)
    return target


if __name__ == '__main__':
    print(fetch())
