"""Load the published CLOAK fork without modifying another OpenPI checkout."""
from pathlib import Path
import subprocess
import sys

from . import UPSTREAM_REVISION, UPSTREAM_URL


def activate(root):
    root = Path(root).resolve()
    try:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f'Clone {UPSTREAM_URL} and check out {UPSTREAM_REVISION}') from exc
    if revision != UPSTREAM_REVISION:
        raise ValueError(f'Expected CLOAK revision {UPSTREAM_REVISION}, got {revision}')
    if subprocess.call(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', 'src']):
        raise ValueError('CLOAK backend source differs from the pinned release')
    source = root / 'src'
    for name, module in tuple(sys.modules.items()):
        if name == 'openpi' or name.startswith('openpi.'):
            file = getattr(module, '__file__', None)
            if file and not Path(file).resolve().is_relative_to(source):
                raise ValueError('A different OpenPI is already imported; start a fresh process')
    sys.path.insert(0, str(source))
    return root
