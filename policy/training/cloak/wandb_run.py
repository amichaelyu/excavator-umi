"""Weights & Biases tracking of one CLOAK run: off unless the config's ``train.wandb`` turns it on.

``<run_dir>/wandb-run.json`` holds the run's W&B settings (policy.configs.schema.WANDB_DEFAULTS, resolved) and
its stable run id, never the API key:

    {"id": "3f9c0a1b2d4e", "project": "umi-excavator", "entity": null, "mode": "online",
     "name": "dirt-orb", "group": "dirt-orb", "tags": []}

``python -m policy train`` writes it (``write``, keeping an id already there) before it syncs the run directory to
the box, so every process of the run (the experiment's evaluations, the trainer, a resumed trainer, a later
``python -m policy eval``) continues the same W&B run. Without the file, or with mode ``disabled``, nothing here
imports wandb or calls it.

  check(settings)          fail fast: wandb importable and, for mode online, WANDB_API_KEY set
  session(run_dir, ...)    wandb.init(id=<stable id>, resume='allow', dir=run_dir, ...) for one process, finished
                           on exit; yields None when the run does not track
  log_eval(run_dir, ...)   one eval report's episode-mean normalized MAE at ``eval/step``

Training logs at ``step`` = the optimizer step. The evaluations run after training's last step and W&B drops steps
lower than a run's current one, so eval rows use their own x axis, ``eval/step`` (``define_metric``).

``WANDB_MODE``, when set, overrides the mode of a run that tracks (online / offline, or disabled for one launch);
``WANDB_ENTITY`` and ``WANDB_BASE_URL`` apply as usual. policy.training.lambda_host forwards these and
``WANDB_API_KEY`` from the launching shell into the run's containers.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import secrets
import sys

from policy.configs.schema import EVAL_BASELINES as BASELINES, WANDB_MODES as MODES

FILE = 'wandb-run.json'
LEGACY_MODES = {'run': 'online', 'dryrun': 'offline'}     # older WANDB_MODE spellings


def load(run_dir):
    """The run's ``wandb-run.json``, or None."""
    path = Path(run_dir) / FILE
    return json.loads(path.read_text()) if path.is_file() else None


def write(run_dir, settings):
    """Write ``settings`` (Config.wandb) to ``<run_dir>/wandb-run.json``, keeping the id already there (a new
    random id otherwise). Returns the record."""
    run_dir = Path(run_dir)
    record = {'id': (load(run_dir) or {}).get('id') or secrets.token_hex(6),
              **{k: v for k, v in settings.items() if k != 'id'}}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / FILE).write_text(json.dumps(record, indent=2) + '\n')
    return record


def mode(settings, environ=None):
    """The mode this process tracks with: ``disabled`` without settings or when they say so; else WANDB_MODE
    when set, else the settings' mode."""
    if not settings or settings.get('mode', 'disabled') == 'disabled':
        return 'disabled'
    environ = os.environ if environ is None else environ
    override = environ.get('WANDB_MODE')
    return LEGACY_MODES.get(override, override) if override else settings['mode']


def check(settings, environ=None, *, package=True):
    """The mode (see ``mode``) after making sure a tracking run can log, before any work starts. RuntimeError if
    wandb is missing (``package``: checked where the run executes, the training image) or mode online has no
    WANDB_API_KEY."""
    environ = os.environ if environ is None else environ
    current = mode(settings, environ)
    if current == 'disabled':
        return current
    if current not in MODES:
        raise RuntimeError(f'WANDB_MODE={current!r}: use one of {MODES}')
    if package:
        try:
            import wandb  # noqa: F401
        except ImportError:
            raise RuntimeError(f'train.wandb is on (mode {current}) but the wandb package is not installed here; it '
                               'is in the training dependency group (the umi-excavator-train image has it)') from None
    if current == 'online' and not environ.get('WANDB_API_KEY'):
        raise RuntimeError('train.wandb mode online needs WANDB_API_KEY: `export WANDB_API_KEY=...` in the shell '
                           'that runs `python -m policy train` (it is forwarded to the box for the run), or set '
                           'train.wandb.mode: offline (WANDB_MODE=offline for one launch)')
    return current


@contextlib.contextmanager
def session(run_dir, *, config=None, environ=None):
    """The run's W&B run for this process: initialized, or resumed under its stable id, with the settings of
    ``<run_dir>/wandb-run.json`` and its files under ``<run_dir>/wandb/``; ``config`` (the training recipe) is
    recorded with it. Yields None when the run does not track. Finished on exit, with exit code 1 when the
    block raised."""
    run_dir = Path(run_dir)
    settings = load(run_dir)
    current = check(settings, environ)
    if current == 'disabled':
        yield None
        return
    import wandb
    if not settings.get('id'):
        settings = write(run_dir, settings)
    run = wandb.init(project=settings.get('project') or 'umi-excavator', entity=settings.get('entity'),
                     name=settings.get('name'), group=settings.get('group'), tags=settings.get('tags') or None,
                     mode=current, config=config, dir=str(run_dir), id=settings['id'], resume='allow')
    ok = False
    try:
        yield run
        ok = True
    finally:
        run.finish(exit_code=0 if ok else 1)


def scalars(metrics):
    """The numeric entries of a metrics row (no ``step``, strings or flags): what training logs per step."""
    return {k: v for k, v in metrics.items()
            if k != 'step' and isinstance(v, (int, float)) and not isinstance(v, bool)}


def eval_metrics(report, *, step, masked=False, split='holdout'):
    """One eval report's W&B row: ``eval/<split>[-masked]/normalized_mae_<baseline>`` for policy, repeat_state
    and training_mean, at ``eval/step`` (0 for eval-initial, pi05_base). None without a policy score."""
    mae = report.get('episode_mean_normalized_mae') or {}
    values = {k: mae[k] for k in BASELINES if isinstance(mae.get(k), (int, float)) and not isinstance(mae[k], bool)}
    if 'policy' not in values:
        return None
    prefix = f'eval/{split}' + ('-masked' if masked else '')
    return {'eval/step': int(step), **{f'{prefix}/normalized_mae_{k}': float(v) for k, v in values.items()}}


def log_eval(run_dir, report, *, step, masked=False, split='holdout', environ=None):
    """Log one eval report (``eval_metrics``) to the run's W&B run when it tracks. Returns the row logged, or None.
    Never raises: the report is already written, so a W&B failure is a warning."""
    row = eval_metrics(report, step=step, masked=masked, split=split)
    if row is None or mode(load(run_dir), environ) == 'disabled':
        return None
    try:
        with session(run_dir, environ=environ) as run:
            run.define_metric('eval/step')
            run.define_metric('eval/*', step_metric='eval/step')
            run.log(row)
    except Exception as ex:
        print(f'W&B: not logged ({type(ex).__name__}: {ex}): {json.dumps(row)}', file=sys.stderr, flush=True)
        return None
    return row
