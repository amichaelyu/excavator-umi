"""Evaluation reports of one CLOAK run and checkpoint selection, as the experiment and the eval stage drive them.

  report_name / cached     ``eval-<label>[-masked][-train].json`` names; reuse of an existing report of the same data
  RunEvaluator             writes one report per fresh ``python -m policy.eval checkpoint`` process (releases
                           parameters and compiled buffers), on the first visible GPU, logged next to the report
  evaluate_run             pi05_base, the given (or all saved) steps, then the mask stress test (out-of-view
                           sources only); picks the best step

Scoring itself (load a checkpoint, run it over a split) is policy.eval.cloak, which runs in that child process:
this module only names, caches and launches, so training (policy.training.cloak.experiment) selects its
checkpoint without importing the evaluation package above it. Report aggregation: policy.eval.report.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from policy.configs import ROOT
from policy.configs.schema import EVALUATE_SPLITS as SPLITS, EVALUATE_STRESS as STRESS
from policy.data.cloak_dataset import digest

CHECKPOINTS = Path('checkpoints') / 'excavator_cloak'   # <run_dir>/checkpoints/excavator_cloak/<mode>/<step>


def report_name(label, *, masked=False, split='holdout'):
    """``eval-initial``, ``eval-250``, ``eval-250-masked``; ``-train`` appended for the train split."""
    return f'eval-{label}' + ('-masked' if masked else '') + ('' if split == 'holdout' else f'-{split}')


def checkpoint_dir(run_dir, mode, step):
    return Path(run_dir) / CHECKPOINTS / mode / str(step)


def saved_steps(run_dir, mode=None):
    """Checkpoint steps on disk; ``mode`` defaults to the only one under checkpoints/excavator_cloak/."""
    root = Path(run_dir) / CHECKPOINTS
    if mode is None:
        modes = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        if len(modes) != 1:
            raise ValueError(f'Pass --mode: {root} holds {modes or "no"} training modes')
        mode = modes[0]
    folder = root / mode
    steps = sorted(int(p.name) for p in folder.iterdir() if p.is_dir() and re.fullmatch(r'\d+', p.name)) \
        if folder.is_dir() else []
    return mode, steps


def cached(path, manifest_sha256, mask_seed=None):
    """The report at ``path`` if it can be reused: same dataset and, for a masked report, the same mask seed.
    A report of another dataset is an error (a run directory belongs to one frozen cohort)."""
    path = Path(path)
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if manifest_sha256 is not None and report.get('manifest_sha256') != manifest_sha256:
        raise ValueError(f'Cached evaluation {path} belongs to another dataset')
    if mask_seed is not None and report.get('mask_seed') != mask_seed:
        return None
    return report


class RunEvaluator:
    """Writes ``<run_dir>/<name>.json`` reports of one run, each by a fresh ``python -m policy.eval checkpoint``
    process on the first visible GPU, logged to ``<run_dir>/<name>.log``. Existing reports of the same dataset are reused."""

    def __init__(self, backend, data, run_dir, *, manifest_sha256=None, cwd=ROOT, env=None, split='holdout',
                 predictions=False, launch=None):
        if split not in SPLITS:
            raise ValueError(f'split must be one of {SPLITS}')
        self.backend, self.data, self.run_dir = str(backend), str(data), Path(run_dir)
        manifest = Path(data) / 'manifest.json'
        # The dataset's manifest when it is readable here (None: unknown, e.g. a remote path).
        self.manifest = json.loads(manifest.read_text()) if manifest.is_file() else None
        if manifest_sha256 is None and self.manifest is not None:
            manifest_sha256 = digest(manifest)
        self.manifest_sha256, self.cwd, self.split, self.predictions = manifest_sha256, cwd, split, predictions
        self.env = dict(os.environ if env is None else env)
        if self.env.get('CUDA_VISIBLE_DEVICES'):
            # One GPU per evaluation: the first of the process's assigned devices, no competing trainer.
            self.env['CUDA_VISIBLE_DEVICES'] = self.env['CUDA_VISIBLE_DEVICES'].split(',')[0]
        self.launch = launch or self._subprocess

    def command(self, output, *, checkpoint=None, pretrained=False, mask_seed=None):
        name = Path(output).stem
        return [sys.executable, '-m', 'policy.eval', 'checkpoint', '--backend', self.backend, '--data', self.data,
                '--run-dir', str(self.run_dir),
                *(['--pretrained'] if pretrained else ['--checkpoint', str(checkpoint)]),
                *(['--mask-seed', str(mask_seed)] if mask_seed is not None else []),
                *(['--split', self.split] if self.split != 'holdout' else []),
                *(['--predictions', str(self.run_dir / f'{name}-predictions.npz')] if self.predictions else []),
                '--output', str(output)]

    def _subprocess(self, cmd, log):
        with open(log, 'a') as f:
            subprocess.run(cmd, cwd=self.cwd, env=self.env, stdout=f, stderr=subprocess.STDOUT, check=True)

    def __call__(self, label, *, checkpoint=None, pretrained=False, mask_seed=None):
        name = report_name(label, masked=mask_seed is not None, split=self.split)
        output = self.run_dir / f'{name}.json'
        report = cached(output, self.manifest_sha256, mask_seed)
        if report is None:
            self.launch(self.command(output, checkpoint=checkpoint, pretrained=pretrained, mask_seed=mask_seed),
                        self.run_dir / f'{name}.log')
            report = json.loads(output.read_text())
        return name, report


def best_step(reports):
    """Step with the lowest episode-mean normalized policy MAE among {step: report}."""
    return min(reports, key=lambda step: reports[step]['episode_mean_normalized_mae']['policy'])


def stress_skip_reason(manifest):
    """Why the synthetic edge-mask stress test cannot run on a dataset with this manifest, or None if it can (or
    the manifest is unknown, None: the evaluation process checks it). The test replaces an empty source mask
    with edge occlusions (policy.training.cloak.masking.augment_out_of_view), so it needs an out-of-view source;
    twin-rendered or reviewed embodiment masks are not."""
    if manifest is None or manifest.get('source_visibility') == 'out_of_view':
        return None
    return ('synthetic edge-mask stress test skipped: it needs an out-of-view source (source_visibility: '
            f'out_of_view), this dataset has {manifest.get("source_visibility", "embodiment masks")!r}')


def evaluate_run(evaluator, *, steps, mode=None, pretrained=True, stress='best', mask_seed=None):
    """pi05_base (``pretrained``), every checkpoint in ``steps`` (a list, or ``'all'`` saved ones), then the
    synthetic-edge-mask stress test of the best step (``stress: best``), of every step (``all``) or none.
    The stress test only runs on an out-of-view source (stress_skip_reason); otherwise it is skipped.
    Returns {'reports': {name: report}, 'best_step': step | None, 'stress': the stress applied,
    'stress_skipped': reason | None}."""
    if stress not in STRESS:
        raise ValueError(f'stress must be one of {STRESS}')
    if stress != 'none' and mask_seed is None:
        raise ValueError('The mask stress test needs a mask seed')
    skipped = stress_skip_reason(evaluator.manifest) if stress != 'none' else None
    if skipped:
        stress = 'none'
    mode, saved = saved_steps(evaluator.run_dir, mode)
    steps = saved if steps == 'all' else [int(s) for s in steps]
    missing = sorted(set(steps) - set(saved))
    if missing:
        raise FileNotFoundError(f'No checkpoint for steps {missing} in {evaluator.run_dir / CHECKPOINTS / mode} '
                                f'(saved: {saved})')
    reports = {}
    if pretrained:
        name, reports[name] = evaluator('initial', pretrained=True)
    by_step = {}
    for step in steps:
        name, by_step[step] = evaluator(step, checkpoint=checkpoint_dir(evaluator.run_dir, mode, step))
        reports[name] = by_step[step]
    best = best_step(by_step) if by_step else None
    for step in {'best': [best] if by_step else [], 'all': steps, 'none': []}[stress]:
        name, reports[name] = evaluator(step, checkpoint=checkpoint_dir(evaluator.run_dir, mode, step),
                                        mask_seed=mask_seed)
    return {'reports': reports, 'best_step': best, 'stress': stress, 'stress_skipped': skipped}
