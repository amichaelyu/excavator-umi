"""The ``evaluate`` stage of a run config: evaluate a trained run's checkpoints outside training.

``evaluate`` section keys (all optional; validated by policy.configs.schema.check_evaluate when the config loads;
defaults keep the train-time behaviour of policy.training.cloak.experiment, which already writes these reports
while training):

  every        int    train-time evaluation interval; the recipe evaluates every multiple of it plus the
                      final step (default: train.save_interval). Used by Config.evaluate_steps / recipe().
  steps        list of ints | 'all' | 'recipe'   checkpoints to evaluate here (default 'recipe': the same
                      steps as training, from ``every``; 'all': every checkpoint saved in the run directory)
  pretrained   bool   also evaluate pi05_base before fine-tuning (eval-initial.json; default true)
  mask_seed    int    seed of the synthetic edge-mask stress test (default: train.seed; also in the recipe)
  stress       'best' | 'all' | 'none'   which steps get the mask stress test, eval-<step>-masked.json
                      (default 'best': the step with the lowest holdout policy error, as in training). Only for
                      an out-of-view source (cohort.masks out_of_view); with twin masks it is skipped (none)
  split        'holdout' | 'train'   recordings to score (default holdout; train writes eval-*-train.json,
                      a fit check, never used for selection)
  predictions  bool   also save per-window predicted/target chunks, eval-<name>-predictions.npz (default false)
  runner       'lambda' | 'local'   where to run: in the umi-excavator-train image on lambda.host (default), or
                      this Python (e.g. already inside the container, or a local GPU)
  gpus         list of ints   GPUs on the box for the evaluation container (default: the first of lambda.gpus)
  backend      path   pinned CLOAK backend for runner local (default /opt/cloak if present, else
                      dataset/cloak-research/upstream)

The action horizon is not an option: it is the model's (train.horizon). Every report carries every metric
(policy/eval/metrics.py), so there is no metric selection either.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from policy.configs import ROOT
from policy.configs.schema import check_evaluate, check_steps
from policy.training import lambda_host
from policy.training.cloak import wandb_run
from policy.training.cloak.evaluator import cached, report_name

from . import report

DEFAULTS = {'steps': 'recipe', 'pretrained': True, 'stress': 'best', 'split': 'holdout', 'predictions': False,
            'runner': 'lambda'}
CONTAINER_BACKEND = '/opt/cloak'
LOCAL_BACKEND = 'dataset/cloak-research/upstream'


def options(cfg, steps=None) -> dict:
    """The resolved ``evaluate`` options of a cloak config; ``steps`` (CLI) overrides evaluate.steps."""
    section = cfg.section('evaluate')
    check_evaluate(section)
    opts = {**DEFAULTS, **{k: v for k, v in section.items() if k != 'every'}}
    if steps is not None:
        steps = 'all' if steps in ('all', ['all']) else steps
        steps = steps if steps in ('all', 'recipe') else [int(s) for s in steps]
        check_steps(steps, 'steps')
        opts['steps'] = steps
    if opts['steps'] == 'recipe':
        opts['steps'] = cfg.evaluate_steps()
    elif opts['steps'] != 'all':
        opts['steps'] = sorted(opts['steps'])
        beyond = [s for s in opts['steps'] if s > cfg.training['steps']]
        if beyond:
            raise ValueError(f'steps {beyond} are beyond train.steps ({cfg.training["steps"]})')
    opts.setdefault('mask_seed', cfg.training['seed'])
    if opts['stress'] != 'none' and cfg.masks != 'out_of_view':
        # Twin-rendered embodiment masks: the source is not out of view, so there is no synthetic edge-mask test
        # (policy.training.cloak.evaluator.stress_skip_reason); training skips it the same way.
        opts['stress_skipped'] = (f'synthetic edge-mask stress test skipped (evaluate.stress: {opts["stress"]}): '
                                  f'it needs an out-of-view source, this cohort has {cfg.masks} embodiment masks')
        opts['stress'] = 'none'
    opts.setdefault('gpus', cfg.section('lambda').get('gpus', [0])[:1])
    opts.setdefault('backend', CONTAINER_BACKEND if Path(CONTAINER_BACKEND).is_dir() else LOCAL_BACKEND)
    return opts


def _manifest_sha256(cfg):
    """The cohort the run trained on: its local manifest, else the run's own records (pulled from the box)."""
    manifest = ROOT / cfg.cohort_data / 'manifest.json'
    if manifest.is_file():
        return hashlib.sha256(manifest.read_bytes()).hexdigest()   # = policy.data.cloak_dataset.digest
    for name in ('experiment.json', 'cloak-run.json'):
        path = ROOT / cfg.run_dir / name
        if path.is_file():
            return json.loads(path.read_text()).get('manifest_sha256')
    return None


def plan(cfg, opts):
    """(report names expected, those already present for the same dataset, whether any work remains).
    Names that depend on reports not yet written (the best step's stress test, 'all' steps) are decided by
    the evaluation itself, so they only make work remain."""
    run, split, manifest = ROOT / cfg.run_dir, opts['split'], _manifest_sha256(cfg)
    names, done, todo = [], set(), opts['steps'] == 'all'

    def check(name, mask_seed=None):
        names.append(name)
        if cached(run / f'{name}.json', manifest, mask_seed) is not None and manifest is not None:
            done.add(name)

    if opts['pretrained']:
        check(report_name('initial', split=split))
    steps = [] if opts['steps'] == 'all' else opts['steps']
    for step in steps:
        check(report_name(step, split=split))
    todo = todo or len(done) < len(names)
    stressed = {'best': [], 'all': steps, 'none': []}[opts['stress']]
    if opts['stress'] == 'best' and steps and not todo:
        reports = {s: json.loads((run / f'{report_name(s, split=split)}.json').read_text()) for s in steps}
        stressed = [min(reports, key=lambda s: reports[s]['episode_mean_normalized_mae']['policy'])]
    elif opts['stress'] == 'best':
        todo = True
    for step in stressed:
        check(report_name(step, masked=True, split=split), opts['mask_seed'])
    return names, done, todo or len(done) < len(names)


def command(cfg, opts, python='python', backend=CONTAINER_BACKEND):
    """``python -m policy.eval run ...`` for this config, paths relative to the repository root."""
    steps = ['all'] if opts['steps'] == 'all' else [str(s) for s in opts['steps']]
    return [python, '-m', 'policy.eval', 'run', '--backend', str(backend), '--data', str(cfg.cohort_data),
            '--run-dir', str(cfg.run_dir), '--mode', cfg.training['mode'], '--steps', *steps,
            *([] if opts['pretrained'] else ['--no-pretrained']), '--stress', opts['stress'],
            '--mask-seed', str(opts['mask_seed']),
            *(['--split', opts['split']] if opts['split'] != 'holdout' else []),
            *(['--predictions'] if opts['predictions'] else [])]


def _check_cohort(cfg, opts):
    manifest = ROOT / cfg.cohort_data / 'manifest.json'
    if opts['split'] == 'holdout' and manifest.is_file() and json.loads(manifest.read_text()).get('holdout') == 'none':
        raise ValueError(f'{cfg.cohort_data} has holdout: none, so there is no held-out evaluation '
                         '(set evaluate.split: train for a training-fit check)')


def run(cfg, host, steps=None, dry_run=False) -> dict:
    """Evaluate the config's trained run outside of training.

    host: a policy.training.lambda_host.Host -> ``python -m policy.eval run`` in the umi-excavator-train image on
    the box against the run directory there, then the reports are copied back to dataset/runs/<name>/.
    None -> the same command with this Python, on the local run directory. Reports that already exist for
    the same dataset are kept; if nothing is missing no evaluation is started.
    Returns {'reports': {name: {'path', 'status'}}, 'summary': summary(cfg)}; status is existing (kept), new
    (written by this call) or planned (dry run, or decided by the evaluation and not written)."""
    dry_run = dry_run or bool(getattr(host, 'dry_run', False))
    opts = options(cfg, steps)
    _check_cohort(cfg, opts)
    if opts.get('stress_skipped'):
        print(opts['stress_skipped'], file=sys.stderr)
    if host is not None:
        host.pull(cfg.run_dir)                       # the reports written so far on the box
    names, done, todo = plan(cfg, opts)
    before = _mtimes(cfg)
    if todo:
        if host is not None:
            _run_remote(cfg, host, opts, dry_run, initial=report_name('initial', split=opts['split']) not in done)
        else:
            _run_local(cfg, opts, dry_run)
    else:
        print(f'All {len(names)} reports exist for this dataset; nothing to evaluate', file=sys.stderr)
    after = _mtimes(cfg)
    reports = {name: 'existing' if name in done else 'planned' for name in names}
    reports.update({name: 'new' for name, m in after.items() if before.get(name) != m})
    return {'reports': {name: {'path': str(cfg.run_dir / f'{name}.json'), 'status': status}
                        for name, status in reports.items()},
            'summary': summary(cfg), **({'stress_skipped': opts['stress_skipped']} if 'stress_skipped' in opts else {})}


def _mtimes(cfg):
    run = ROOT / cfg.run_dir
    return {p.stem: p.stat().st_mtime_ns for p in run.glob('eval-*.json') if report.parse_name(p.name)} \
        if run.is_dir() else {}


def _run_local(cfg, opts, dry_run):
    backend = opts['backend']
    cmd = command(cfg, opts, sys.executable, backend)
    print('+ ' + shlex.join(cmd), file=sys.stderr, flush=True)
    if dry_run:
        return
    if not (ROOT / cfg.run_dir / 'cloak-run.json').is_file():
        raise FileNotFoundError(f'{cfg.run_dir} has no cloak-run.json: train first')
    subprocess.run(cmd, cwd=ROOT, check=True)


def _run_remote(cfg, host, opts, dry_run, initial):
    lambda_gpus = cfg.section('lambda').get('gpus', [0])
    train = f'train-{cfg.name}'
    if not dry_run and set(opts['gpus']) & set(lambda_gpus):
        out = host.ssh(f'docker ps -q --filter name=^{train}$', capture=True)
        if out and out.strip():
            raise RuntimeError(f'{train} is still running on {host.address} on GPUs {lambda_gpus}; it evaluates '
                               'its own steps. Wait for it, or set evaluate.gpus to free GPUs')
    host.push_code()
    host.push_data(cfg.cohort_data)
    host.ensure_image()
    if opts['pretrained'] and initial:
        host.docker_run('fetch-pi05-base', ['python', '-m', 'policy.training.weights'], detach=False, replace=True)
    # A run that tracks with W&B logs each new report to its run (<run_dir>/wandb-run.json on the box).
    env = lambda_host.forwarded_env() if wandb_run.mode(cfg.wandb) != 'disabled' else {}
    host.docker_run(f'eval-{cfg.name}', command(cfg, opts), gpus=opts['gpus'], detach=False, replace=True,
                    **({'env': env} if env else {}))
    host.pull(cfg.run_dir)


def summary(cfg) -> dict:
    """The local reports of the config's run (dataset/runs/<name>/) as a dict; ``table`` is printable."""
    run = ROOT / cfg.run_dir
    s = {**report.aggregate(run), 'run_dir': str(cfg.run_dir)}
    return {**s, 'table': report.format_table(s)}
