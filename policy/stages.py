"""The training hub's pipeline: one config, every stage, in order (``python -m policy <stage> <config>``).

    reconstruct  ORB-SLAM3+smoother trajectories              policy.training.stages.reconstruct   reconstruct.runner
    labels       twin IK replays -> four-joint action labels  policy.preprocessing.labels          labels.runner
    cohort       export, embodiment masks, frozen split       policy.preprocessing.cohort          cohort.runner
    train        π0.5 / CLOAK experiment on the Lambda box    policy.training.stages.train         lambda
    eval         evaluation of the trained run                policy.eval.stage.run                evaluate.runner
    serve        bundle and serve the selected step           policy.training.stages.serve         lambda

``all`` runs reconstruct -> labels -> cohort -> train -> eval; every stage skips work whose output exists, so
``all`` can be rerun after a failure. Utilities: status, stop, recipe, image, check (validate + print the plan).

This module (with __main__) is the top of policy/: it may import every subpackage, which only import downward
(configs < data < digital_twin < processing < preprocessing < training < eval; policy/tests/test_layering.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
import sys

from policy.configs import ROOT
from policy.configs.schema import EVALUATE_RUNNERS, PREPROCESS_RUNNERS, RECONSTRUCT_RUNNERS
from policy.eval import stage as evaluation
from policy import preprocessing
from policy.training import stages as training

PIPELINE = ('reconstruct', 'labels', 'cohort', 'train', 'eval')          # `all`, in order
STAGES = (*PIPELINE, 'serve')
UTILITIES = ('status', 'stop', 'recipe', 'image', 'check')
COMMANDS = (*STAGES, 'all', *UTILITIES)
RUNNERS = {'reconstruct': RECONSTRUCT_RUNNERS, 'labels': PREPROCESS_RUNNERS, 'cohort': PREPROCESS_RUNNERS,
           'eval': EVALUATE_RUNNERS}


@dataclass
class Run:
    """One invocation: the config plus the command line's one-off options."""
    cfg: object
    dry_run: bool = False
    host: str | None = None          # overrides lambda.host
    runner: str | None = None        # overrides <stage>.runner for the one stage named on the command line
    resume: bool = False             # train: continue from the latest checkpoint
    register: bool = False           # serve: add the endpoint to configs/hub/local_checkpoints.json
    steps: list | str | None = None  # eval: these saved steps (or 'all') instead of the evaluate section's

    @cached_property
    def lambda_host(self):
        return training.host_for(self.cfg, self.host, self.dry_run)

    def where(self, stage):
        return self.runner or self.cfg.runner(stage)


def run(cfg, command, **options):
    """Run ``command`` (a stage, ``all`` or a utility) for ``cfg``."""
    r = Run(cfg, **options)
    if command not in COMMANDS:
        raise ValueError(f'Unknown stage {command!r}; one of {COMMANDS}')
    if r.runner:
        if command == 'all':
            raise ValueError('--runner overrides one stage; set <stage>.runner in the config for `all`')
        if r.runner not in RUNNERS.get(command, ()):
            raise ValueError(f'{command} has no runner choice' if command not in RUNNERS
                             else f'{command} runs on one of {RUNNERS[command]}, not {r.runner!r}')
    if command == 'check':
        return check(cfg, r.host)
    if command == 'all':
        for stage in PIPELINE:
            _banner(stage, r)
            STAGE_FUNCTIONS[stage](r, in_all=True)
        return None
    return STAGE_FUNCTIONS[command](r)


def _banner(stage, r):
    where = r.cfg.runner(stage)
    if where == 'lambda':
        where += f' {training.lambda_address(r.cfg, r.host) or "?"}'
    print(f'\n== {stage} ({where}) ==', file=sys.stderr, flush=True)


def _reconstruct(r, in_all=False):
    training.reconstruct(r.cfg, r.dry_run, r.host, r.where('reconstruct'))


def _preprocess(stage):
    def go(r, in_all=False):
        if r.where(stage) == 'docker':
            return preprocessing.in_docker(r.cfg, stage, r.dry_run)
        return getattr(preprocessing, stage)(r.cfg, r.dry_run)
    return go


def _train(r, in_all=False):
    training.train(r.cfg, r.lambda_host, r.resume, wait=in_all)


def _eval(r, in_all=False):
    """``policy.eval.stage.run(cfg, host, steps, dry_run)``: host is the Lambda box, or None (``evaluate.runner:
    local``) to evaluate on this machine."""
    host = None if r.where('eval') == 'local' else r.lambda_host
    result = evaluation.run(r.cfg, host, steps=r.steps, dry_run=r.dry_run)
    table = (result.get('summary') or {}).get('table') if isinstance(result, dict) else None
    if table:
        for name, report in result.get('reports', {}).items():
            print(f'{name:28s} {report.get("status", "")}')
        print(table)
    elif result:
        print(json.dumps(result, indent=2, default=str))
    return result


def _serve(r, in_all=False):
    training.serve(r.cfg, r.lambda_host, r.register)


def _status(r, in_all=False):
    training.status(r.cfg, r.lambda_host)


def _stop(r, in_all=False):
    training.stop(r.cfg, r.lambda_host)


def _image(r, in_all=False):
    print(r.lambda_host.ensure_image())


def _recipe(r, in_all=False):
    print(json.dumps(r.cfg.recipe(), indent=2))


STAGE_FUNCTIONS = {'reconstruct': _reconstruct, 'labels': _preprocess('labels'), 'cohort': _preprocess('cohort'),
                   'train': _train, 'eval': _eval, 'serve': _serve, 'status': _status, 'stop': _stop,
                   'image': _image, 'recipe': _recipe}


# --- check: the resolved plan -------------------------------------------------------------------------------------

@dataclass
class Step:
    stage: str
    where: str
    detail: str
    inputs: list
    outputs: list
    done: bool | None = None         # output already there (None: not tracked locally)


def _exists(path):
    return (ROOT / path).exists()


def _any(path, pattern):
    return any((ROOT / path).glob(pattern))


def plan(cfg, host=None):
    """Every stage of ``cfg`` with where it runs, what it reads and writes, and whether its output exists."""
    box = f'{training.lambda_address(cfg, host) or "<lambda.host unset>"}:{training.lambda_root(cfg)}'
    src, lab, coh = cfg.section('source'), cfg.section('labels'), cfg.section('cohort')
    srv, training_opts = cfg.section('serve'), cfg.training
    recon, labels, run = cfg.dataset('reconstructions'), cfg.dataset('labels'), cfg.run_dir
    places = {'local': 'this machine', 'docker': 'Docker on this machine', 'lambda': f'Docker on {box}'}
    runner = cfg.runner('reconstruct')
    selection = cfg.dataset('cohorts') / 'selection.json'
    roots = [Path(r['data_root']) for r in src['recordings']]
    image = {'local': '', 'docker': ', umi-excavator-processing image', 'lambda': ', umi-excavator-processing image'}[runner]
    steps = [Step('reconstruct', f'{runner}: {places[runner]}{image}',
                  f'policy.processing.reconstruct_orbslam ({cfg.method}), {cfg.section("reconstruct").get("workers", 8)} workers',
                  [*dict.fromkeys(roots), cfg.calibration],
                  [recon / '<id>' / 'trajectory.csv'], _any(recon, '*/trajectory.csv'))]
    for stage, detail, inputs, outputs, done in (
            ('labels', f'digital_twin.sim.replay, profile {lab.get("profile", "full4_calibrated")}, planner '
                       f'{lab.get("planner", "lookahead")}, contact_height {lab.get("contact_height")}, '
                       f'pitch_clip {bool(lab.get("pitch_clip"))}, {lab.get("workers", 8)} workers',
             [recon], [labels / '<id>' / 'actions.csv'], _any(labels, '*/metrics.json')),
            ('cohort', f'policy.preprocessing.export (quality {coh.get("quality_policy", "lenient")}), '
                       f'{cfg.masks} embodiment masks, frozen train/holdout split',
             [Path(selection), recon, labels], [cfg.cohort_data],
             _exists(cfg.cohort_data / 'manifest.json'))):
        where = cfg.runner(stage)
        steps.append(Step(stage, f'{where}: {places[where]}' + (', umi-excavator-preprocess image' if where == 'docker' else ''),
                          detail, inputs, outputs, done))
    gpus, tracking = cfg.section('lambda').get('gpus', [0]), cfg.wandb
    wandb = 'W&B off' if tracking['mode'] == 'disabled' else \
        (f'W&B {tracking["mode"]} {tracking["entity"] + "/" if tracking["entity"] else ""}{tracking["project"]}, '
         f'run {tracking["name"]}, group {tracking["group"]}' + (f', tags {tracking["tags"]}' if tracking['tags'] else ''))
    steps.append(Step('train', f'lambda: {places["lambda"]}, umi-excavator-train image, GPUs {gpus}',
                      f'policy.training.cloak.experiment, mode {training_opts["mode"]}, {training_opts["steps"]} steps, '
                      f'batch {training_opts["batch"]}, fsdp {training_opts["fsdp"]}, holdout eval at '
                      f'{cfg.evaluate_steps()}; {wandb}',
                      [cfg.cohort_data], [run / 'recipe.json', run / 'result.json'], _exists(run / 'result.json')))
    where = cfg.runner('eval')
    steps.append(Step('eval', f'{where}: {places[where]}',
                      f'policy.eval.stage.run (evaluate: {json.dumps(cfg.section("evaluate"))})',
                      [run / 'result.json'], [run], None))
    port = srv.get('port', 8020)
    steps.append(Step('serve', f'lambda: {places["lambda"]}, GPUs {srv.get("gpus", [0])}',
                      f'policy.training.cloak.bundle, step {srv.get("step", "best")}, 127.0.0.1:{port} (ssh -L)',
                      [run / 'result.json'], [run / 'bundle-<step>'], None))
    return steps


def check(cfg, host=None, out=None):
    """Print the validated config's plan (load() has already validated every section)."""
    out = out or sys.stdout
    rel = cfg.path.relative_to(ROOT) if cfg.path.is_relative_to(ROOT) else cfg.path
    print(f'{cfg.name}: kind {cfg.kind}, task {cfg.task} ({rel}) - config OK', file=out)
    print('`all` runs ' + ' -> '.join(PIPELINE) + '; serve after.', file=out)
    steps = plan(cfg, host)
    produced = {}                    # output path -> the stage that writes it
    for i, step in enumerate(steps, 1):
        state = '' if step.done is None else ('  [done]' if step.done else '  [pending]')
        print(f'\n{i}. {step.stage:<12}{step.where}{state}\n   {step.detail}', file=out)
        for path in step.inputs:
            source = next((stage for out_path, stage in produced.items()
                           if path == out_path or out_path.is_relative_to(path)), None)
            note = f'  (from {source})' if source else ('' if _exists(path) else '  (MISSING)')
            print(f'   in   {path}{note}', file=out)
        for path in step.outputs:
            print(f'   out  {path}', file=out)
            produced.setdefault(path, step.stage)
    return steps
