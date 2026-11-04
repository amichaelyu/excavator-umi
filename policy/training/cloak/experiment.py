"""Run a task-specific CLOAK experiment with isolated normalization and checkpoints.

Data check, pi05_base evaluation, training, then a holdout evaluation of every recipe step, the mask stress test
of the best one, and result.json with the selected step. Each evaluation is a ``python -m policy.eval checkpoint``
process launched by policy.training.cloak.evaluator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from policy.data.cloak_dataset import CloakDataset, digest, local_path

from .evaluator import RunEvaluator, evaluate_run, report_name


def execute(recipe, backend, root, *, resume=False):
    root, backend = Path(root).resolve(), Path(backend).resolve()
    cfg = json.loads(Path(recipe).read_text())
    if cfg.get('schema') != 'actathon.cloak.experiment.v1':
        raise ValueError('Unknown experiment recipe')
    data, run_dir = (local_path(root, cfg[k]) for k in ('data', 'run_dir'))
    train, held = CloakDataset(data), CloakDataset(data, 'holdout')
    if train.manifest.get('task_group') != cfg['task_group']:
        raise ValueError('Experiment task does not match the prepared dataset')
    if any(e.get('group') != cfg['task_group'] for e in train.manifest['episodes']):
        raise ValueError('A separate task model cannot contain another task')
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = {'recipe': cfg, 'recipe_sha256': digest(recipe),
             'manifest_sha256': digest(data / 'manifest.json')}
    stamp_path = run_dir / 'experiment.json'
    if stamp_path.exists() and json.loads(stamp_path.read_text()) != stamp:
        raise ValueError('Experiment identity changed; choose a new run directory')
    stamp_path.write_text(json.dumps(stamp, indent=2) + '\n')
    common = ['--backend', str(backend), '--data', str(data), '--run-dir', str(run_dir)]
    flag = lambda key: '--' + key.replace('_', '-')
    # Booleans are store_true flags in the trainer's CLI: present when true, absent when false.
    options = [part for key, value in cfg['training'].items() if value is not False
               for part in ([flag(key)] if value is True else [flag(key), str(value)])]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1',
           'XLA_PYTHON_CLIENT_MEM_FRACTION': '.90'}
    def call(args, log):
        with (run_dir / log).open('a') as f:
            subprocess.run([sys.executable, '-m', 'policy.training.cloak', *args], cwd=root, env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=True)

    call(['train', *common, *options, '--check-data'], 'data-check.log')
    # Evaluations (python -m policy.eval checkpoint) run on one of this process's assigned GPUs, with no competing
    # trainer, each in a fresh process that releases parameters and compiled training buffers.
    evaluator = RunEvaluator(backend, data, run_dir, manifest_sha256=stamp['manifest_sha256'], cwd=root, env=env)

    checkpoint_root = run_dir / 'checkpoints' / 'excavator_cloak' / cfg['training']['mode']
    if not held.episodes:                       # holdout: none -> train on everything, keep the last step
        call(['train', *common, *options, *(['--resume'] if resume else [])], 'training.log')
        step = cfg['training']['steps']
        result = {'task_group': cfg['task_group'], 'selected_step': step, 'checkpoint': str(checkpoint_root / str(step)),
                  'manifest_sha256': stamp['manifest_sha256'], 'joint_order': train.joints,
                  'train_episodes': len(train.episodes), 'holdout_episodes': 0,
                  'selection_scope': 'No held-out recordings (holdout: none): every recording trains and the final step is used. No held-out action error.',
                  'limitations': train.manifest.get('limitations', [])}
        (run_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    _, initial = evaluator('initial', pretrained=True)
    call(['train', *common, *options, *(['--resume'] if resume else [])], 'training.log')
    run = evaluate_run(evaluator, steps=cfg['evaluate_steps'], mode=cfg['training']['mode'], pretrained=False,
                       stress='best', mask_seed=cfg['mask_seed'])
    best = run['best_step']
    reports, stress = run['reports'], run['reports'][report_name(best, masked=True)]
    result = {'task_group': cfg['task_group'], 'selected_step': best,
              'checkpoint': str(checkpoint_root / str(best)), 'manifest_sha256': stamp['manifest_sha256'],
              'joint_order': train.joints, 'train_episodes': len(train.episodes), 'holdout_episodes': len(held.episodes),
              'initial': initial['episode_mean_normalized_mae'],
              'selected': reports[report_name(best)]['episode_mean_normalized_mae'],
              'masked': stress['episode_mean_normalized_mae'],
              'selection_scope': 'Minimum whole-recording holdout action error among these checkpoints. Holdout was used for checkpoint selection; no untouched test set or physical task success measurement.',
              'limitations': train.manifest.get('limitations', [])}
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    print(json.dumps(execute(a.recipe, a.backend, a.root, resume=a.resume), indent=2), flush=True)


if __name__ == '__main__':
    main()
