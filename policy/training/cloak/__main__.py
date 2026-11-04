"""CLOAK π0.5 on handheld-bucket demonstrations: join robot masks to a frozen cohort, then train.

Examples, with the pinned Stanford backend and the training dependency group installed:
  python -m policy.training.cloak.twin_masks --data <cohort> --output <masks>
  python -m policy.training.cloak prepare --data <cohort> --masks <masks> --output <prepared>
  python -m policy.training.cloak train --backend dataset/cloak-research/upstream --data <prepared> --run-dir dataset/runs/<name> --check-data

Held-out evaluation of a run's checkpoints is ``python -m policy.eval``. ``train`` defaults are
policy.configs.schema.TRAIN_DEFAULTS, the same values ``python -m policy train <config>`` fills in.
Serving a trained checkpoint is ``python -m policy.training.cloak.bundle serve``.
"""
import argparse
import json
from pathlib import Path

from policy.configs.schema import ACTION_REPRESENTATIONS, MODES, TRAIN_DEFAULTS as D


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Join reviewed geometric masks to a frozen handheld export')
    prepare.add_argument('--data', type=Path, required=True)
    prepare.add_argument('--masks', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    oov = commands.add_parser('prepare-out-of-view', help='Prepare a confirmed invisible source bucket')
    oov.add_argument('--data', type=Path, required=True)
    oov.add_argument('--output', type=Path, required=True)
    oov.add_argument('--embodiment', required=True)
    oov.add_argument('--evidence', type=Path, required=True, help='JSON with confirmation and source_manifest_sha256')
    oov.add_argument('--content-box', type=int, nargs=4, required=True, metavar=('X0', 'Y0', 'X1', 'Y1'))
    train = commands.add_parser('train', help='Fine-tune using the released CLOAK attention implementation')
    train.add_argument('--backend', type=Path, required=True)
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--run-dir', type=Path, required=True)
    train.add_argument('--mode', choices=MODES, default=D['mode'])
    train.add_argument('--action-representation', choices=ACTION_REPRESENTATIONS, default=D['action_representation'],
                       help='delta uses the released joint-training transform; served outputs remain absolute')
    train.add_argument('--steps', type=int, default=D['steps'])
    train.add_argument('--batch', type=int, default=D['batch'])
    train.add_argument('--fsdp', type=int, default=D['fsdp'])
    train.add_argument('--seed', type=int, default=D['seed'])
    train.add_argument('--horizon', type=int, default=D['horizon'])
    train.add_argument('--resume', action='store_true')
    train.add_argument('--initial-checkpoint', type=Path)
    train.add_argument('--check-data', action='store_true')
    train.add_argument('--host-optimizer', action='store_true', help='Keep full float32 Adam updates on CPU for smaller GPUs')
    train.add_argument('--microbatch', type=int, default=1, help='GPU microbatch size with --host-optimizer')
    train.add_argument('--save-interval', type=int, default=D['save_interval'])
    train.add_argument('--keep-period', type=int, default=D['keep_period'], help='Retain these checkpoints plus the latest one')
    train.add_argument('--peak-lr', type=float, default=D['peak_lr'],
                       help='cosine schedule peak (decays to peak/10); 20261013 retrain: 1e-4 at batch 128')
    train.add_argument('--warmup', type=int, default=D['warmup'], help='warmup steps (capped at steps/10)')
    train.add_argument('--stop-at', type=int, help='Save and stop at this step for evaluation; preserve the full LR schedule')
    return p


def main():
    p = build_parser()
    a = p.parse_args()
    if a.command == 'prepare':
        from policy.data.cloak_dataset import prepare
        result = prepare(a.data, a.masks, a.output)
        result = {'episodes': len(result['episodes']), 'output': str(a.output)}
    elif a.command == 'prepare-out-of-view':
        from policy.data.cloak_dataset import prepare_out_of_view
        result = prepare_out_of_view(a.data, a.output, embodiment=a.embodiment,
                                     evidence=json.loads(a.evidence.read_text()), content_box=a.content_box)
        result = {'episodes': len(result['episodes']), 'output': str(a.output), 'source_visibility': 'out_of_view'}
    elif a.command == 'train':
        from .training import run
        result = run(a.backend, a.data, a.run_dir, check_data=a.check_data, stop_at=a.stop_at,
                     **{k: getattr(a, k) for k in ('mode', 'steps', 'batch', 'fsdp', 'seed', 'horizon', 'resume', 'initial_checkpoint', 'host_optimizer', 'microbatch', 'save_interval', 'keep_period', 'action_representation', 'peak_lr', 'warmup')})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
