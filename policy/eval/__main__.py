"""Offline evaluation of trained CLOAK / π0.5 runs (inside the umi-excavator-train image for checkpoint/run).

  checkpoint  one report: a trained checkpoint (or --pretrained pi05_base) of a run over a dataset split
  run         a run directory's reports: pi05_base, the given steps (or all saved), the mask stress test;
              one fresh process per report on the first visible GPU, existing reports of the same data reused
  report      summary table of one or more run directories' eval-*.json (no JAX needed)

Examples (in the container the backend is /opt/cloak and the repository is /workspace):
  python -m policy.eval run --backend /opt/cloak --data dataset/cohorts/<name>/frozen/<method> \\
      --run-dir dataset/runs/<name> --steps 250 500 --mask-seed 20261005
  python -m policy.eval checkpoint --backend /opt/cloak --data <cohort> --run-dir <run> \\
      --checkpoint <run>/checkpoints/excavator_cloak/<mode>/<step> --output <run>/eval-<step>.json
  python -m policy.eval report dataset/runs/<name> [--json]
"""
import argparse
import json
from pathlib import Path
import sys

from policy.configs.schema import EVALUATE_SPLITS as SPLITS, EVALUATE_STRESS as STRESS


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m policy.eval', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest='command', required=True)
    one = commands.add_parser('checkpoint', help='Write one report for a checkpoint or pi05_base')
    weights = one.add_mutually_exclusive_group(required=True)
    weights.add_argument('--checkpoint', type=Path)
    weights.add_argument('--pretrained', action='store_true', help='Evaluate initialization before fine-tuning')
    one.add_argument('--output', type=Path, required=True)
    one.add_argument('--mask-seed', type=int, help='Evaluate reproducible synthetic edge occlusions')
    one.add_argument('--policy-data', type=Path,
                     help='Dataset the run trained on, when --data is a different evaluation set; its train '
                          'split also sets the normalization and training-mean baseline')
    one.add_argument('--predictions', type=Path, help='Also save per-window predicted/target chunks (.npz)')
    run = commands.add_parser('run', help="Evaluate a run directory's checkpoints, reusing existing reports")
    run.add_argument('--steps', nargs='+', required=True, help="checkpoint steps, or 'all' saved ones")
    run.add_argument('--mode', help='training mode folder under checkpoints/excavator_cloak (default: the only one)')
    run.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='skip pi05_base')
    run.add_argument('--stress', choices=STRESS, default='best', help='mask stress test of the best step, all, none')
    run.add_argument('--mask-seed', type=int, help='seed of the stress test masks (required unless --stress none)')
    run.add_argument('--predictions', action='store_true', help='also save <report>-predictions.npz')
    for cmd in (one, run):
        cmd.add_argument('--backend', type=Path, required=True)
        cmd.add_argument('--data', type=Path, required=True)
        cmd.add_argument('--run-dir', type=Path, required=True)
        cmd.add_argument('--split', choices=SPLITS, default='holdout')
    rep = commands.add_parser('report', help='Summary table of eval-*.json reports')
    rep.add_argument('run_dirs', type=Path, nargs='+')
    rep.add_argument('--json', action='store_true')
    a = p.parse_args(argv)

    if a.command == 'checkpoint':
        from .cloak import evaluate_checkpoint
        evaluate_checkpoint(a.backend, a.data, a.run_dir, a.output, checkpoint=a.checkpoint, pretrained=a.pretrained,
                            mask_seed=a.mask_seed, policy_data=a.policy_data, predictions=a.predictions,
                            split=a.split)
        from .report import row
        print(json.dumps(row(a.output.stem, json.loads(a.output.read_text())), indent=2))
    elif a.command == 'run':
        from policy.training.cloak.evaluator import RunEvaluator, evaluate_run
        from .report import aggregate, format_table
        if a.stress != 'none' and a.mask_seed is None:
            p.error('--mask-seed is required unless --stress none')
        steps = 'all' if a.steps == ['all'] else [int(s) for s in a.steps]
        evaluator = RunEvaluator(a.backend.resolve(), a.data.resolve(), a.run_dir.resolve(), split=a.split,
                                 predictions=a.predictions)
        result = evaluate_run(evaluator, steps=steps, mode=a.mode, pretrained=a.pretrained, stress=a.stress,
                              mask_seed=a.mask_seed)
        if result['stress_skipped']:
            print(result['stress_skipped'], file=sys.stderr)
        print(format_table(aggregate(a.run_dir)))
        print(json.dumps({'reports': sorted(result['reports']), 'best_step': result['best_step'],
                          'stress_skipped': result['stress_skipped']}))
    else:
        from .report import aggregate, format_table
        summaries = [aggregate(d) for d in a.run_dirs]
        print(json.dumps(summaries, indent=2) if a.json else format_table(summaries))


if __name__ == '__main__':
    main()
