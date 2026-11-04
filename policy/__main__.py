"""The training hub: one config, every stage. Recordings in, served policy out.

  python -m policy <stage> <config> [options]

<config> is a name in configs/policy/ (dirt-orb) or a path to a YAML file. Everything a stage does comes from
the config (policy/configs/schema.py lists and validates every key); the options below are one-offs.

kind: cloak (π0.5 fine-tune, plain or CLOAK-masked, on handheld recordings), in pipeline order:
  reconstruct   ORB-SLAM3+smoother trajectories              reconstruct.runner: local | docker | lambda
  labels        twin IK replays -> four-joint action labels  labels.runner: local | docker
  cohort        export, embodiment masks, frozen split       cohort.runner: local | docker
  train         π0.5 / CLOAK experiment, detached            Lambda (lambda.host)
  eval          evaluate the trained run (policy.eval)       evaluate.runner: lambda | local
  serve         bundle and serve the selected step           Lambda; --register adds it to policy rollout
  all           reconstruct -> labels -> cohort -> train (waits for it) -> eval; finished work is skipped
utilities:
  check         validate the config and print the plan: where each stage runs, its inputs and outputs
  status        train / serve container state, log tail, holdout evaluations (copied back)
  stop          remove the config's train and serve containers
  recipe        print the experiment recipe the config produces
  image         build the training image on the Lambda box (train / serve do this when needed)
"""
import argparse
import sys

from policy import configs
from policy import stages


def steps_arg(value):
    if value == 'all':
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value!r} is not a step number or all') from None


def parser():
    p = argparse.ArgumentParser(prog='python -m policy', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=stages.COMMANDS, metavar='stage', help=' | '.join(stages.COMMANDS))
    p.add_argument('config', help='a name in configs/policy/ (dirt-orb) or a path to a YAML config')
    p.add_argument('--dry-run', action='store_true', help='print every command without running it')
    p.add_argument('--host', help='override lambda.host, e.g. ubuntu@203.0.113.7 (every Lambda stage)')
    p.add_argument('--runner', help="override this stage's runner (reconstruct: local|docker|lambda; "
                                    'labels, cohort: local|docker; eval: lambda|local)')
    p.add_argument('--docker', action='store_true', help='same as --runner docker')
    p.add_argument('--resume', action='store_true', help='train: continue from the latest checkpoint in run_dir')
    p.add_argument('--register', action='store_true', help='serve: add the endpoint to configs/hub/local_checkpoints.json')
    p.add_argument('--steps', type=steps_arg, nargs='+', metavar='STEP',
                   help='eval: these saved steps, or `all` (every saved checkpoint), instead of evaluate.steps')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.docker:
        if a.runner not in (None, 'docker'):
            p.error('--docker conflicts with --runner ' + a.runner)
        a.runner = 'docker'
    if a.steps and 'all' in a.steps:
        if len(a.steps) > 1:
            p.error('--steps all takes no step numbers')
        a.steps = 'all'
    try:
        cfg = configs.load(a.config)
        stages.run(cfg, a.stage, dry_run=a.dry_run, host=a.host, runner=a.runner, resume=a.resume,
                   register=a.register, steps=a.steps)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f'python -m policy {a.stage}: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
