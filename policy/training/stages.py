"""Training stages. cloak: reconstruct (here, in Docker or on Lambda), then train, status, serve, stop on Lambda
(select, labels and cohort are policy.preprocessing).
Run through ``python -m policy <stage> <config>`` (policy/stages.py orchestrates).

Each stage reads only the config and the previous stage's output directory, and skips work whose
output already exists, so a stage can be rerun after a failure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import sys
import time

from policy.configs import ROOT
from policy.configs.schema import LABEL_PROFILES, RECONSTRUCT_RUNNERS, TASKS
from policy import preprocessing
from policy.data import local_config

from . import lambda_host
from .cloak import wandb_run
from .lambda_host import Host

# The policy rollout app's checkpoint registry (a path, not an import: serve --register
# runs on the machine that runs the hub).
REGISTRY = ROOT / 'configs' / 'hub' / 'local_checkpoints.json'


_local = preprocessing._local


def reconstruct(cfg, dry_run=False, address=None, runner=None):
    """Camera trajectories for every selected recording (``<output>/<id>/trajectory.csv``).

    ``reconstruct.runner`` picks where: ``local`` (this Python; ORB-SLAM3 itself runs in the
    umi-excavator-processing image through policy/processing/docker/run.sh), ``docker``
    (everything in the umi-excavator-processing image on this machine) or ``lambda`` (that image on lambda.host,
    or ``address``; inputs are synced there and the output pulled back). ``runner`` overrides the config.
    """
    rc = cfg.section('reconstruct')
    runner = runner or rc.get('runner', 'local')
    if runner not in RECONSTRUCT_RUNNERS:
        raise ValueError(f'reconstruct.runner must be one of {RECONSTRUCT_RUNNERS}, not {runner!r}')
    for data_root, manifest, ids in preprocessing.reconstruct_inputs(cfg, dry_run):
        paths = {'manifest': manifest, 'calibration': cfg.calibration, 'data_root': data_root,
                 'output': cfg.dataset('reconstructions'),
                 'work': Path('.cache/reconstruct') / cfg.name}
        _reconstruct(cfg, paths, ids, runner, dry_run, address)


def _reconstruct(cfg, paths, ids, runner, dry_run, address):
    """One data root's selected recordings (``paths['manifest']``) through the configured runner."""
    rc = cfg.section('reconstruct')

    def command(p):
        return ['-m', 'policy.processing.reconstruct_orbslam', '--manifest', p['manifest'], '--calibration',
                p['calibration'], '--data-root', p['data_root'], '--output', p['output'], '--workers',
                rc.get('workers', 8), '--work', p['work']]

    if runner == 'local':
        _local([sys.executable, *command(paths)], dry_run)
    elif runner == 'docker':
        _reconstruct_docker(paths, command, dry_run)
    else:
        _reconstruct_lambda(cfg, paths, command, host_for(cfg, address, dry_run), ids)


def _reconstruct_docker(paths, command, dry_run):
    """The processing image on this machine, the repository at /workspace. Inputs and outputs outside the
    repository (or reached through a symlink out of it) are mounted at their own absolute paths."""
    tag = lambda_host.processing_image_tag()
    _local(['sh', '-c', lambda_host.processing_build_command(tag)], dry_run)
    mounts, args = ['-v', f'{ROOT}:/workspace'], {}
    for key, path in paths.items():
        real = (ROOT / path).resolve()
        if real.is_relative_to(ROOT):
            args[key] = real.relative_to(ROOT)
        else:
            writable = key in ('output', 'work')
            if writable and not dry_run:
                real.mkdir(parents=True, exist_ok=True)   # else Docker creates it owned by root
            mounts += ['-v', f'{real}:{real}' + ('' if writable else ':ro')]
            args[key] = real
    user = f'{os.getuid()}:{os.getgid()}'
    _local(['docker', 'run', '--rm', '--init', '--ipc=host', '--network=none', '--user', user, *mounts,
            '-w', '/workspace', tag, 'python', *command(args)], dry_run)


def _reconstruct_lambda(cfg, paths, command, host, ids=None):
    """Sync policy/processing/ and the inputs, build the processing image on the box, run it there, pull the output.
    Inputs outside the repository go to .cache/inputs/<name>/ on the box. Of the data root only the selected
    recordings (``ids``) are synced: outputs can live beside them (dataset/reconstructions, dataset/runs, ...)."""
    args = {}
    for key in ('manifest', 'calibration', 'data_root'):
        real = (ROOT / paths[key]).resolve()
        args[key] = real.relative_to(ROOT) if real.is_relative_to(ROOT) else Path('.cache/inputs') / cfg.name / real.name
    args['output'], args['work'] = paths['output'], paths['work']
    host.push_code(lambda_host.PROCESSING_CODE)
    for key in ('manifest', 'calibration', 'data_root'):
        host.push_data(paths[key], args[key], only=ids if key == 'data_root' else None)
    tag = host.ensure_processing_image()
    host.docker_run(_container('reconstruct', cfg), ['python', *map(str, command(args))], detach=False,
                    replace=True, image=tag)
    host.pull(paths['output'])


def lambda_address(cfg, address=None):
    """The Lambda box: ``--host``, else the config's lambda.host, else hosts.lambda.host of configs/hub/local.json."""
    return address or cfg.section('lambda').get('host') or local_config.lambda_host()


def lambda_root(cfg):
    """The checkout on the Lambda box: lambda.root, else hosts.lambda.root (default ~/umi-excavator)."""
    return cfg.section('lambda').get('root') or local_config.lambda_root()


def host_for(cfg, address=None, dry_run=False):
    address = lambda_address(cfg, address)
    if not address:
        raise ValueError('Set hosts.lambda.host in configs/hub/local.json (or lambda.host in the config), '
                         'or pass --host ubuntu@IP')
    return Host(address, lambda_root(cfg), dry_run)


def _container(prefix, cfg):
    return f'{prefix}-{cfg.name}'


def _running(host, name):
    out = host.ssh(f'docker ps -q --filter name=^{name}$', capture=True)
    return bool(out and out.strip())


def trained(cfg, host=None):
    """True once the run has its ``result.json`` here, or on the box (then copied back)."""
    if (ROOT / cfg.run_dir / 'result.json').is_file():
        return True
    if host is None or host.dry_run:
        return False
    if host.ssh(f'test -f {shlex.quote(str(cfg.run_dir / "result.json"))}', check=False) != 0:
        return False
    host.pull(cfg.run_dir)
    return True


def train(cfg, host, resume=False, wait=False, poll=None):
    """Sync code and the frozen cohort, then run policy.training.cloak.experiment in the image, detached.

    A finished run (``result.json``) is skipped unless ``resume``. ``wait`` (``python -m policy all``) follows
    the container until it exits, polling every ``lambda.poll`` seconds, and copies the results back; a run
    already in progress is then waited on instead of refused.

    ``train.wandb`` on: ``<run_dir>/wandb-run.json`` (settings and the stable W&B run id) goes to the box with the
    run directory, and WANDB_API_KEY / _ENTITY / _MODE / _BASE_URL from this shell into the container. A run
    whose mode is online stops here, before anything is synced, when WANDB_API_KEY is not set.
    """
    name = _container('train', cfg)
    if not resume and trained(cfg, host):
        print(f'{cfg.run_dir} is already trained (result.json); skipping train', file=sys.stderr)
        return
    if not host.dry_run and _running(host, name):
        if not wait:
            raise RuntimeError(f'{name} is already running on {host.address}; `stop` it first')
        print(f'{name} is already running on {host.address}; waiting for it', file=sys.stderr)
        return wait_for(cfg, host, poll)
    tracking = cfg.wandb
    tracked = wandb_run.check(tracking, package=False) != 'disabled'
    recipe = cfg.run_dir / 'recipe.json'
    # A run that tracked before keeps its record (and id) with the config's current mode, e.g. disabled.
    track_file = tracking['mode'] != 'disabled' or (ROOT / cfg.run_dir / wandb_run.FILE).is_file()
    if host.dry_run:
        print(f'# write {recipe}:\n' + json.dumps(cfg.recipe(), indent=2), file=sys.stderr)
        if track_file:
            print(f'# write {cfg.run_dir / wandb_run.FILE} (W&B, + its stable run id): {json.dumps(tracking)}',
                  file=sys.stderr)
    else:
        (ROOT / cfg.run_dir).mkdir(parents=True, exist_ok=True)
        (ROOT / recipe).write_text(json.dumps(cfg.recipe(), indent=2) + '\n')
        if track_file:
            wandb_run.write(ROOT / cfg.run_dir, tracking)
    host.push_code()
    host.push_data(cfg.cohort_data)
    host.push_data(cfg.run_dir)
    host.ensure_image()
    host.docker_run('fetch-pi05-base', ['python', '-m', 'policy.training.weights'], detach=False, replace=True)
    gpus = cfg.section('lambda').get('gpus', [0])
    env = lambda_host.forwarded_env() if tracked else {}
    host.docker_run(name, ['python', '-m', 'policy.training.cloak.experiment', '--recipe', str(recipe),
                           '--backend', '/opt/cloak', *(['--resume'] if resume else [])], gpus=gpus, replace=True,
                    **({'env': env} if env else {}))
    print(f'Started {name} on {host.address} (GPUs {gpus}). Follow it with: python -m policy status {_arg(cfg)}', flush=True)
    if wait:
        wait_for(cfg, host, poll)


def wait_for(cfg, host, poll=None):
    """Poll the train container until it exits, copy the run back and fail if training failed."""
    name = _container('train', cfg)
    poll = poll or cfg.section('lambda').get('poll', 60)
    print(f'# wait for {name} on {host.address} to exit (every {poll} s), then pull {cfg.run_dir}', file=sys.stderr)
    if host.dry_run:
        return
    while True:
        state = (host.ssh(f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {name}",
                          capture=True, check=False) or '').split()
        if len(state) != 2:
            raise RuntimeError(f'{name} is not on {host.address}')
        if state[0] not in ('running', 'created', 'restarting'):
            break
        time.sleep(poll)
    host.pull(cfg.run_dir)
    if state[1] != '0' or not (ROOT / cfg.run_dir / 'result.json').is_file():
        raise RuntimeError(f'{name} exited with code {state[1]} and no result.json; see python -m policy status {_arg(cfg)}')


def _arg(cfg):
    """How to name this config on the command line."""
    from policy.configs.schema import CONFIGS
    return cfg.path.stem if cfg.path.parent == CONFIGS else str(cfg.path)


def status(cfg, host):
    """Container state, the training log tail, and the evaluations so far (copied back locally)."""
    for prefix in ('train', 'serve'):
        name = _container(prefix, cfg)
        host.ssh(f'docker ps -a --filter name=^{name}$ --format "{name}: {{{{.Status}}}}"', check=False)
    host.ssh(f'tail -n 5 {shlex.quote(str(cfg.run_dir / "training.log"))} 2>/dev/null', check=False)
    host.pull(cfg.run_dir)
    run = ROOT / cfg.run_dir
    for path in sorted(run.glob('eval-*.json'), key=lambda p: (len(p.stem), p.stem)):
        mae = json.loads(path.read_text()).get('episode_mean_normalized_mae', {})
        print(f'{path.stem:24s} policy {mae.get("policy", float("nan")):.3f}  '
              + '  '.join(f'{k} {v:.3f}' for k, v in mae.items() if k != 'policy' and isinstance(v, float)))
    if (run / 'result.json').is_file():
        result = json.loads((run / 'result.json').read_text())
        print(f'done: selected step {result["selected_step"]}')


def serve(cfg, host, register=False):
    """Export a bundle of the chosen step and serve it on the box (127.0.0.1:<port>, reach it with ssh -L)."""
    if register:                                                    # before starting anything on the box
        check_registrable(cfg)
    if register and not host.dry_run and not REGISTRY.is_file():
        raise FileNotFoundError(f'{REGISTRY}: no hub checkpoint registry here; register from the hub machine '
                                '(create it there from configs/hub/local_checkpoints.example.json)')
    srv = cfg.section('serve')
    step, port = srv.get('step', 'best'), srv.get('port', 8020)
    if step == 'best':
        host.pull(cfg.run_dir)
        result = ROOT / cfg.run_dir / 'result.json'
        if result.is_file():
            step = json.loads(result.read_text())['selected_step']
        elif host.dry_run:
            step = '<selected_step>'
        else:
            raise FileNotFoundError('No result.json yet: training is not finished; set serve.step to a saved step')
    mode = cfg.training['mode']
    checkpoint = cfg.run_dir / 'checkpoints' / 'excavator_cloak' / mode / str(step)
    bundle = cfg.run_dir / f'bundle-{step}'
    script = (f'test -f {bundle}/bundle.json || python -m policy.training.cloak.bundle export --data {cfg.cohort_data} '
              f'--run-dir {cfg.run_dir} --checkpoint {checkpoint} --output {bundle} && '
              f'exec python -m policy.training.cloak.bundle serve --backend /opt/cloak --bundle {bundle} --port {port}')
    host.ensure_image()
    host.docker_run(_container('serve', cfg), ['sh', '-c', script], gpus=srv.get('gpus', [0]), ports=[port],
                    replace=True)
    print(f'Serving step {step} on {host.address}:{port}. Tunnel: ssh -N -L {port}:127.0.0.1:{port} {host.address}')
    entry = checkpoint_entry(cfg, host, step, port)
    if register and not host.dry_run:
        if not REGISTRY.is_file():
            raise FileNotFoundError(f'{REGISTRY}: no hub checkpoint registry here; register from the hub machine '
                                    '(create it there from configs/hub/local_checkpoints.example.json)')
        doc = json.loads(REGISTRY.read_text())
        doc['checkpoints'] = [c for c in doc['checkpoints'] if c['id'] != entry['id']] + [entry]
        REGISTRY.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
        print(f'Registered {entry["id"]} in {REGISTRY.relative_to(ROOT)}')
    else:
        print(json.dumps(entry, indent=2, ensure_ascii=False))


def check_registrable(cfg):
    """Refuse ``serve --register`` for a policy the hub's registry would not accept: it serves only labels in the
    registry's policy space (digital_twin.policy.checkpoints.CONTRACT, calibrated), so e.g. labels.profile full4
    (handheld_reconstructed) cannot be registered."""
    from digital_twin.policy.checkpoints import CONTRACT
    profile = cfg.section('labels').get('profile', 'full4_calibrated')
    space = LABEL_PROFILES[profile]
    if space != CONTRACT[2]:
        accepted = [p for p, s in LABEL_PROFILES.items() if s == CONTRACT[2]]
        raise ValueError(f'serve --register: labels.profile {profile} trains {space} joints, but the hub checkpoint '
                         f'registry ({REGISTRY}) only accepts {CONTRACT[2]} policies '
                         f'(labels.profile {" or ".join(accepted)}). Serve without --register.')


def checkpoint_entry(cfg, host, step, port):
    """The policy rollout app's registry entry (configs/hub/local_checkpoints.json) for this server.
    The box address stays out of it: the server is reached through the SSH port forward on the hub machine."""
    profile = cfg.section('labels').get('profile', 'full4_calibrated')
    ident = f'{cfg.name}-{step}'
    entry = {'id': ident, 'task': cfg.task, 'mode': cfg.training['mode'], 'prompt': TASKS[cfg.task], 'step': step,
             'gateway_url': f'ws://127.0.0.1:{port}',
             'gateway_env': 'UMI_EXCAVATOR_' + ident.upper().replace('-', '_') + '_URL'}
    if LABEL_PROFILES[profile] != 'calibrated':      # the registry skips it; serve --register refuses it
        entry['policy_space'] = LABEL_PROFILES[profile]
    return entry


def stop(cfg, host, which=('train', 'serve')):
    for prefix in which:
        host.ssh(f'docker rm -f {_container(prefix, cfg)}', check=False)
