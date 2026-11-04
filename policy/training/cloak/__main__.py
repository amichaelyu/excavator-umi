"""CLOAK π0.5 on handheld-bucket demonstrations: robot masks for a frozen cohort, then train and serve.

Examples, with the pinned Stanford backend and the cloak dependency group installed:
  python -m policy.training.cloak render-masks --data <cohort> --config <geometry> --embodiment <name> --image-geometry letterbox224 --output <masks>
  python -m policy.training.cloak prepare --data <cohort> --masks <masks> --output <prepared>
  python -m policy.training.cloak train --backend dataset/cloak-research/upstream --data <prepared> --run-dir dataset/runs/<name> --check-data

Held-out evaluation of a run's checkpoints is ``python -m policy.eval``. ``train`` defaults are
policy.configs.schema.TRAIN_DEFAULTS, the same values ``python -m policy train <config>`` fills in.

render-masks writes reviewed=false until the overlays have been checked. The
default self_mask.json is unmeasured and is deliberately rejected for training.
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
    render = commands.add_parser('render-masks', help='Render per-frame masks from measured source geometry')
    render.add_argument('--data', type=Path, required=True)
    render.add_argument('--config', type=Path, required=True)
    render.add_argument('--embodiment', required=True)
    render.add_argument('--image-geometry', choices=['native', 'letterbox224'], required=True)
    render.add_argument('--output', type=Path, required=True)
    calibrate = commands.add_parser('calibrate', help='Fit camera mount to supplied silhouette annotations')
    calibrate.add_argument('--config', type=Path, required=True)
    calibrate.add_argument('--annotations', type=Path, required=True,
                           help='NPZ: masks [N,H,W] bool and state [N,2] in twin radians')
    calibrate.add_argument('--output', type=Path, required=True, help='New fitted config; remains unmeasured until reviewed')
    calibrate.add_argument('--iterations', type=int, default=200)
    train = commands.add_parser('train', help='Fine-tune using the released CLOAK attention implementation')
    serve = commands.add_parser('serve', help='Serve a CLOAK checkpoint locally, without motor control')
    for cmd in (train, serve):
        cmd.add_argument('--backend', type=Path, required=True)
        cmd.add_argument('--data', type=Path, required=True)
        cmd.add_argument('--run-dir', type=Path, required=True)
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
    serve.add_argument('--checkpoint', type=Path, required=True)
    serve.add_argument('--port', type=int, default=8010)
    serve.add_argument('--geometry-config', type=Path)
    serve.add_argument('--source-profile', help='Enable retargeting from this source twin profile')
    serve.add_argument('--target-profile')
    serve.add_argument('--source-seed', type=float, nargs=2)
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
    elif a.command == 'render-masks':
        from .geometry import render_masks
        result = render_masks(a.data, a.config, a.output, embodiment=a.embodiment, image_geometry=a.image_geometry)
        result = {'episodes': len(result['episodes']), 'reviewed': False, 'output': str(a.output)}
    elif a.command == 'train':
        from .training import run
        result = run(a.backend, a.data, a.run_dir, check_data=a.check_data, stop_at=a.stop_at,
                     **{k: getattr(a, k) for k in ('mode', 'steps', 'batch', 'fsdp', 'seed', 'horizon', 'resume', 'initial_checkpoint', 'host_optimizer', 'microbatch', 'save_interval', 'keep_period', 'action_representation', 'peak_lr', 'warmup')})
    elif a.command == 'calibrate':
        import numpy as np
        from digital_twin.sim.self_mask import SelfMasker
        from .calibration import fit_mount
        from policy.data.cloak_dataset import digest
        if a.output.exists():
            p.error('Use a new output path; preserve the prior calibration')
        cfg = json.loads(a.config.read_text())
        if cfg.get('state_joints') != ['boom', 'bucket'] or not cfg.get('source_embodiment'):
            p.error('Config needs source_embodiment and state_joints [boom,bucket]')
        with np.load(a.annotations, allow_pickle=False) as z:
            states, masks = z['state'], z['masks']
        if states.shape != (len(masks), 2) or not np.isfinite(states).all():
            p.error('Invalid annotation states')
        w, h = cfg['intrinsics']['image_size']
        if masks.shape[1:] != (h, w):
            p.error('Annotation masks must match native calibrated camera dimensions')
        renderer = SelfMasker(config={**cfg, 'dilate_px': 0})
        try:
            result = fit_mount(renderer, [dict(zip(cfg['state_joints'], q)) for q in states], masks,
                               max_iterations=a.iterations)
        finally:
            renderer.close()
        cfg.update(mount=result['mount'], measured=False, calibration_fit=result,
                   calibration_evidence={'annotations_sha256': digest(a.annotations)})
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(cfg, indent=2) + '\n')
    else:
        from .policy import load_policy, GeometryPolicy
        if a.source_profile and not (a.target_profile and a.source_seed and a.geometry_config):
            p.error('Retargeting needs source/target profiles, source seed and geometry config')
        if (a.target_profile or a.source_seed) and not a.source_profile:
            p.error('Retargeting options require --source-profile')
        policy = load_policy(a.backend, a.data, a.run_dir, a.checkpoint)
        if a.geometry_config:
            retargeter = None
            if a.source_profile:
                from .geometry import BucketRetargeter
                retargeter = BucketRetargeter(a.source_profile, a.target_profile)
            policy = GeometryPolicy(policy, a.geometry_config, retargeter, a.source_seed)
        # Use the repository's OpenPI-compatible wire protocol. Binding is local.
        from .server import serve_policy
        try:
            serve_policy(policy, a.port)
        finally:
            if isinstance(policy, GeometryPolicy):
                policy.close()
        return
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
