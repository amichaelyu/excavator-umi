"""Export and serve a portable checkpoint without optimizer state or episode payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from policy.data.cloak_dataset import digest, local_path

from . import ASSET, UPSTREAM_REVISION, UPSTREAM_URL


def requirements():
    """Serving requirements shipped in each bundle: the repository's dependencies plus its cloak group."""
    import tomllib
    doc = tomllib.loads((Path(__file__).resolve().parents[3] / 'pyproject.toml').read_text())
    return '\n'.join(['# umi-excavator pyproject.toml: [project].dependencies + [dependency-groups].cloak',
                      *doc['project']['dependencies'], *doc['dependency-groups']['cloak']]) + '\n'


def export_bundle(data, run_dir, checkpoint, output):
    data, run_dir, checkpoint, output = (Path(p).resolve() for p in (data, run_dir, checkpoint, output))
    record = json.loads((run_dir / 'cloak-run.json').read_text())
    if (record['upstream_revision'] != UPSTREAM_REVISION
            or record['manifest_sha256'] != digest(data / 'manifest.json')):
        raise ValueError('Checkpoint run and frozen dataset do not match')
    mode = record['options']['mode']
    expected = run_dir / 'checkpoints' / 'excavator_cloak' / mode
    if checkpoint.parent != expected or not checkpoint.name.isdigit():
        raise ValueError('Checkpoint must belong to this run')
    if not (checkpoint / '_CHECKPOINT_METADATA').is_file():
        raise ValueError('Checkpoint save has not completed')
    stats = checkpoint / 'assets' / ASSET / 'norm_stats.json'
    if digest(stats) != digest(run_dir / 'assets' / ASSET / 'norm_stats.json'):
        raise ValueError('Checkpoint normalization differs from this run')
    if not (checkpoint / 'params' / '_METADATA').is_file():
        raise ValueError('Checkpoint parameters are missing')
    # Do not export paths that would keep a dependency on the original training host.
    portable_record = {**record, 'options': {**record['options'], 'initial_checkpoint': None}}
    manifest = json.loads((data / 'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data').mkdir()
    (output / 'run').mkdir()
    shutil.copy2(data / 'manifest.json', output / 'data/manifest.json')
    (output / 'run/cloak-run.json').write_text(json.dumps(portable_record, indent=2) + '\n')
    relative_checkpoint = Path('run/checkpoints/excavator_cloak') / mode / checkpoint.name
    target = output / relative_checkpoint
    target.mkdir(parents=True)
    shutil.copytree(checkpoint / 'params', target / 'params')
    shutil.copytree(checkpoint / 'assets', target / 'assets')
    shutil.copytree(checkpoint / 'assets', output / 'run/assets')
    evaluation = run_dir / f'eval-{checkpoint.name}.json'
    if evaluation.is_file():
        shutil.copy2(evaluation, output / 'evaluation.json')
    (output / 'requirements.txt').write_text(requirements())
    files = {str(p.relative_to(output)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(output.rglob('*')) if p.is_file()}
    metadata = {'schema': 'actathon.cloak.bundle.v1', 'upstream_url': UPSTREAM_URL,
                'upstream_revision': UPSTREAM_REVISION, 'base_weights': 'gs://openpi-assets/checkpoints/pi05_base',
                'task_group': manifest.get('task_group'), 'step': int(checkpoint.name),
                'data': 'data', 'run_dir': 'run', 'checkpoint': str(relative_checkpoint),
                'action_contract': manifest['episodes'][0]['action_contract'],
                'action_horizon': record['options']['horizon'], 'mode': mode,
                'source_visibility': manifest.get('source_visibility'),
                'limitations': manifest.get('limitations', []),
                'contents': 'Full model parameters and normalization. No optimizer state or demonstration payloads.',
                'files': files, 'payload_bytes': sum(f['bytes'] for f in files.values())}
    (output / 'bundle.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def verify_bundle(root):
    root = Path(root).resolve()
    metadata = json.loads((root / 'bundle.json').read_text())
    if metadata.get('schema') != 'actathon.cloak.bundle.v1' or metadata['upstream_revision'] != UPSTREAM_REVISION:
        raise ValueError('Unsupported bundle or backend revision')
    for name, expected in metadata['files'].items():
        path = local_path(root, name)
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Bundle payload mismatch: {name}')
    required = {'data/manifest.json', 'run/cloak-run.json', f'run/assets/{ASSET}/norm_stats.json',
                f"{metadata['checkpoint']}/params/_METADATA",
                f"{metadata['checkpoint']}/assets/{ASSET}/norm_stats.json"}
    if not required.issubset(metadata['files']):
        raise ValueError('Bundle inventory is missing required inference files')
    return metadata


def load_bundle(backend, root):
    from .policy import load_policy
    root = Path(root).resolve()
    metadata = verify_bundle(root)
    return load_policy(backend, local_path(root, metadata['data']), local_path(root, metadata['run_dir']),
                       local_path(root, metadata['checkpoint']))


def load_base_bundle(backend, root):
    """Public pi05_base weights under a verified bundle's data, run config, norm stats and prompts.

    This is the no-fine-tune ablation arm: everything except the parameters
    comes from the bundle, so it should be paired with the bundle that has no
    CLOAK masking (the task's baseline) for a fair comparison.
    """
    from .policy import load_policy
    root = Path(root).resolve()
    metadata = verify_bundle(root)
    policy = load_policy(backend, local_path(root, metadata['data']), local_path(root, metadata['run_dir']),
                         pretrained=True)
    if policy.metadata.get('weights') != 'pretrained_initialization':
        raise ValueError('Base serving must report pretrained_initialization weights')
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    for key in ('data', 'run-dir', 'checkpoint', 'output'):
        export.add_argument('--' + key, type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--bundle', type=Path, required=True)
    serve = commands.add_parser('serve', help='Local prediction server, no motor commands')
    serve.add_argument('--backend', type=Path, required=True)
    serve.add_argument('--bundle', type=Path, required=True)
    serve.add_argument('--port', type=int, default=8010)
    base = commands.add_parser('serve-base', help='Serve public pi05_base weights with a bundle\'s config '
                               'and normalization (no fine-tune ablation), no motor commands')
    base.add_argument('--backend', type=Path, required=True)
    base.add_argument('--bundle', type=Path, required=True)
    base.add_argument('--port', type=int, default=8020)
    args = parser.parse_args()
    if args.command in ('serve', 'serve-base'):
        from .server import serve_policy
        load = load_bundle if args.command == 'serve' else load_base_bundle
        serve_policy(load(args.backend, args.bundle), args.port)
        return
    result = (export_bundle(args.data, args.run_dir, args.checkpoint, args.output)
              if args.command == 'export' else verify_bundle(args.bundle))
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
