"""serve-base: public pi05_base weights under a bundle's config. No GPU or backend needed."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from policy.training.cloak import ASSET, UPSTREAM_REVISION, bundle
from policy.data.cloak_dataset import digest


def make_bundle(tmp):
    source = tmp / 'source'
    data, run = source / 'data', source / 'run'
    data.mkdir(parents=True)
    run.mkdir()
    (data / 'manifest.json').write_text(json.dumps({'episodes': [{'action_contract': {
        'joints': ['slew', 'boom', 'stick', 'bucket']}}], 'task_group': 'dirt'}))
    (run / 'cloak-run.json').write_text(json.dumps({'upstream_revision': UPSTREAM_REVISION,
        'manifest_sha256': digest(data / 'manifest.json'), 'options': {
            'mode': 'baseline', 'horizon': 10, 'initial_checkpoint': None}}))
    checkpoint = run / 'checkpoints/excavator_cloak/baseline/1000'
    (checkpoint / 'params').mkdir(parents=True)
    (checkpoint / 'params/_METADATA').write_text('{}')
    (checkpoint / '_CHECKPOINT_METADATA').write_text('{}')
    for base in (run, checkpoint):
        path = base / 'assets' / ASSET
        path.mkdir(parents=True)
        (path / 'norm_stats.json').write_text('{"norm_stats":{}}')
    output = tmp / 'bundle'
    bundle.export_bundle(data, run, checkpoint, output)
    return output


class ServeBaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = make_bundle(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_pretrained_weights_with_bundle_config(self):
        policy = SimpleNamespace(metadata={'mode': 'baseline', 'weights': 'pretrained_initialization'})
        with mock.patch('policy.training.cloak.policy.load_policy', return_value=policy) as load:
            self.assertIs(bundle.load_base_bundle('/backend', self.root), policy)
        args, kwargs = load.call_args
        self.assertEqual(args, ('/backend', self.root.resolve() / 'data', self.root.resolve() / 'run'))
        self.assertEqual(kwargs, {'pretrained': True})

    def test_rejects_policy_reporting_trained_weights(self):
        policy = SimpleNamespace(metadata={'weights': '/models/x/run/checkpoints/1000'})
        with mock.patch('policy.training.cloak.policy.load_policy', return_value=policy):
            with self.assertRaisesRegex(ValueError, 'pretrained_initialization'):
                bundle.load_base_bundle('/backend', self.root)

    def test_corrupted_bundle_is_refused_before_loading(self):
        (self.root / 'run/assets' / ASSET / 'norm_stats.json').write_text('{"tampered": 1}')
        with mock.patch('policy.training.cloak.policy.load_policy') as load:
            with self.assertRaisesRegex(ValueError, 'payload mismatch'):
                bundle.load_base_bundle('/backend', self.root)
        load.assert_not_called()

    def _main(self, *argv):
        served = {}
        policy = SimpleNamespace(metadata={'weights': 'pretrained_initialization'})
        fake_server = SimpleNamespace(serve_policy=lambda p, port: served.update(policy=p, port=port))
        with mock.patch.object(sys, 'argv', ['bundle', *argv]), \
                mock.patch.dict(sys.modules, {'policy.training.cloak.server': fake_server}), \
                mock.patch('policy.training.cloak.policy.load_policy', return_value=policy) as load:
            bundle.main()
        return served, load, policy

    def test_cli_serve_base(self):
        served, load, policy = self._main('serve-base', '--backend', '/b', '--bundle', str(self.root),
                                          '--port', '8021')
        self.assertEqual(served, {'policy': policy, 'port': 8021})
        self.assertEqual(load.call_args.kwargs, {'pretrained': True})

    def test_cli_serve_base_default_port(self):
        served, _, _ = self._main('serve-base', '--backend', '/b', '--bundle', str(self.root))
        self.assertEqual(served['port'], 8020)

    def test_cli_serve_still_uses_trained_checkpoint(self):
        served, load, _ = self._main('serve', '--backend', '/b', '--bundle', str(self.root), '--port', '8012')
        self.assertEqual(served['port'], 8012)
        self.assertEqual(load.call_args.kwargs, {})
        self.assertEqual(load.call_args.args[3].name, '1000')

    def test_cli_serve_base_requires_bundle(self):
        with mock.patch.object(sys, 'argv', ['bundle', 'serve-base', '--backend', '/b']), \
                mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            bundle.main()


if __name__ == '__main__':
    unittest.main()
