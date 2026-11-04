"""Checkpoint registry and runner contract of the CLOAK π0.5 servers (the inference dashboard side:
hub/tests/test_inference.py); all robot transports are test doubles."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from digital_twin.policy import checkpoints as lc, runner as pr


class RunnerContractTests(unittest.TestCase):
    def test_bridge_connection_failure_is_identified_before_policy_connect(self):
        logs = []
        bridge = pr.BridgeExecutor({}, 'localhost:8888', dry_run=False, log=logs.append)
        with patch('websockets.sync.client.connect', side_effect=ConnectionRefusedError(61, 'refused')):
            with self.assertRaises(ConnectionRefusedError):
                bridge.connect()
        self.assertEqual(logs, ['connecting to hardware bridge: ws://localhost:8888/api/gpio/ws'])

    def test_runner_refuses_other_control_spaces(self):
        base = ['--gateway-url', 'ws://127.0.0.1:8040']
        for flags in (['--hz', '20'], ['--chunk', '11']):
            with self.subTest(flags=flags), self.assertRaises(pr.RunnerError):
                pr.PolicyRunner(pr.build_parser().parse_args(base + flags))
        for flags in (['--dataset-version', '2'], ['--profile', 'excavator_joint_targets'], ['--action-space', 'pulse'],
                      ['--policy-space', 'prior'], ['--no-joint-transform']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                pr.build_parser().parse_args(base + flags)


class RegistryTests(unittest.TestCase):
    def test_registry_lists_only_pi05_tabletop_excavator_policies(self):
        example = lc.DEFAULT_REGISTRY.with_name('local_checkpoints.example.json')   # the local file is per machine
        rows = lc.entries(example)
        self.assertTrue(rows)
        self.assertTrue(all(e['task'] in ('dirt', 'bottlecap') and e['mode'] in ('pretrained', 'baseline', 'cloak',
                                                                                 'cloak_oov', 'noaug') for e in rows))
        self.assertTrue(all((e['backend'], e['profile'], e['policy_space']) == lc.CONTRACT for e in rows))
        self.assertEqual(len({e['gateway_url'] for e in rows}), len(rows))   # one server port per policy
        self.assertEqual([r['order'] for r in lc.runs(example)], list(range(len(rows))))

    def test_other_contracts_are_skipped_with_a_warning(self):
        """One entry outside the contract (e.g. serve's handheld_reconstructed policy_space) is left out and
        logged; the other entries still list, find and run."""
        import tempfile
        from pathlib import Path
        path = Path(tempfile.mkdtemp()) / 'checkpoints.json'
        good = {'id': 'good', 'gateway_url': 'ws://127.0.0.1:2', 'step': 500}
        for backend, profile, space in (('openpi', 'excavator_joint_targets', 'prior'),
                                        ('cloak', 'pi05_tabletop_excavator', 'handheld_reconstructed')):
            bad = {'id': 'bad', 'backend': backend, 'profile': profile, 'policy_space': space,
                   'gateway_url': 'ws://127.0.0.1:1'}
            path.write_text(json.dumps({'checkpoints': [bad, good]}))
            with self.subTest(space=space), self.assertLogs(lc.__name__, 'WARNING') as logs:
                self.assertEqual([e['id'] for e in lc.entries(path)], ['good'])
                self.assertEqual([r['name'] for r in lc.runs(path)], ['good'])
                self.assertIsNone(lc.find('bad', path))
                self.assertEqual(lc.find('good', path)['policy_space'], lc.POLICY_SPACE)
            self.assertIn("'bad'", logs.output[0])
            self.assertIn(space, logs.output[0])

    def test_add_stays_strict_next_to_a_skipped_entry(self):
        """add() still checks the server's contract, and writes only contract entries; a skipped entry is kept
        in the file (not silently dropped) for the user to fix."""
        import tempfile
        from pathlib import Path
        path = Path(tempfile.mkdtemp()) / 'checkpoints.json'
        bad = {'id': 'bad', 'policy_space': 'handheld_reconstructed', 'gateway_url': 'ws://127.0.0.1:1'}
        path.write_text(json.dumps({'checkpoints': [bad]}))

        class Server:
            metadata_error = None

            def __init__(self, url, prompt, connect_timeout):
                pass

            def metadata(self):
                if Server.metadata_error:
                    raise ValueError(Server.metadata_error)
                return {'mode': 'cloak'}

            def close(self):
                pass

        with patch.object(lc, 'CloakPolicy', Server), self.assertLogs(lc.__name__, 'WARNING'):
            Server.metadata_error = 'policy_space handheld_reconstructed is not calibrated'
            with self.assertRaisesRegex(ValueError, 'not calibrated'):
                lc.add('x', 'ws://127.0.0.1:3', 'dig', path=path)
            Server.metadata_error = None
            entry = lc.add('ok', 'ws://127.0.0.1:3', 'dig', path=path)
        self.assertEqual((entry['backend'], entry['profile'], entry['policy_space']), lc.CONTRACT)
        self.assertEqual([e['id'] for e in json.loads(path.read_text())['checkpoints']], ['bad', entry['id']])


if __name__ == '__main__':
    unittest.main()
