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
    def test_registry_lists_only_cloak_full4_policies(self):
        rows = lc.entries()       # the registered CLOAK runs; which ones changes as runs are added
        self.assertTrue(rows)
        self.assertTrue(all(e['task'] in ('dirt', 'bottlecap') and e['mode'] in ('pretrained', 'baseline', 'cloak',
                                                                                 'cloak_oov', 'noaug') for e in rows))
        self.assertTrue(all((e['backend'], e['profile'], e['policy_space']) == lc.CONTRACT
                            and e['joints'] == ['slew', 'boom', 'stick', 'bucket'] for e in rows))
        self.assertEqual(len({e['gateway_url'] for e in rows}), len(rows))   # one server port per policy
        self.assertEqual([r['order'] for r in lc.runs()], list(range(len(rows))))

    def test_other_contracts_are_refused(self):
        import tempfile
        from pathlib import Path
        path = Path(tempfile.mkdtemp()) / 'checkpoints.json'
        for backend, profile, space in (('openpi', 'excavator_joint_targets', 'prior'),
                                        ('cloak', 'cloak_full4', 'handheld_reconstructed')):
            path.write_text(json.dumps({'checkpoints': [{'id': 'x', 'backend': backend, 'profile': profile,
                                                         'policy_space': space, 'gateway_url': 'ws://127.0.0.1:1'}]}))
            with self.subTest(space=space), self.assertRaisesRegex(ValueError, 'Unsupported local checkpoint contract'):
                lc.entries(path)


if __name__ == '__main__':
    unittest.main()
