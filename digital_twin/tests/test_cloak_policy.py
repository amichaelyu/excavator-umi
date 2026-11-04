"""4-joint CLOAK policy protocol; the policy server is a local test double."""
import copy
import threading
import unittest
from unittest.mock import patch

import numpy as np
from websockets.sync.server import serve

from digital_twin.policy import cloak as cp
from digital_twin.policy.cloak import JOINTS, CloakPolicy
from digital_twin.policy.vendor.openpi import msgpack_numpy as codec

METADATA = {'schema': 'actathon.cloak.policy.v1', 'mode': 'cloak_oov', 'source_embodiment': 'bucket1-handheld',
            'action_contract': {'joints': ['slew', 'boom', 'stick', 'bucket'], 'units': 'radians',
                                'type': 'absolute next joint target', 'sampling_hz': 10, 'profile': 'full4_calibrated',
                                'policy_space': 'calibrated'},
            'action_horizon': 10, 'task_group': 'dirt', 'weights': '/ckpt/29999'}
RGB = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
STATE = [0.05, -0.4, 0.9, -1.2]


class FakeCloak:
    def __init__(self, metadata=None, actions=None, delay=False, error=None):
        self.requests = []
        self.release = threading.Event()
        self.metadata = copy.deepcopy(METADATA if metadata is None else metadata)
        self.actions = np.tile(np.array([0.1, 0.2, 0.3, 0.4], np.float32), (10, 1)) if actions is None else actions

        def handle(ws):
            try:
                ws.send(codec.packb(self.metadata))
                for raw in ws:
                    self.requests.append(codec.unpackb(raw))
                    if delay:
                        self.release.wait(2)
                    if error:  # Same as policy/training/cloak/server.py: str, then close.
                        ws.send(error)
                        ws.close(code=1011, reason='Inference failed')
                        return
                    ws.send(codec.packb({'actions': self.actions, 'policy_timing': {'infer_ms': 80.0}}))
            except Exception:
                pass  # A timeout/contract rejection intentionally disconnects.

        self.server = serve(handle, '127.0.0.1', 0, max_size=cp.MAX_MESSAGE)
        self.url = 'ws://127.0.0.1:' + str(self.server.socket.getsockname()[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.thread.join(3)


def with_md(**changes):
    md = copy.deepcopy(METADATA)
    for k, v in changes.items():
        if k in md['action_contract']:
            md['action_contract'][k] = v
        else:
            md[k] = v
    return md


class Base(unittest.TestCase):
    def connect(self, **kw):
        server = FakeCloak(**kw)
        self.addCleanup(server.close)
        client = CloakPolicy(server.url, 'scoop the dirt', infer_timeout=0.2)
        self.addCleanup(client.close)
        return server, client


class ContractTests(Base):
    def test_accepts_cloak_contract(self):
        _, client = self.connect()
        md = client.metadata()
        self.assertEqual(JOINTS, ('slew', 'boom', 'stick', 'bucket'))
        self.assertEqual({k: md[k] for k in ('action_horizon', 'action_dim', 'joints', 'policy_backend',
                                             'policy_space', 'mode', 'needs_mask', 'raw_text_required')},
                         {'action_horizon': 10, 'action_dim': 4, 'joints': list(JOINTS), 'policy_backend': 'cloak',
                          'policy_space': 'calibrated', 'mode': 'cloak_oov', 'needs_mask': True,
                          'raw_text_required': False})
        self.assertEqual(md['server']['task_group'], 'dirt')
        self.assertIs(client.metadata(), md)

    def test_modes_and_mask_requirement(self):
        for changes, mode, needs in [({'mode': 'baseline'}, 'baseline', False), ({'mode': 'cloak'}, 'cloak', True),
                                     ({'weights': 'pretrained_initialization'}, 'pretrained', True),
                                     ({'mode': 'baseline', 'weights': 'pretrained_initialization'}, 'pretrained', False)]:
            with self.subTest(changes=changes):
                _, client = self.connect(metadata=with_md(**changes))
                md = client.metadata()
                self.assertEqual((md['mode'], md['needs_mask']), (mode, needs))

    def test_only_the_calibrated_label_joint_space_is_accepted(self):
        # labels from the calibrated full4 twin (= real joint space); anything else, or no space, is refused
        for space, ok in [('calibrated', True), ('handheld_reconstructed', False), ('prior', False), (None, False)]:
            with self.subTest(space=space):
                md = copy.deepcopy(METADATA)
                md['action_contract']['policy_space'] = space
                if space is None:
                    del md['action_contract']['policy_space']
                _, client = self.connect(metadata=md)
                if ok:
                    self.assertEqual(client.metadata()['policy_space'], space)
                else:
                    with self.assertRaisesRegex(ValueError, '4-joint CLOAK'):
                        client.metadata()

    def test_rejects_other_contracts(self):
        for changes in [{'schema': 'other'}, {'joints': ['boom', 'bucket']}, {'joints': ['boom', 'slew', 'stick', 'bucket']},
                        {'units': 'degrees'}, {'type': 'delta'}, {'sampling_hz': 20}, {'action_horizon': 50},
                        {'mode': 'mystery'}]:
            with self.subTest(changes=changes):
                server, client = self.connect(metadata=with_md(**changes))
                with self.assertRaisesRegex(ValueError, 'Endpoint is not the expected 4-joint CLOAK policy'):
                    client.metadata()
                self.assertIsNone(client._ws)
                self.assertEqual(server.requests, [])

    def test_rejects_2joint_openpi_metadata(self):
        _, client = self.connect(metadata={'action_horizon': 10, 'action_contract': {
            'joints': ['boom', 'bucket'], 'units': 'radians', 'type': 'absolute next joint target', 'frequency_hz': 10}})
        with self.assertRaisesRegex(ValueError, '4-joint CLOAK'):
            client.metadata()

    def test_bad_url_and_prompt(self):
        for url, prompt in [('http://127.0.0.1:1', 'x'), ('ws://u:p@127.0.0.1:1', 'x'), ('ws://127.0.0.1:1', ' ')]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                CloakPolicy(url, prompt)


class InferTests(Base):
    def test_native_frame_mask_and_state_on_the_wire(self):
        server, client = self.connect()
        mask = np.zeros(RGB.shape[:2], bool)
        mask[600:, 500:800] = True
        actions, latency, server_ms = client.infer(RGB, STATE, mask)
        np.testing.assert_array_equal(actions, server.actions)
        self.assertEqual((actions.dtype, actions.shape), (np.float32, (10, 4)))
        self.assertGreater(latency, 0)
        self.assertEqual(server_ms, 80.0)
        obs = server.requests[0]
        self.assertEqual(set(obs), {'rgb', 'embodiment_mask', 'state', 'prompt'})
        # Sent already letterboxed to 224x224 (the server's own preprocessing, which is then the identity).
        from policy.training.cloak.masking import letterbox
        rgb224, mask224 = letterbox(RGB, mask)
        np.testing.assert_array_equal(obs['rgb'], rgb224)
        np.testing.assert_array_equal(obs['embodiment_mask'], mask224)
        np.testing.assert_array_equal(letterbox(obs['rgb'], obs['embodiment_mask'])[0], rgb224)
        self.assertEqual(obs['embodiment_mask'].dtype, np.bool_)
        self.assertEqual((obs['state'].dtype, obs['state'].shape), (np.float32, (4,)))
        self.assertEqual(obs['prompt'], 'scoop the dirt')

    def test_mask_required_for_cloak_but_optional_for_baseline(self):
        server, client = self.connect()
        with self.assertRaisesRegex(ValueError, 'cloak_oov policy requires'):
            client.infer(RGB, STATE)
        self.assertEqual(server.requests, [])
        server, client = self.connect(metadata=with_md(mode='baseline'))
        client.infer(RGB, STATE)
        mask = server.requests[0]['embodiment_mask']
        self.assertEqual((mask.dtype, mask.shape, mask.any()), (np.bool_, (224, 224), False))

    def test_invalid_inputs_rejected_before_send(self):
        server, client = self.connect()
        ok = np.zeros((720, 1280), bool)
        for rgb, state, mask in [(RGB.astype(np.float32), STATE, ok), (RGB[..., :2], STATE, ok),
                                 (np.dstack([RGB, RGB[..., :1]]), STATE, ok), (RGB[..., 0], STATE, ok),
                                 (RGB, STATE[:2], ok), (RGB, [0, np.nan, 0, 0], ok), (RGB, [0, np.inf, 0, 0], ok),
                                 (RGB, STATE, ok.astype(np.uint8)), (RGB, STATE, np.zeros((224, 224), bool)),
                                 (RGB, STATE, ok.T)]:
            with self.assertRaises(ValueError):
                client.infer(rgb, state, mask)
        self.assertEqual(server.requests, [])
        self.assertIsNotNone(client._ws)  # Local validation leaves the connection usable.
        client.infer(RGB, STATE, ok)

    def test_large_frame_is_letterboxed_before_upload(self):
        server, client = self.connect()
        big = np.zeros((2160, 3840, 3), np.uint8)
        client.infer(big, STATE, np.zeros(big.shape[:2], bool))
        self.assertEqual(server.requests[0]['rgb'].shape, (224, 224, 3))
        self.assertEqual(server.requests[0]['embodiment_mask'].shape, (224, 224))

    def test_server_error_string_raises_and_closes(self):
        server, client = self.connect(error='This policy requires 4 state joints in its declared order')
        with self.assertRaisesRegex(ValueError, 'server error.*4 state joints'):
            client.infer(RGB, STATE, np.zeros((720, 1280), bool))
        self.assertIsNone(client._ws)
        self.assertIsNone(client._metadata)

    def test_bad_actions_raise_and_close(self):
        for actions in (np.zeros((10, 2), np.float32), np.zeros((50, 4), np.float32), np.full((10, 4), np.nan)):
            with self.subTest(shape=actions.shape):
                _, client = self.connect(actions=actions)
                with self.assertRaisesRegex(ValueError, 'finite'):
                    client.infer(RGB, STATE, np.zeros((720, 1280), bool))
                self.assertIsNone(client._ws)

    def test_timeout_disconnects_and_reconnects(self):
        server, client = self.connect(delay=True)
        with self.assertRaises(TimeoutError):
            client.infer(RGB, STATE, np.zeros((720, 1280), bool))
        self.assertIsNone(client._ws)
        server.release.set()
        client.infer_timeout = 5
        self.assertEqual(client.infer(RGB, STATE, np.zeros((720, 1280), bool))[0].shape, (10, 4))

    def test_refused_local_tunnel_names_endpoint(self):
        client = CloakPolicy('ws://127.0.0.1:18013/?token=secret', 'scoop the dirt')
        with patch('websockets.sync.client.connect', side_effect=ConnectionRefusedError(61, 'refused')):
            with self.assertRaises(ConnectionRefusedError) as caught:
                client.metadata()
        self.assertIn('ws://127.0.0.1:18013', str(caught.exception))
        self.assertIn('SAME computer', str(caught.exception))
        self.assertNotIn('secret', str(caught.exception))
        self.assertIsNone(client._stack)

    def test_reset_and_close_are_idempotent(self):
        _, client = self.connect()
        client.metadata()
        client.reset()
        client.close()
        client.close()
        self.assertIsNone(client._ws)
        self.assertEqual(client.metadata()['action_dim'], 4)


if __name__ == '__main__':
    unittest.main()
