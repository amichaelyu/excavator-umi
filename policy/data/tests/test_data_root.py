"""Dataset-root precedence and Git LFS pointer detection for policy.data.data_root (stdlib only)."""
import tempfile
import unittest
from pathlib import Path

from policy.data import data_root as dr

POINTER = b'version https://git-lfs.github.com/spec/v1\noid sha256:' + b'0' * 64 + b'\nsize 28897280\n'


class DataRootTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo-data'

    def resolve(self, cli=None, env=None):
        return dr.resolve_data_root(cli, env=env or {}, repo_default=self.repo)

    def test_precedence(self):
        self.assertEqual(self.resolve(), self.repo)                      # nothing else given
        self.assertEqual(self.resolve(env={dr.ENV_VAR: '/env'}), Path('/env'))       # env beats the repo
        self.assertEqual(self.resolve('/cli', env={dr.ENV_VAR: '/env'}), Path('/cli'))  # flag beats env
        self.assertEqual(self.resolve(env={dr.ENV_VAR: ''}), self.repo)  # empty env ignored

    def test_one_variable_shared_with_the_docker_mounts(self):
        repo = Path(dr.__file__).resolve().parents[2]
        self.assertEqual((dr.ENV_VAR, dr.REPO_DEFAULT), ('UMI_DATA_DIR', repo / 'dataset'))
        self.assertIn('${UMI_DATA_DIR:-./dataset}:/data', (repo / 'compose.yaml').read_text())
        self.assertIn('data_dir=${UMI_DATA_DIR:-', (repo / 'policy/processing/docker/run.sh').read_text())

    def test_pointer_detection(self):
        pointer, real = self.root / 'recording.tar', self.root / 'real.tar'
        pointer.write_bytes(POINTER); real.write_bytes(b'\x00' * 1024)
        self.assertTrue(dr.is_lfs_pointer(pointer))
        self.assertFalse(dr.is_lfs_pointer(real))
        self.assertFalse(dr.is_lfs_pointer(self.root / 'missing.tar'))
        self.assertFalse(dr.is_lfs_pointer(self.root))                    # a directory
        self.assertEqual(dr.require_real_file(real), real)
        with self.assertRaises(dr.LfsPointerError) as ctx:
            dr.require_real_file(pointer)
        self.assertIn('git lfs pull', str(ctx.exception))
        self.assertIn(dr.ENV_VAR, str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
