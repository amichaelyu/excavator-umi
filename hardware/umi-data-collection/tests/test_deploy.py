"""The Pi deploy scripts: they parse, deploy.sh refuses to run without a target, and what it packs exists."""
import os
from pathlib import Path
import re
import subprocess
import unittest

HOME = Path(__file__).resolve().parents[1]
DEPLOY = HOME / 'deploy'


class DeployScriptTests(unittest.TestCase):
    def test_scripts_parse(self):
        for script in sorted(DEPLOY.glob('*.sh')):
            subprocess.run(['bash', '-n', str(script)], check=True)

    def test_deploy_needs_a_target(self):
        env = {k: v for k, v in os.environ.items() if k != 'UMI_PI_TARGET'}
        for args in ([], ['--no-build'], ['--diagnose'], ['-oProxyCommand=x']):
            result = subprocess.run(['bash', str(DEPLOY / 'deploy.sh'), *args], env=env,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, args)
            self.assertIn('Usage:', result.stderr)

    def test_packed_files_exist(self):
        script = (DEPLOY / 'deploy.sh').read_text()
        packed = [({'dir': DEPLOY, 'home': HOME}[var] / name) for var, name in
                  re.findall(r'"\$task_(dir|home)/([\w./-]+)"', script)
                  if not name.startswith('.devices') and name != 'pi-diagnostics.txt']
        self.assertIn(HOME / 'devices' / 'imus-bucket.json', packed)
        self.assertIn(DEPLOY / 'deploy-on-pi.sh', packed)
        for path in packed:
            self.assertTrue(path.exists(), path)


if __name__ == '__main__':
    unittest.main()
