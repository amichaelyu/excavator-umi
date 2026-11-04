"""Self-check of the umi-excavator-processing image: orbslam_probe loads and rejects bad input, runner selection, imports.

    docker run --rm umi-excavator-processing python -m policy.processing.docker.check_image

The image build runs it too. It checks that the probe starts and rejects incomplete IMU coverage, not SLAM accuracy.
"""
import os
from pathlib import Path
import subprocess
import tempfile

PROBE = os.environ['ORBSLAM3_PROBE']
VOCABULARY = os.environ['ORBSLAM3_VOCABULARY']

linked = subprocess.run(['ldd', PROBE], capture_output=True, text=True, check=True).stdout
assert 'not found' not in linked, f'{PROBE}: missing shared libraries\n{linked}'
usage = subprocess.run([PROBE], capture_output=True, text=True, timeout=30)
assert usage.returncode == 2 and 'Usage' in usage.stderr, (usage.returncode, usage.stderr)
assert Path(VOCABULARY).is_file(), VOCABULARY

# IMU readings that start after, or end before, the camera frames are rejected before ORB-SLAM3 starts.
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / 'mav0/imu0').mkdir(parents=True)
    times = root / 'timestamps.txt'
    times.write_text('1000000000\n2000000000\n')
    for first, last in [(1_500_000_000, 3_000_000_000), (500_000_000, 2_000_000_000)]:
        (root / 'mav0/imu0/data.csv').write_text('#timestamp,gx,gy,gz,ax,ay,az\n'
                                                 f'{first},0,0,0,0,0,9.80665\n{last},0,0,0,0,0,9.80665\n')
        result = subprocess.run([PROBE, VOCABULARY, 'unused-settings', str(root), str(times), str(root), 'imu'],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 2 and 'IMU must bracket camera timestamps' in result.stderr, result.stderr

from policy.processing.slam import orbslam  # noqa: E402
from policy.processing.slam import fusion, pipeline, report  # noqa: E402,F401

assert orbslam.native_probe() == (Path(PROBE), Path(VOCABULARY))
print('Passed: orbslam_probe loads and rejects incomplete IMU coverage, the pipeline selects it, processing imports.')
