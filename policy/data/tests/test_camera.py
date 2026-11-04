"""policy.data.camera: configs/hardware/camera-calibration/camera.json loading, validation and section updates."""
import json

import numpy as np
import pytest

from policy.data import camera


def test_repo_config_has_every_section():
    cfg = camera.load()
    assert set(camera.SECTIONS) <= set(cfg)
    c, m = cfg['intrinsics'], cfg['camera_imu']
    assert (c['camera_model'], c['distortion_model'], c['image_size']) == ('pinhole', 'equidistant', [1280, 720])
    assert tuple(c['image_size'][::-1]) == camera.FRAME_SHAPE
    assert len(c['distortion']) == 4 and 500 < c['matrix'][0][0] < 600
    assert m['sign_convention'] == camera.SIGN and abs(m['offset_s']) < .05
    assert not {'recording_id', 'source_sha256', 'results'} & (set(c) | set(m))   # the example names no machine's fit
    assert (camera.REPO / cfg['target']['svg']).is_file()


def test_env_names_another_file(tmp_path, monkeypatch):
    path = tmp_path / 'camera.json'
    default = camera.camera_path()                  # tests: configs/hardware/camera-calibration/camera.example.json (conftest.py)
    assert default == camera.DEFAULT_CAMERA.with_name('camera.example.json')
    path.write_text(json.dumps({'intrinsics': camera.intrinsics()}))
    monkeypatch.setenv(camera.CAMERA_ENV, str(path))
    assert camera.camera_path() == path and camera.load() == {'intrinsics': camera.intrinsics(default)}
    monkeypatch.delenv(camera.CAMERA_ENV)
    assert camera.camera_path() == default


def test_update_replaces_sections_and_keeps_the_rest(tmp_path):
    path = tmp_path / 'camera.json'
    path.write_text(camera.camera_path().read_text())
    original = camera.load(path)
    lens = {**original['intrinsics'], 'matrix': [[540.0, 0, 640.0], [0, 540.0, 360.0], [0, 0, 1]], 'recording_id': 'x'}
    data = json.loads(path.read_text())
    data['camera_imu']['offset_s'] = 0.02           # someone else's change after we read the file
    path.write_text(json.dumps(data))
    new = camera.update({'intrinsics': lens}, path)
    assert camera.load(path) == new and new['intrinsics'] == lens
    assert new['camera_imu']['offset_s'] == 0.02 and new['target'] == original['target']
    assert [p.name for p in tmp_path.iterdir()] == ['camera.json']                      # no temp file left
    assert '"image_size": [1280, 720]' in path.read_text()                                # numbers on one line


@pytest.mark.parametrize('sections, message', [
    ({'lens': {}}, 'unknown camera sections'),
    ({'intrinsics': {'camera_model': 'pinhole', 'distortion_model': 'equidistant', 'matrix': np.eye(3).tolist(),
                     'distortion': [0, 0, 0], 'image_size': [1280, 720]}}, 'distortion coefficients'),
    ({'camera_imu': {'offset_s': 0.5, 'R_cam_imu': np.eye(3).tolist(), 'gyro_bias_dps': [0, 0, 0]}}, 'offset_s'),
    ({'camera_imu': {'offset_s': 0.0, 'R_cam_imu': (2 * np.eye(3)).tolist(), 'gyro_bias_dps': [0, 0, 0]}}, 'rotation'),
    ({'target': {'target_type': 'checkerboard'}}, 'aprilgrid'),
])
def test_update_rejects_bad_sections_and_leaves_the_file(tmp_path, sections, message):
    path = tmp_path / 'camera.json'
    path.write_text(camera.camera_path().read_text())
    before = path.read_text()
    with pytest.raises(ValueError, match=message):
        camera.update(sections, path)
    assert path.read_text() == before
