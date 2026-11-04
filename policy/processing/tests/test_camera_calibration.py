"""policy.processing.camera_calibration: AprilGrid geometry, detection on the real SVG, view selection, the fisheye
fit, the camera/IMU estimators (policy.processing.slam.calib) and the configs/camera.json it writes."""
import csv
import io
import json
import re
import tarfile

import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation, Slerp

from policy.data import camera
from policy.processing import camera_calibration as cal
from policy.processing.slam import pipeline
from policy.processing.slam.calib import CalibrationError, GyroAverager, fit_offset, rotation_intervals
from policy.processing.slam.recording import Fisheye

FOLDER = cal.REPO / 'calibration-target'
RESULTS = cal.RESULTS / '20260927T081916Z-970437b0'   # gitignored raw outputs; absent in a fresh clone


def render_svg(scale=6.0, margin=120):
    """Rasterise aprilgrid.svg (only filled axis-aligned paths); returns the image and the SVG-unit -> pixel map."""
    svg = (FOLDER / 'aprilgrid.svg').read_text()
    span = float(re.search(r'viewBox="0 (-?[\d.]+) ([\d.]+)', svg).group(2))
    image = np.full((int(span * scale) + 2 * margin,) * 2, 255, np.uint8)
    for d in re.findall(r'<path d="([^"]+)"', svg):
        pts = np.array([[float(a), float(b)] for a, b in re.findall(r'([-\d.]+) ([-\d.]+)', d)])
        px = np.column_stack([pts[:, 0] * scale + margin, (pts[:, 1] + span) * scale + margin])
        cv2.fillPoly(image, [np.round(px * 16).astype(np.int32)], 0, shift=4)
    return image, span


def test_repo_target_matches_the_measured_grid():
    t = cal.load_target()                                           # configs/camera.json target
    assert (t.cols, t.rows, t.size, t.spacing) == (6, 6, 0.0127, 0.3)
    assert t.extent == pytest.approx((0.09525, 0.09525))
    assert np.array(camera.load()['target']['grid_mm']) / 1000 == pytest.approx(t.extent)
    assert cal.load_target(grid_mm=150).size == pytest.approx(0.02)   # 150 mm / 7.5 tag edges


def test_object_points_follow_grid_layout():
    t = cal.load_target()
    points = cal.object_points(t)
    assert points.shape == (144, 3) and np.all(points[:, 2] == 0)
    ids = np.concatenate([cal.corner_ids(t, tag) for tag in range(t.tags)])
    assert sorted(ids) == list(range(144))   # every point belongs to exactly one tag corner
    s, pitch = t.size, t.pitch
    np.testing.assert_allclose(points[cal.corner_ids(t, 0)], [[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]])
    np.testing.assert_allclose(points[cal.corner_ids(t, 1)][0], [pitch, 0, 0])     # tag 1 is to the right
    np.testing.assert_allclose(points[cal.corner_ids(t, 6)][0], [0, pitch, 0])     # tag 6 is above
    np.testing.assert_allclose(points[cal.corner_ids(t, 35)][2], [*t.extent, 0])   # far top-right corner


def test_detection_on_the_svg_matches_object_points():
    t = cal.load_target()
    image, span = render_svg()
    ids, pixels = cal.detect(image, t, cal.make_detector())
    assert len(ids) == 144
    # The rendered grid is the target scaled, with image y pointing down: pixels = A @ [X, Y, 1].
    X = np.column_stack([cal.object_points(t)[ids, :2], np.ones(len(ids))])
    A, *_ = np.linalg.lstsq(X, pixels, rcond=None)
    # A wrong tag layout or corner order would be off by a tag (~170 px); rasterisation leaves ~1 px.
    assert np.max(np.linalg.norm(X @ A - pixels, axis=1)) < 2.0
    assert A[0, 0] > 0 and A[1, 1] < 0 and A[0, 0] == pytest.approx(-A[1, 1], rel=1e-3)
    assert abs(A[1, 0]) < 1e-3 * A[0, 0] and abs(A[0, 1]) < 1e-3 * A[0, 0]
    # The SVG is 7.5 tag edges wide, so its width in points over the extent gives the same scale.
    assert A[0, 0] == pytest.approx(6.0 * span / t.extent[0], rel=5e-3)


def test_select_views_drops_near_duplicates():
    ids = np.arange(20)
    base = np.random.default_rng(0).uniform(0, 500, (20, 2))
    view = lambda shift: {'ids': ids, 'points': base + shift, 'sharpness': 1.0}   # noqa: E731
    views = [view(0), view(1), view(40), view(41), view(200)]
    assert cal.select_views(views, min_motion=10, max_views=10, far=1e4) == [0, 3, 4]   # one of each pair
    assert len(cal.select_views(views, min_motion=0.1, max_views=2, far=1e4)) == 2
    unrelated = {'ids': np.arange(100, 120), 'points': base, 'sharpness': 1.0}   # shares no corners
    assert cal.view_distance(views[0], unrelated, far=1e4) == 1e4


def lens():
    c = camera.intrinsics()
    return np.array(c['matrix']), np.array(c['distortion']), tuple(c['image_size'])


def test_fisheye_fit_recovers_a_known_camera():
    K, D, size = lens()
    t = cal.load_target()
    points = cal.object_points(t)
    rng = np.random.default_rng(3)
    views = []
    while len(views) < 25:
        r = rng.normal(0, 0.45, 3)
        tv = np.array([-0.045, -0.045, 0]) + rng.normal(0, 0.04, 3) + [0, 0, 0.13]
        px = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), r, tv, K, D)[0].reshape(-1, 2)
        cam = (cv2.Rodrigues(r)[0] @ points.T).T + tv
        keep = (cam[:, 2] > 0.03) & np.all((px > 5) & (px < np.array(size) - 5), axis=1)
        if keep.sum() >= 40:
            views.append({'ids': np.flatnonzero(keep), 'points': px[keep] + rng.normal(0, 0.2, (keep.sum(), 2))})
    fit = cal.calibrate(views, t, size, min_views=15, outlier_px=1.0)
    np.testing.assert_allclose(cal.intrinsics(fit['K']), cal.intrinsics(K), atol=2.0)
    assert fit['rms'] == pytest.approx(0.2 * np.sqrt(2), rel=0.15)
    assert np.all(fit['std'] > 0) and np.all(fit['std'][:4] < 2.0)


def test_outputs_match_the_repo_calibration():
    """configs/camera.json holds the 2026-09-27 fit; its raw outputs (gitignored) are in configs/camera-calibration."""
    K, D, size = lens()
    if RESULTS.is_dir():
        assert cal.camchain(K, D, size) == yaml.safe_load((RESULTS / 'camchain.yaml').read_text())
    repo = camera.intrinsics()
    fit = {'K': K, 'D': D, 'std': np.r_[repo['std']['fx_fy_cx_cy'], repo['std']['distortion']], 'rms': 0.69,
           'corners': 6404, 'used': list(range(60)), 'coverage': 0.72}
    section = cal.intrinsics_section(fit, size, {'recording_id': repo['recording_id']})
    assert {k: section[k] for k in repo if k not in ('reprojection_rms_px', 'method', 'source_sha256', 'results')} == \
        {k: repo[k] for k in repo if k not in ('reprojection_rms_px', 'method', 'source_sha256', 'results')}
    assert repo['results'] == str(RESULTS.relative_to(cal.REPO))
    camera.validate({'intrinsics': section})


# ------------------------------------------------------------------------------------------------ camera / IMU


def test_known_positive_and_negative_offsets_with_axes_bias_and_noise():
    t = np.arange(0, 30, .005)
    gyro = np.column_stack([.55 * np.sin(2.1 * t + .015 * t * t) + .18 * np.sin(4.1 * t),
                            .35 * np.cos(1.7 * t) + .2 * np.sin(3.3 * t + .02 * t * t),
                            .4 * np.sin(1.2 * t) + .22 * np.cos(2.7 * t)])
    bias = np.array([.01, -.06, .02])
    starts = np.arange(2, 27, .05); ends = starts + .05
    r = Rotation.from_euler('xyz', [37, -28, 106], degrees=True).as_matrix()
    rng = np.random.default_rng(32)
    for offset in (.087, -.043):
        visual = GyroAverager(t, gyro)(starts, ends, offset) @ r.T + rng.normal(0, .003, (len(starts), 3))
        result = fit_offset(starts, ends, visual, t, gyro + bias, bias)
        assert result['params'][0] == pytest.approx(offset, abs=.002)
        assert Rotation.from_matrix(Rotation.from_rotvec(result['params'][1:4]).as_matrix() @ r.T).magnitude() < .01
        assert result['sensitivity'] > 1.05 and not result['ambiguous']


def test_insufficient_or_one_axis_motion_is_rejected():
    t = np.arange(0, 12, .005); starts = np.arange(2, 9, .05)
    with pytest.raises(CalibrationError, match='speed'):
        fit_offset(starts, starts + .05, np.zeros((len(starts), 3)), t, np.zeros((len(t), 3)))
    gyro = np.column_stack([np.sin(3 * t), t * 0, t * 0])
    with pytest.raises(CalibrationError, match='three rotation axes'):
        fit_offset(starts, starts + .05, GyroAverager(t, gyro)(starts, starts + .05, .03), t, gyro)
    with pytest.raises(CalibrationError, match='does not cover'):
        GyroAverager(t, gyro, strict=True)(starts, starts + .05, -2.5)


def test_pose_rotation_sign_matches_body_gyro():
    omega, dt = np.array([.2, -.3, .4]), .05
    ra = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    rb = Rotation.from_rotvec(-omega * dt).as_matrix() @ ra
    np.testing.assert_allclose(Rotation.from_matrix(ra @ rb.T).as_rotvec() / dt, omega, atol=1e-10)


def trajectory(duration=30., still=5., offset=.063, seed=14):
    """Camera orientations R_wc on the IMU clock (Slerp) from a body rate that is zero for ``still`` s at both ends,
    and the camera stamps whose frames show them (stamp + offset)."""
    t = np.arange(0, duration + 1, .002)
    ramp = np.clip(np.minimum(t - still, duration - still - t), 0, 1)
    omega = ramp[:, None] * np.column_stack([.4 * np.sin(2.1 * t + .015 * t * t) + .1 * np.sin(4.1 * t),
                                             .3 * np.cos(1.7 * t) + .1 * np.sin(3.3 * t),
                                             .35 * np.sin(1.2 * t) + .15 * np.cos(2.7 * t)])
    steps = Rotation.from_rotvec((omega[:-1] + omega[1:]) * .001)
    R = [Rotation.identity()]
    for s in steps:
        R.append(R[-1] * s)
    return t, omega, Slerp(t, Rotation.concatenate(R)), np.arange(0, duration, .05), offset


def test_noisy_camera_poses_keep_correct_time_offset():
    """Independent SO(3) integration, followed by camera pose noise: catches phase shifts that a synthetic
    angular-velocity-only test misses."""
    t, omega, slerp, stamps, offset = trajectory(24, still=1)
    extrinsic = Rotation.from_euler('xyz', [37, -28, 106], degrees=True)   # R_cam_imu
    rng = np.random.default_rng(14)
    stamps = stamps[(stamps > 1) & (stamps < 22)]
    measured = Rotation.from_rotvec(rng.normal(0, .003, (len(stamps), 3))) * slerp(stamps + offset).inv()
    poses = list(zip(stamps, measured.as_matrix(), np.zeros(len(stamps))))
    gyro = omega @ extrinsic.as_matrix()                                     # IMU axes: R_imu_cam omega_cam
    bias = np.array([.01, -.06, .02])
    for span in (.1, .15):
        starts, ends, visual = rotation_intervals(poses, 2, 21, span)
        result = fit_offset(starts, ends, visual, t, gyro + bias, bias)
        assert result['params'][0] == pytest.approx(offset, abs=.004)
        assert result['rms'] < np.deg2rad(5) and result['sensitivity'] > 1.05


def test_rotation_intervals_do_not_bridge_missing_tracking():
    times = np.r_[np.arange(0, 1, .05), np.arange(1.3, 2, .05)]
    poses = [(t, Rotation.from_rotvec([.1 * t, 0, 0]).as_matrix(), 0) for t in times]
    starts, ends, _ = rotation_intervals(poses, 0, 2)
    assert np.all((ends - starts > .08) & (ends - starts < .13))
    assert not np.any((starts < 1) & (ends > 1.2))


# ------------------------------------------------------------------------------------------------ the tool


def take(tmp_path, offset=.0123, bias_dps=(1.1, -3.4, .1)):
    """A synthetic recorder take: recording.tar (metadata, frames.csv, imu.csv) of the camera turning in front of the
    grid, and the fisheye projections of the grid per frame (what detect() would find). Returns (tar, detections,
    R_cam_imu)."""
    K, D, size = lens()
    t, omega, slerp, stamps, offset = trajectory(offset=offset)
    R_cam_imu = np.array(camera.camera_imu()['R_cam_imu'])
    points = cal.object_points(cal.load_target())
    rng = np.random.default_rng(5)
    detections = []
    for i, (s, R_wc) in enumerate(zip(stamps, slerp(stamps + offset).as_matrix())):
        c = [.0475 + .02 * np.sin(.3 * s), .0475 + .02 * np.cos(.4 * s), -.13 + .03 * np.sin(.25 * s)]   # camera centre
        r = cv2.Rodrigues(R_wc.T)[0]
        px = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), r, -R_wc.T @ c, K, D)[0].reshape(-1, 2)
        keep = np.all((px > 5) & (px < np.array(size) - 5), axis=1)
        detections.append((np.flatnonzero(keep), px[keep] + rng.normal(0, .2, (keep.sum(), 2))) if keep.sum() >= 24 else None)
    origin = 100.
    it = np.arange(0, t[-1], 1 / 208)
    gyro = np.rad2deg(np.column_stack([np.interp(it, t, w) for w in omega.T]) @ R_cam_imu) + bias_dps
    files = {'metadata.json': json.dumps({'id': '20990101T000000Z-synthetic', 'started_monotonic_s': origin}),
             'frames.csv': 'sequence,camera_timestamp_ns,offset,size\n' + ''.join(
                 f'{i},{round((origin + s) * 1e9)},0,0\n' for i, s in enumerate(stamps)),
             'imu.csv': 'read_monotonic_ns,gx_dps,gy_dps,gz_dps\n' + ''.join(
                 f'{round((origin + a) * 1e9)},{g[0]},{g[1]},{g[2]}\n' for a, g in zip(it, gyro)),
             'video.mjpg': ''}
    path = tmp_path / 'take.tar'
    with tarfile.open(path, 'w') as archive:
        for name, text in files.items():
            info = tarfile.TarInfo(name); data = text.encode(); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path, stamps, detections, R_cam_imu


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """The tool on a synthetic take: frames() and detect() replay the projections, the rest runs for real."""
    path, stamps, detections, R_cam_imu = take(tmp_path)
    current = [0]

    def frames(video, sample_hz, decode_all=False):
        for i, s in enumerate(stamps):
            current[0] = i
            yield i, s, i % round(20 / sample_hz) == 0, np.zeros((720, 1280), np.uint8)

    monkeypatch.setattr(cal, 'frames', frames)
    monkeypatch.setattr(cal, 'detect', lambda gray, t, detector, min_tags=6: detections[current[0]])
    monkeypatch.setattr(cal, 'sharpness', lambda gray, points: 1.0)
    config = tmp_path / 'camera.json'
    config.write_text(camera.camera_path().read_text())
    return path, config, R_cam_imu


def test_tool_writes_intrinsics_and_camera_imu(synthetic, tmp_path):
    path, config, R_cam_imu = synthetic
    before = camera.load(config)
    assert cal.main([str(path), '--output', str(tmp_path / 'out'), '--config', str(config)]) == 0
    after = camera.load(config)
    K, D, _ = lens()
    np.testing.assert_allclose(np.array(after['intrinsics']['matrix']), K, atol=2.0)
    assert after['intrinsics']['recording_id'] == '20990101T000000Z-synthetic'
    assert after['intrinsics']['source_sha256'] == cal.sha256(path)
    m = after['camera_imu']
    assert m['check']['status'] == 'passed_internal_timing_check'
    assert m['offset_s'] == pytest.approx(.0123, abs=.003)
    assert Rotation.from_matrix(np.array(m['R_cam_imu']) @ R_cam_imu.T).magnitude() < np.deg2rad(1)
    np.testing.assert_allclose(m['gyro_bias_dps'], (1.1, -3.4, .1), atol=.3)
    assert after['target'] == before['target'] and after['description'] == before['description']
    out = tmp_path / 'out'
    assert {p.name for p in out.iterdir()} == {'camchain.yaml', 'results.txt', 'views.csv', 'time-check.json', 'camera.json'}
    assert json.loads((out / 'camera.json').read_text()) == {k: after[k] for k in ('intrinsics', 'camera_imu')}
    with (out / 'views.csv').open() as f:
        assert {'used', 'duplicate'} <= {row['status'] for row in csv.DictReader(f)}


def test_dry_run_and_no_imu_leave_the_config(synthetic, tmp_path):
    path, config, _ = synthetic
    text = config.read_text()
    assert cal.main([str(path), '--output', str(tmp_path / 'dry'), '--config', str(config), '--dry-run']) == 0
    assert config.read_text() == text and 'camera_imu' in json.loads((tmp_path / 'dry' / 'camera.json').read_text())
    assert cal.main([str(path), '--output', str(tmp_path / 'lens'), '--config', str(config), '--no-imu',
                     '--grid-mm', '95.25']) == 0
    after = camera.load(config)
    assert after['camera_imu'] == json.loads(text)['camera_imu']             # lens (and target) only
    assert after['target']['grid_mm'] == [95.25, 95.25] and after['target']['tagSize'] == pytest.approx(.0127)
    assert not (tmp_path / 'lens' / 'time-check.json').exists()


# ------------------------------------------------------------------------------------------------ readers


def test_slam_reads_the_camera_config(tmp_path, monkeypatch):
    cam = Fisheye.load()
    np.testing.assert_array_equal(cam.K, lens()[0])
    m = camera.camera_imu()
    monkeypatch.setattr(pipeline, 'CALIBRATION', None)
    monkeypatch.setattr(pipeline, 'ARTIFACTS', None)
    cal_, R_bc, off = pipeline.load_calibration()                         # default: configs/camera.json
    assert off == m['offset_s'] and cal_['per_episode'] == {}
    np.testing.assert_allclose(R_bc, np.array(m['R_cam_imu']).T, atol=1e-6)
    np.testing.assert_allclose(pipeline.gyro_bias_prior(cal_, 'any'), np.deg2rad(m['gyro_bias_dps']))
    # A per-dataset natural-feature fit keeps its per-episode bias priors (0 for episodes it did not fit).
    per = {'pooled': {'offset_s': -.01, 'R_cam_imu': np.eye(3).tolist(), 'gyro_bias_dps': [[1, 2, 3], [4, 5, 6]]},
           'per_episode': {'a': {}, 'b': {}}}
    (tmp_path / 'calibration.json').write_text(json.dumps(per))
    assert pipeline.set_artifacts(tmp_path, tmp_path / 'calibration.json') == tmp_path.resolve()
    cal_, R_bc, off = pipeline.load_calibration()
    assert off == -.01
    np.testing.assert_allclose(pipeline.gyro_bias_prior(cal_, 'b'), np.deg2rad([4, 5, 6]))
    np.testing.assert_allclose(pipeline.gyro_bias_prior(cal_, 'c'), 0)


def test_natural_section_from_a_pooled_fit(monkeypatch, tmp_path):
    fake = {'pooled': {'offset_s': -.008, 'offset_sd_s': .001, 'R_cam_imu': np.eye(3).tolist(), 'pairs': 900,
                       'gyro_bias_dps': [[1, -3, 0], [1.2, -3.4, .2], [.8, -3.2, 0]], 'residual_median_dps': 1.,
                       'speed_correlation': .99},
            'per_episode': {'a': {}, 'b': {}, 'c': {}}, 'leave_one_out': [{'offset_ms': -8.2}, {'offset_ms': -7.9}]}
    monkeypatch.setattr(pipeline, 'calibrate', lambda work, episodes: fake)
    full, section = cal.natural(['a', 'b', 'c'], tmp_path)
    assert full is fake and section['gyro_bias_dps'] == [1, -3.2, 0] and section['offset_s'] == -.008
    assert section['check']['leave_one_out_offset_ms'] == [-8.2, -7.9]
    camera.validate({'camera_imu': section})


def test_orbslam_settings_read_the_camera_config():
    from policy.processing.slam import orbslam
    from policy.processing.slam.fusion import ImuNoise
    noise = ImuNoise()
    settings = orbslam.settings_yaml(Fisheye.load(), np.eye(3), np.zeros(3), noise)
    assert 'Camera.type: "KannalaBrandt8"' in settings and f'Camera1.fx: {lens()[0][0, 0]:.10f}' in settings
    w, h = camera.intrinsics()['image_size']
    assert f'Camera.width: {w}' in settings and f'Camera.height: {h}' in settings
    assert f'IMU.GyroWalk: {noise.gyro_walk:.3e}' in settings and f'IMU.AccWalk: {noise.accel_walk:.3e}' in settings
    assert 'IMU.' not in orbslam.settings_yaml(Fisheye.load(), np.eye(3), np.zeros(3), noise, inertial=False)
