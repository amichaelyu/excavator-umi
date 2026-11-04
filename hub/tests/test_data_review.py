import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from hub.apps import _common as common
from hub.apps.data_review import core as wv
from hub.app import Hub, create_app
from hub.apps import data_review as tva

REPO = Path(__file__).resolve().parents[2]
REPLAY = REPO / 'artifacts' / 'soil-replay'      # optional local replay / trajectory fixtures (tests skip without)
SLAM = REPO / 'artifacts' / 'soil-vio'


def viewer(replay, slam, data) -> tva.DataReview:
    """A DataReview over one (replay, trajectory, data root) source, labels in a file under the data root."""
    index = wv.EpisodeIndex([wv.Source('test', Path(replay), Path(slam), (Path(data),))])
    return tva.DataReview(index, wv.LabelStore(Path(data) / 'labels.json'))


def fake_episode(replay_root, slam_root, rid, profile='full4_calibrated'):
    """The files load_episodes needs: metrics.json + a four-joint fk_dynamic.csv, and a trajectory.csv."""
    (replay_root / rid).mkdir(parents=True)
    (slam_root / rid).mkdir(parents=True)
    (replay_root / rid / 'metrics.json').write_text(json.dumps({'profile': profile, 'duration_s': 1.0}))
    (replay_root / rid / 'fk_dynamic.csv').write_text('t_s,qpos_slew,qpos_boom,qpos_stick,qpos_bucket\n0,0,0,0,0\n')
    (slam_root / rid / 'trajectory.csv').write_text('t_s,status\n0,tracked\n0.5,tracked\n')


def hub_client(service, app_id='data-review', module='data_review') -> TestClient:
    """The real hub app (routing, error shape, host guard) with ``service`` as the running ``app_id``."""
    with tempfile.TemporaryDirectory() as tmp:
        reg = Path(tmp) / 'apps.json'
        reg.write_text(json.dumps({'apps': [{'id': app_id, 'module': module, 'name': app_id}]}))
        hub = Hub(registry=reg, local=None, autostart=False)
    hub.apps[app_id].service = service
    return TestClient(create_app(hub))


def _get(client, path, headers=None):
    r = client.get(path, headers=headers or {})
    return r.status_code, r.headers, r.content


class TimeMappingTests(unittest.TestCase):
    def test_original_and_replay_round_trip(self):
        t = np.linspace(0, 12, 101)
        np.testing.assert_allclose(wv.original_episode_time(wv.original_video_time(t, 0.572121), 0.572121), t)
        for s in (1.0, 1.7, 3.2):
            r = wv.replay_video_time(t, s)
            np.testing.assert_allclose(r, t * s)
            np.testing.assert_allclose(wv.replay_episode_time(r, s), t)

    def test_local_slowdown_and_nonzero_recording_origin(self):
        # Half a recording second needs two robot seconds, then playback speeds up.
        clock = {'source_t_s': [7., 7.5, 8.], 'replay_t_s': [0., 2., 3.]}
        source = np.array([7., 7.25, 7.5, 7.75, 8.])
        robot = np.array([0., 1., 2., 2.5, 3.])
        np.testing.assert_allclose(wv.replay_video_time(source, 1.3, clock), robot)
        np.testing.assert_allclose(wv.replay_episode_time(robot, 1.3, clock), source)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'actions.csv').write_text('t_s,source_t_s\n0,7\n2,7.5\n3,8\n')
            ep = wv.Episode('retimed', root, root, {'time_scale': 1.3, 'duration_s': 3.},
                            {}, source, ['tracked'] * len(source))
            self.assertEqual(ep.replay_clock, clock)
            np.testing.assert_allclose(ep.replay_time(source), robot)
            self.assertEqual(ep.duration, 8.)

    def test_missing_or_invalid_source_clock_is_not_silently_desynchronized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for csv_text in ('t_s,source_t_s\n0,1\n1,1\n', 't_s\n0\n1\n'):
                (root / 'actions.csv').write_text(csv_text)
                ep = wv.Episode('bad-clock', root, root, {'source_time': {'nonuniform': True}},
                                {}, np.array([0., 1.]), ['tracked', 'tracked'])
                with self.assertRaises(ValueError):
                    ep.replay_time(.5)


class TrajectoryTests(unittest.TestCase):
    def test_orbslam3_camera_path_for_the_page(self):
        with tempfile.TemporaryDirectory() as d:
            replay, slam = Path(d) / 'replay', Path(d) / 'slam'
            fake_episode(replay, slam, 'r1')
            ep = wv.load_episodes(replay, slam, Path(d))['r1']
            self.assertIsNone(ep.trajectory())                   # t_s / status only: no poses
            # camera at x = 0.1 then 0.2; identity, then 90 deg about world z (optical axis +z stays +z)
            (slam / 'r1' / 'trajectory.csv').write_text(
                't_s,px,py,pz,qw,qx,qy,qz,status\n0,0.1,0,0,1,0,0,0,tracked\n'
                f'0.5,0.2,0,0,{np.cos(np.pi / 4)},0,0,{np.sin(np.pi / 4)},imu_only\n')
            tr = wv.load_episodes(replay, slam, Path(d))['r1'].trajectory()
            self.assertEqual(tr['p'], [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
            np.testing.assert_allclose(tr['fwd'], [[0, 0, 1], [0, 0, 1]], atol=1e-3)
            # 90 deg about camera x: the optical axis turns to world -y
            (slam / 'r1' / 'trajectory.csv').write_text(
                't_s,px,py,pz,qw,qx,qy,qz,status\n'
                f'0,0,0,0,{np.cos(np.pi / 4)},{np.sin(np.pi / 4)},0,0,tracked\n')
            np.testing.assert_allclose(wv.load_episodes(replay, slam, Path(d))['r1'].trajectory()['fwd'],
                                       [[0, -1, 0]], atol=1e-3)

    def test_rows_without_a_pose_are_null_and_the_episode_still_loads(self):
        with tempfile.TemporaryDirectory() as d:
            replay, slam = Path(d) / 'replay', Path(d) / 'slam'
            fake_episode(replay, slam, 'r1')
            (replay / 'r1' / 'fk_dynamic.csv').write_text(     # a recorded four-joint replay, as the labels stage writes
                't_s,qpos_slew,qpos_boom,qpos_stick,qpos_bucket,tip_x,tip_y,tip_z,tip_err_m\n0,0,0,0,0,0.3,0,0.1,0\n'
                '0.5,0,0,0,0,0.3,0,0.1,0\n')
            (replay / 'r1' / 'fk_kinematic.csv').write_text(
                't_s,q_slew,q_boom,q_stick,q_bucket,tip_x,tip_y,tip_z\n0,0,0,0,0,0.3,0,0.1\n0.5,0,0,0,0,0.3,0,0.1\n')
            (replay / 'r1' / 'actions.csv').write_text(
                't_s,tip_target_x,tip_target_y,tip_target_z,pitch_target,reachable\n0,0.3,0,0.1,0,1\n0.5,0.3,0,0.1,0,1\n')
            (slam / 'r1' / 'trajectory.csv').write_text(      # the writer keeps pose-less frames as NaN
                't_s,px,py,pz,qw,qx,qy,qz,status\n0,0.1,0,0,1,0,0,0,tracked\n0.5,nan,nan,nan,nan,nan,nan,nan,imu_only\n')
            eps = wv.load_episodes(replay, slam, Path(d))
            self.assertEqual(eps['r1'].trajectory()['p'], [[0.1, 0.0, 0.0], None])
            client = hub_client(tva.DataReview(eps, wv.LabelStore(Path(d) / 'labels.json')))
            r = client.get('/api/data-review/episode?id=r1')
            self.assertEqual(r.status_code, 200, r.text)
            self.assertNotIn('NaN', r.text)                                  # strict JSON a browser can parse
            self.assertEqual(json.loads(r.text)['trajectory']['fwd'][1], None)


class PureHelperTests(unittest.TestCase):
    def test_parse_range(self):
        self.assertIsNone(common.parse_range(None, 100))
        self.assertIsNone(common.parse_range('items=0-5', 100))
        self.assertEqual(common.parse_range('bytes=0-9', 100), (0, 9))
        self.assertEqual(common.parse_range('bytes=90-', 100), (90, 99))
        self.assertEqual(common.parse_range('bytes=-10', 100), (90, 99))
        self.assertEqual(common.parse_range('bytes=50-500', 100), (50, 99))
        self.assertEqual(common.parse_range('bytes=0-1, 5-6', 100), (0, 1))
        for bad in ('bytes=100-', 'bytes=5-2', 'bytes=-0'):
            with self.assertRaises(ValueError):
                common.parse_range(bad, 100)

    def test_extract_jpeg_from_multipart(self):
        part = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8abc\xff\xd9\r\n--frame'
        self.assertEqual(common.extract_jpeg(part), b'\xff\xd8abc\xff\xd9')
        self.assertIsNone(common.extract_jpeg(b'--frame\r\n\xff\xd8partial'))

    def test_unreachable_live_source_is_502(self):
        status, body, ctype = common.fetch_live_frame('http://127.0.0.1:9/preview.jpg', timeout=0.5)
        self.assertEqual(status, 502)
        self.assertIn('error', json.loads(body))


    def test_original_video_offset_from_frames_csv(self):
        import io
        import tarfile
        ep = '20260927T070203Z-bf0ec1b6'
        if not (SLAM / ep / 'summary.json').is_file():
            self.skipTest('ORB-SLAM3 artifacts missing')
        t0 = json.loads((SLAM / ep / 'summary.json').read_text())['time_origin']['raw_camera_timestamp_ns_at_t0']
        with tempfile.TemporaryDirectory() as root:
            d = Path(root) / ep
            d.mkdir(parents=True)
            csv_bytes = f'camera_timestamp_ns,sequence\n{t0 - 572121000},1\n{t0},2\n'.encode()
            with tarfile.open(d / 'recording.tar', 'w') as tf:
                info = tarfile.TarInfo('frames.csv'); info.size = len(csv_bytes)
                tf.addfile(info, io.BytesIO(csv_bytes))
            (d / 'video.mp4').write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:0\n')
            self.assertIsNone(wv.load_episodes(REPLAY, SLAM, Path(root))[ep].original_video)   # LFS pointer
            (d / 'video.mp4').write_bytes(b'\x00\x00\x00\x18ftypmp42')
            e = wv.load_episodes(REPLAY, SLAM, Path(root))[ep]
            self.assertEqual(e.original_video, d / 'video.mp4')
            self.assertAlmostEqual(e.video_offset_s, 0.572121, places=9)
            self.assertEqual(e.video_frames, 2)


@unittest.skipUnless((REPLAY / '20260927T070203Z-bf0ec1b6' / 'fk_dynamic.csv').is_file(), 'replay artifacts missing')
class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()        # empty data root: no original videos
        cls.srv = viewer(REPLAY, SLAM, cls.tmp.name)
        cls.client = hub_client(cls.srv)
        cls.base = '/api/data-review'
        cls.ep = '20260927T070203Z-bf0ec1b6'

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        cls.tmp.cleanup()

    def get(self, path, headers=None):
        return _get(self.client, path, headers)

    def test_index_lists_episodes_with_time_scale(self):
        status, _, body = self.get(self.base + '/index')
        self.assertEqual(status, 200)
        idx = json.loads(body)
        ids = {e['id'] for e in idx['episodes']}
        expected = {d.name for d in REPLAY.iterdir() if (d / 'metrics.json').is_file() and (SLAM / d.name / 'trajectory.csv').is_file()}
        self.assertEqual(ids, expected)
        self.assertGreaterEqual(len(ids), 8)
        for e in idx['episodes']:
            m = json.loads((REPLAY / e['id'] / 'metrics.json').read_text())
            self.assertEqual(e['time_scale'], m['time_scale'])
            self.assertIn('replay.mp4', e['media'])
            self.assertNotIn('video.mp4', e['media'])
            self.assertGreater(e['duration_s'], 1)
        self.assertNotIn('live', idx)

    def test_path_error_summary_pools_episodes(self):
        status, _, body = self.get(self.base + '/path_error_summary')
        self.assertEqual(status, 200)
        s = json.loads(body)
        per = [e.path_error() for e in self.srv.episodes.values() if e.has_mode()]
        n = sum(p['ik']['samples'] for p in per)
        mse = sum(p['ik']['tip_mse_mm2'] * p['ik']['samples'] for p in per) / n
        self.assertEqual(s['ik']['episodes'], len(per))
        self.assertEqual(s['ik']['samples'], n)
        self.assertAlmostEqual(s['ik']['tip']['mse_mm2'], mse, places=6)
        self.assertAlmostEqual(s['ik']['tip']['rmse_mm'] ** 2, mse, places=6)
        self.assertLessEqual(s['ik']['tip']['median_episode_rmse_mm'], s['ik']['tip']['worst_episode_rmse_mm'])
        self.assertIn('rmse_deg', s['ik']['angle'])
        rmse = sorted(p['ik']['tip_rmse_mm'] for p in per)
        b = s['ik']['tip']['bands']
        self.assertAlmostEqual(b['good_max'], float(np.percentile(rmse, 100 / 3)), places=6)
        self.assertAlmostEqual(b['medium_max'], float(np.percentile(rmse, 200 / 3)), places=6)
        self.assertLessEqual(b['good_max'], b['medium_max'])

    def test_episode_detail(self):
        status, _, body = self.get(self.base + f'/episode?id={self.ep}')
        d = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(len(d['frame_t']), len(d['status']))
        self.assertEqual(set(d['fk']), {'twin'})                # one twin, one FK / IK
        twin = d['fk']['twin']
        self.assertEqual(twin['joints'], ['slew', 'boom', 'stick', 'bucket'])
        self.assertEqual(len(twin['q'][0]), 4)
        # a two-joint replay is re-solved on the twin: no video of it, every target reached
        self.assertIsNone(twin['video'])
        self.assertLess(max(twin['tip_err']), 1e-3)
        self.assertEqual(self.get(self.base + '/episode?id=nope')[0], 404)

    def test_no_height_offset(self):
        d = json.loads(self.get(self.base + f'/episode?id={self.ep}')[2])
        self.assertFalse({'dz', 'height'} & set(d))

    def test_range_requests(self):
        f = REPLAY / self.ep / 'replay.mp4'
        size, data = f.stat().st_size, f.read_bytes()
        url = self.base + f'/media/{self.ep}/replay.mp4'
        status, h, body = self.get(url, {'Range': 'bytes=100-1123'})
        self.assertEqual(status, 206)
        self.assertEqual(h['Content-Range'], f'bytes 100-1123/{size}')
        self.assertEqual(body, data[100:1124])
        status, h, body = self.get(url, {'Range': 'bytes=-50'})
        self.assertEqual((status, h['Content-Range'], body), (206, f'bytes {size - 50}-{size - 1}/{size}', data[-50:]))
        status, h, body = self.get(url)
        self.assertEqual((status, h['Accept-Ranges'], len(body)), (200, 'bytes', size))
        self.assertEqual(h['Content-Type'], 'video/mp4')
        status, h, _ = self.get(url, {'Range': f'bytes={size}-'})
        self.assertEqual((status, h['Content-Range']), (416, f'bytes */{size}'))
        self.assertEqual(self.get(self.base + f'/media/{self.ep}/metrics.json')[0], 404)
        self.assertEqual(self.get(self.base + f'/media/{self.ep}/..%2F..%2Fmetrics.json')[0], 404)
        self.assertEqual(self.get(self.base + f'/media/{self.ep}/video.mp4')[0], 404)

    def test_sim_frame(self):
        import cv2
        for cam, mode in (('overview', 'twin'), ('side', 'twin'), ('bucket_camera', 'twin'),
                          ('overview', None)):
            q = f'episode={self.ep}&t=3.3&camera={cam}' + (f'&mode={mode}' if mode else '')
            status, h, body = self.get(self.base + f'/sim_frame?{q}')
            self.assertEqual(status, 200, body[:200])
            self.assertEqual(h['Content-Type'], 'image/jpeg')
            img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(img.shape, (wv.FRAME_H, wv.FRAME_W, 3))
        bad = {'episode=nope&t=1': 404, f'episode={self.ep}&t=abc': 400, f'episode={self.ep}&t=-1': 400,
               f'episode={self.ep}&t=999': 400, f'episode={self.ep}&t=nan': 400,
               f'episode={self.ep}&t=1&camera=top': 400, f'episode={self.ep}&t=1&mode=pulses': 400,
               f'episode={self.ep}&t=1&mode=dynamic': 400}
        for qs, code in bad.items():
            self.assertEqual(self.get(self.base + '/sim_frame?' + qs)[0], code, qs)

    def test_view_and_no_live_proxy(self):
        status, h, body = self.get('/web/views/data-review/view.js')
        self.assertEqual(status, 200)
        self.assertIn(b'export async function mount', body)
        status, _, body = self.get(self.base + '/nope')
        self.assertEqual((status, json.loads(body)), (404, {'error': 'not found'}))
        self.assertEqual(self.get(self.base + '/live_frame')[0], 404)


@unittest.skipUnless((REPLAY / '20260927T070203Z-bf0ec1b6' / 'fk_dynamic.csv').is_file()
                     and (REPLAY / '20260927T070234Z-baea98b3' / 'fk_dynamic.csv').is_file(), 'replay artifacts missing')
class LiveIndexTests(unittest.TestCase):
    """Temp replay/ORB-SLAM3 roots + per-recording metadata.json labels; episodes added after start must appear."""
    A, B = '20260927T070203Z-bf0ec1b6', '20260927T070234Z-baea98b3'

    @staticmethod
    def _copy(ep, replay_root, slam_root):
        import shutil
        (replay_root / ep).mkdir(parents=True)
        (slam_root / ep).mkdir(parents=True)
        for f in ('fk_kinematic.csv', 'actions.csv', 'joints.png', 'fk_dynamic.csv', 'metrics.json'):
            if (REPLAY / ep / f).is_file():
                shutil.copy(REPLAY / ep / f, replay_root / ep / f)
        for f in ('summary.json', 'trajectory.csv'):
            if (SLAM / ep / f).is_file():
                shutil.copy(SLAM / ep / f, slam_root / ep / f)

    def test_index_rescans_and_carries_label_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            replay, slam, data = tmp / 'replay', tmp / 'slam', tmp / 'data'
            replay.mkdir(); slam.mkdir(); data.mkdir()
            for ep, label in ((self.A, ''), (self.B, 'ball')):
                (data / ep).mkdir()
                (data / ep / 'metadata.json').write_text(json.dumps({'id': ep, 'label': label}))
            self._copy(self.A, replay, slam)
            srv = viewer(replay, slam, data)
            client, base = hub_client(srv), '/api/data-review'

            def get(path):
                return _get(client, path)
            try:
                idx = json.loads(get(base + '/index')[2])
                self.assertEqual([e['id'] for e in idx['episodes']], [self.A])
                a = idx['episodes'][0]
                self.assertEqual((a['label'], a['task'], a['group']), ('', '', 'dig'))
                m = json.loads((REPLAY / self.A / 'metrics.json').read_text())
                self.assertEqual(a['yaw_deg'], m.get('yaw_deg'))
                self.assertEqual(a['heading_source'], m.get('heading_source'))
                self.assertEqual(a['retarget_scale'], m['retarget']['scale'])
                self.assertEqual(get(base + f'/episode?id={self.B}')[0], 404)
                self._copy(self.B, replay, slam)                  # appears without a restart
                idx = json.loads(get(base + '/index')[2])
                by = {e['id']: e for e in idx['episodes']}
                self.assertEqual(set(by), {self.A, self.B})
                self.assertEqual((by[self.B]['label'], by[self.B]['group'], by[self.B]['task']), ('ball', 'ball', ''))
                status, _, body = get(base + f'/episode?id={self.B}')
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)['label'], 'ball')
                import shutil
                shutil.rmtree(replay / self.A)                   # removals are picked up too
                ids = [e['id'] for e in json.loads(get(base + '/index')[2])['episodes']]
                self.assertEqual(ids, [self.B])
            finally:
                srv.close()


class SourceTests(unittest.TestCase):
    """Sources come from policy run configs (configs/hub/apps.json ``sources``): their stage outputs."""

    def test_config_source_paths_come_from_the_policy_config(self):
        from policy.configs import schema
        cfg = schema.load('dirt-orb')
        src = wv.config_source({'config': 'dirt-orb'})
        self.assertEqual((src.name, src.suffix, src.method), ('dirt-orb', '', cfg.method))
        self.assertEqual(src.replay_root, REPO / cfg.dataset('labels'))
        self.assertEqual(src.slam_root, REPO / cfg.dataset('reconstructions'))
        self.assertEqual(src.split_file, REPO / cfg.dataset('cohorts') / 'selection.json')
        self.assertEqual(src.data_roots, tuple(dict.fromkeys(REPO / r['data_root']
                                                             for r in cfg.section('source')['recordings'])))
        self.assertEqual(wv.config_source('dirt-orb', primary=False).suffix, '.dirt-orb')
        for bad in (None, {}, {'config': ''}, {'config': 'dirt-orb', 'replay_root': 'x'}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                wv.config_source(bad)
        with self.assertRaises(FileNotFoundError):
            wv.config_source('no-such-config')
        with self.assertRaises(ValueError):
            wv.config_sources(['dirt-orb', {'config': 'dirt-orb'}])

    def test_apps_json_data_review_names_only_policy_configs(self):
        apps = json.loads((REPO / 'configs' / 'hub' / 'apps.json').read_text())['apps']
        opts = next(a for a in apps if a['id'] == 'data-review')['options']
        self.assertEqual(set(opts), {'sources', 'labels_file'})
        self.assertEqual([s.name for s in wv.config_sources(opts['sources'])], ['dirt-orb'])

    def test_start_requires_sources_and_labels_file(self):
        import inspect
        params = inspect.signature(tva.start).parameters
        self.assertIs(params['sources'].default, inspect.Parameter.empty)
        self.assertIs(params['labels_file'].default, inspect.Parameter.empty)
        with tempfile.TemporaryDirectory() as tmp:
            srv = tva.start(None, ['dirt-orb'], str(Path(tmp) / 'labels.json'))
            try:
                self.assertEqual(srv.labels.split_files, [wv.config_source('dirt-orb').split_file])
                self.assertEqual(srv.index.sources[0].name, 'dirt-orb')
            finally:
                srv.close()

    def test_sources_switch_suffix_and_split_exemption(self):
        a, b = 'rec-a', 'rec-b'
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for n in ('one', 'two'):
                fake_episode(tmp / n / 'labels', tmp / n / 'recon', a)
            fake_episode(tmp / 'two' / 'labels', tmp / 'two' / 'recon', b, profile='arm2_calibrated')
            (tmp / 'one' / 'selection.json').write_text(json.dumps({'episodes': [{'id': a, 'split': 'train'}]}))
            one = wv.Source('one', tmp / 'one' / 'labels', tmp / 'one' / 'recon', (tmp / 'data',),
                            tmp / 'one' / 'selection.json', 'orbslam3_smoother')
            two = wv.Source('two', tmp / 'two' / 'labels', tmp / 'two' / 'recon', (), tmp / 'two' / 'selection.json',
                            suffix='.two')
            index = wv.EpisodeIndex([one, two])
            eps = index.get()
            self.assertEqual(set(eps), {a, a + '.two', b + '.two'})
            self.assertEqual((eps[a].source, eps[a + '.two'].source), ('one', 'two'))
            self.assertEqual(eps[a + '.two'].index_entry()['recording'], a)
            self.assertEqual(eps[a].profile, 'full4_calibrated')
            self.assertEqual(eps[b + '.two'].profile, 'full4')                 # profiles the twin no longer builds
            self.assertEqual(index.info()['sources'], [{'id': 'one', 'name': 'one (orbslam3_smoother)'},
                                                       {'id': 'two', 'name': 'two'}])
            self.assertEqual(wv.EpisodeIndex([one]).info(), {})                # one source: no picker
            self.assertEqual(index.split_exempt(), {a + '.two', b + '.two'})   # two has no selection.json yet
            v = tva.DataReview(index, wv.LabelStore(tmp / 'labels.json', index.split_files()))
            try:
                self.assertEqual(v.label_exempt(), {b})                        # a is checked through source one
                self.assertEqual(v.labels.upsert(b, {'verdict': 'good'}, [a, b], v.label_exempt())['verdict'], 'good')
                self.assertEqual(v.labels.upsert(a, {'verdict': 'bad'}, [a, b], v.label_exempt())['verdict'], 'bad')
            finally:
                v.close()

    def test_selection_split_accepts_train_holdout_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'sel.json').write_text(json.dumps({
                'episodes': [{'id': 'a', 'split': 'train'}, {'id': 'b', 'split': 'holdout'}],
                'excluded_by_config': ['c']}))
            (tmp / 'plain.json').write_text(json.dumps({'train': ['d'], 'holdout': []}))
            store = wv.LabelStore(tmp / 'labels.json', [tmp / 'sel.json', tmp / 'plain.json', tmp / 'missing.json'])
            self.assertEqual(store.split(), {'train': ['a', 'd'], 'holdout': ['b'], 'excluded': ['c']})
            ids = ['a', 'b', 'c', 'd', 'e']
            for i in 'abcd':
                store.upsert(i, {'verdict': 'good'}, ids)
            with self.assertRaises(ValueError):
                store.upsert('e', {'verdict': 'good'}, ids)
            self.assertIsNone(wv.LabelStore(tmp / 'labels.json', tmp / 'missing.json').split())
            self.assertEqual(wv.DEFAULT_PROMPT, 'scoop the dirt')

    def test_read_labels_over_several_data_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.assertEqual(wv.read_labels(tmp), {})
            for root, label in (('first', 'dig'), ('second', 'ball')):
                (tmp / root / 'x').mkdir(parents=True)
                (tmp / root / 'x' / 'metadata.json').write_text(json.dumps({'label': label}))
            (tmp / 'second' / 'y').mkdir()
            (tmp / 'second' / 'y' / 'metadata.json').write_text(json.dumps({'label': 'ball'}))
            self.assertEqual(wv.read_labels([tmp / 'first', tmp / 'second', tmp / 'none']),
                             {'x': {'label': 'dig', 'task': ''}, 'y': {'label': 'ball', 'task': ''}})
        self.assertEqual(wv.read_labels(None), {})


def _post(client, path, body, ctype='application/json'):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    r = client.post(path, content=data, headers={'Content-Type': ctype} if ctype else {})
    return r.status_code, json.loads(r.content or b'{}')


class LabelServerTests(unittest.TestCase):
    """Annotation API (soil_success_labels.v2) on a fixed fake episode table, temp labels + split files."""
    TRAIN = ['20260927T070203Z-bf0ec1b6', '20260927T070234Z-baea98b3', '20260927T070252Z-d408a7ef']
    HOLD = ['20260927T095739Z-8eda6a95']
    OTHER = 'synthetic'                        # shown by the viewer but not in the study split

    def setUp(self):
        from types import SimpleNamespace
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.labels_path, split = tmp / 'out' / 'labels.json', tmp / 'split.json'
        split.write_text(json.dumps({'train': self.TRAIN, 'holdout': self.HOLD}))
        eps = {i: SimpleNamespace(label='') for i in self.TRAIN + self.HOLD + [self.OTHER]}
        eps[self.TRAIN[2]] = SimpleNamespace(label='ball')
        self.srv = tva.DataReview(eps, labels=wv.LabelStore(self.labels_path, split))
        self.client = hub_client(self.srv)
        self.base = '/api/data-review'

    def tearDown(self):
        self.srv.close(); self.tmp.cleanup()

    def get(self, path):
        return _get(self.client, path)

    def post(self, path, body, ctype='application/json'):
        return _post(self.client, path, body, ctype)

    def summary(self):
        status, _, body = self.get(self.base + '/labels/summary')
        self.assertEqual(status, 200)
        return json.loads(body)

    def test_empty_template(self):
        status, _, body = self.get(self.base + '/labels')
        d = json.loads(body)
        self.assertEqual(status, 200)
        d.pop('resolved_prompts')
        self.assertEqual(d, {'schema': 'soil_success_labels.v2', 'default_prompt': wv.DEFAULT_PROMPT,
                             'group_prompts': {'dig': '', 'ball': ''}, 'annotated_at': None,
                             'labels': {}, 'pending': {}})
        self.assertFalse(self.labels_path.exists())
        s = self.summary()
        self.assertEqual((s['total'], s['unlabeled'], s['labeled'], s['train_ready']), (5, 5, 0, False))

    def test_round_trip_summary_and_readiness(self):
        a, b, c = self.TRAIN
        st, out = self.post(self.base + f'/labels/{a}', {'verdict': 'good'})      # no reviewer needed
        self.assertEqual(st, 200, out)
        self.assertEqual(set(out['entry']), {'success', 'verdict', 'updated_at'})
        self.assertEqual((out['entry']['success'], out['entry']['verdict']), (True, 'good'))
        self.assertEqual(self.post(self.base + f'/labels/{b}', {'verdict': 'bad'})[0], 200)
        self.assertEqual(self.post(self.base + f'/labels/{c}', {'verdict': 'skip'})[0], 200)
        d = json.loads(self.get(self.base + '/labels')[2])
        resolved = d.pop('resolved_prompts')
        self.assertEqual(json.loads(self.labels_path.read_text()), d)
        self.assertEqual(set(d), {'schema', 'default_prompt', 'group_prompts', 'annotated_at', 'labels', 'pending'})
        self.assertEqual(d['schema'], 'soil_success_labels.v2')
        self.assertTrue(d['annotated_at'])
        self.assertEqual(set(d['labels']), {a, b})
        self.assertEqual(d['labels'][b]['success'], False)
        self.assertEqual(set(d['pending'][c]), {'verdict', 'updated_at'})
        self.assertEqual(d['pending'][c]['verdict'], 'skip')
        self.assertEqual(resolved[a], {'prompt': wv.DEFAULT_PROMPT, 'source': 'default', 'group': 'dig'})
        s = self.summary()
        self.assertEqual({k: s[k] for k in ('good', 'bad', 'skip', 'unlabeled', 'labeled')},
                         {'good': 1, 'bad': 1, 'skip': 1, 'unlabeled': 2, 'labeled': 2})
        self.assertEqual(s['train']['unlabeled'], [c])
        self.assertEqual((s['holdout']['labeled'], s['train_ready']), (0, False))
        self.assertEqual(s['verdicts'][c], 'skip')
        self.post(self.base + f'/labels/{c}', {'verdict': 'good'})           # skip -> good moves the entry
        s = self.summary()
        self.assertTrue(s['train_ready'])
        self.assertFalse(s['all_ready'])
        self.assertNotIn(c, json.loads(self.labels_path.read_text())['pending'])
        self.post(self.base + f'/labels/{c}', {'verdict': None})              # clearing -> unlabeled
        s = self.summary()
        self.assertEqual((s['unlabeled'], s['train_ready']), (3, False))
        self.assertNotIn(c, json.loads(self.labels_path.read_text())['labels'])
        self.assertEqual(list(self.labels_path.parent.glob('*.partial')), [])  # atomic writes leave nothing

    def test_prompt_meta_resolution_and_readiness(self):
        a, b, ball = self.TRAIN
        meta = self.base + '/labels/meta'
        s = self.summary()
        self.assertEqual((s['prompts']['resolved'], s['prompts_ready']), (5, True))   # built-in default
        st, out = self.post(meta, {'default_prompt': '  scoop   soil ', 'group_prompts': {'ball': 'scoop the ball'}})
        self.assertEqual(st, 200, out)
        self.assertEqual(out['default_prompt'], 'scoop soil')
        self.assertEqual(out['group_prompts'], {'dig': '', 'ball': 'scoop the ball'})
        self.post(meta, {'group_prompts': {'dig': 'dig here'}})                     # partial update keeps ball
        self.assertEqual(json.loads(self.labels_path.read_text())['group_prompts'], {'dig': 'dig here', 'ball': 'scoop the ball'})
        # a prompt without a verdict is kept under pending (verdict null); a later verdict keeps it
        st, out = self.post(self.base + f'/labels/{a}', {'prompt': 'scoop left pile'})
        self.assertEqual((st, out['entry']['verdict'], out['entry']['prompt']), (200, None, 'scoop left pile'))
        self.assertEqual(self.summary()['verdicts'][a], None)
        self.assertEqual(self.post(self.base + f'/labels/{a}', {'verdict': 'good'})[0], 200)
        d = json.loads(self.get(self.base + '/labels')[2])
        self.assertEqual(d['labels'][a], {'success': True, 'verdict': 'good', 'prompt': 'scoop left pile',
                                          'updated_at': d['labels'][a]['updated_at']})
        self.assertNotIn(a, d['pending'])
        r = d['resolved_prompts']
        self.assertEqual((r[a]['prompt'], r[a]['source']), ('scoop left pile', 'episode'))
        self.assertEqual((r[b]['prompt'], r[b]['source']), ('dig here', 'group'))
        self.assertEqual((r[ball]['prompt'], r[ball]['source'], r[ball]['group']), ('scoop the ball', 'group', 'ball'))
        self.post(meta, {'group_prompts': {'dig': ''}})                             # falls back to default
        self.assertEqual(json.loads(self.get(self.base + '/labels')[2])['resolved_prompts'][b]['prompt'], 'scoop soil')
        self.post(self.base + f'/labels/{a}', {'prompt': ''})                   # clear override, verdict kept
        d = json.loads(self.get(self.base + '/labels')[2])
        self.assertNotIn('prompt', d['labels'][a])
        self.assertEqual(d['labels'][a]['verdict'], 'good')
        self.assertEqual(d['resolved_prompts'][a]['source'], 'default')
        self.post(self.base + f'/labels/{b}', {'prompt': 'tmp'})                # prompt-only entry ...
        st, out = self.post(self.base + f'/labels/{b}', {'prompt': ''})         # ... removed when cleared
        self.assertEqual((st, out['entry']), (200, None))
        self.assertNotIn(b, json.loads(self.labels_path.read_text())['pending'])
        # prompts_ready: every train episode needs a non-empty resolved prompt
        self.post(meta, {'default_prompt': ''})
        s = self.summary()
        self.assertEqual((s['prompts']['resolved'], s['prompts_ready']), (1, False))   # only the ball group
        self.assertEqual(s['resolved_prompts'][ball], 'scoop the ball')
        self.post(self.base + f'/labels/{a}', {'prompt': 'p a'})
        self.assertFalse(self.summary()['prompts_ready'])
        self.post(self.base + f'/labels/{b}', {'prompt': 'p b'})
        s = self.summary()
        self.assertTrue(s['prompts_ready'])
        self.assertEqual(s['prompts']['missing'], [self.HOLD[0], self.OTHER])
        # validation
        for body in ({'default_prompt': 'x' * 513}, {'default_prompt': 'a\nb'}, {'default_prompt': 'a\tb'},
                     {'default_prompt': 5}, {'group_prompts': ['ball']}, {'group_prompts': {'ball': 'a\x07'}},
                     {'group_prompts': {'bad group!': 'x'}}):
            self.assertEqual(self.post(meta, body)[0], 400, body)
        url = self.base + f'/labels/{b}'
        for body in ({'prompt': 'y' * 513}, {'prompt': 'a\rb'}, {'prompt': 3}):
            self.assertEqual(self.post(url, body)[0], 400, body)
        self.assertEqual(self.post(url, {'prompt': 'z' * 512})[0], 200)

    def test_bad_requests(self):
        ep = self.TRAIN[0]
        url = self.base + f'/labels/{ep}'
        self.assertEqual(self.post(url, {'verdict': 'good'})[0], 200)
        for body in (b'', b'{not json', b'[1, 2]', b'"x"', {}, {'verdict': 'meh'}, {'verdict': 'good', 'x': 1},
                     {'verdict': 'good', 'notes': 'gone'}, {'verdict': 'good', 'reviewer': 'Ada'},
                     {'verdict': 'good', 'success': True}, {'verdict': 'good', 't': 1.0}):
            self.assertEqual(self.post(url, body)[0], 400, body)
        self.assertEqual(self.post(url, {'verdict': 'good'}, ctype='text/plain')[0], 400)
        self.assertEqual(self.post(self.base + '/labels/nope', {'verdict': 'good'})[0], 404)
        self.assertEqual(self.post(self.base + f'/labels/{self.OTHER}', {'verdict': 'good'})[0], 400)
        for body in ({}, {'reviewer': 'Ada'}, {'criterion': 'x'}, {'owner': 'x'}):
            self.assertEqual(self.post(self.base + '/labels/meta', body)[0], 400, body)
        self.assertEqual(self.post('/api/other', {})[0], 404)
        self.assertEqual(json.loads(self.labels_path.read_text())['labels'][ep]['success'], True)

    def test_v1_file_is_migrated(self):
        a, b, c = self.TRAIN
        self.labels_path.parent.mkdir(parents=True)
        self.labels_path.write_text(json.dumps({
            'schema': 'soil_success_labels.v1', 'reviewer': 'Ada', 'criterion': 'full scoop',
            'default_prompt': 'scoop the dirt', 'group_prompts': {'dig': 'd', 'ball': 'scoop the ball out of the dirt'},
            'annotated_at': '2026-09-27T10:00:00+00:00', 'reviewed_at': '2026-09-27T10:00:00+00:00',
            'labels': {a: {'success': True, 'verdict': 'good', 'notes': 'n', 'flags': ['scale looks off'], 't_s': 4.0,
                           'evidence': 'e', 'reviewer': 'Ada', 'updated_at': 'T1', 'prompt': 'grab it'}},
            'pending': {b: {'success': None, 'verdict': 'skip', 'notes': '', 'flags': [], 't_s': None,
                            'evidence': 'e', 'reviewer': 'Ada', 'updated_at': 'T2'},
                        c: {'success': None, 'verdict': None, 'evidence': 'e', 'updated_at': 'T3', 'prompt': 'p'},
                        self.HOLD[0]: {'success': None, 'verdict': None, 'notes': 'x', 'updated_at': 'T4'}}}))
        want = {'schema': 'soil_success_labels.v2', 'default_prompt': 'scoop the dirt',
                'group_prompts': {'dig': 'd', 'ball': 'scoop the ball out of the dirt'},
                'annotated_at': '2026-09-27T10:00:00+00:00',
                'labels': {a: {'success': True, 'verdict': 'good', 'prompt': 'grab it', 'updated_at': 'T1'}},
                'pending': {b: {'verdict': 'skip', 'updated_at': 'T2'},
                            c: {'verdict': None, 'prompt': 'p', 'updated_at': 'T3'}}}
        d = json.loads(self.get(self.base + '/labels')[2])
        self.assertEqual(d.pop('resolved_prompts')[c]['prompt'], 'p')
        self.assertEqual(d, want)
        self.assertEqual(self.summary()['verdicts'][b], 'skip')
        self.post(self.base + f'/labels/{self.HOLD[0]}', {'verdict': 'bad'})    # first save rewrites as v2
        f = json.loads(self.labels_path.read_text())
        self.assertEqual(f['schema'], 'soil_success_labels.v2')
        self.assertEqual(f['labels'][a], want['labels'][a])
        self.assertFalse({'reviewer', 'criterion', 'reviewed_at'} & set(f))


if __name__ == '__main__':
    unittest.main()
