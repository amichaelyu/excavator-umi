"""Local web visualizer: MuJoCo excavator twin next to the bucket webcam, time-synchronized.

Domain code of the hub's twin replay viewer (hub app ``twin-viewer``: routes in hub/apps/twin_viewer/__init__.py,
page in hub/web/views/twin-viewer/); its options are set in configs/hub/apps.json or saved from the hub page into
configs/hub/local.json. This module has no HTTP code: episode loading, the time mapping, path errors, the offscreen
twin renderer and the success labels store (the live-camera fetch and HTTP Range are in hub/apps/_common.py).

Every source is the output of one policy run config (configs/policy/<name>.yaml, listed under the app's
``sources`` option in configs/hub/apps.json): the reconstruct stage's trajectories (``dataset/reconstructions/<name>``,
the "VIO root") and the labels stage's twin IK replays of them (``dataset/labels/<name>``, the replay root; made by
``python -m policy labels <name>`` with the config's labels options), with the original videos from the config's
``source.recordings`` data roots. ORB-SLAM3 reconstructions have no VIO overlay video, so the camera pane shows the
original video.mp4. When there is more than one source the page shows a source picker; episodes of every source but
the first have ids ``<recording id>.<config name>``.

``dz`` (episode detail and sim frames, metres, +-MAX_DZ_M): raise / lower the episode's
retargeted tip path in the model frame; the twin is re-solved on the shifted targets (IK, about
0.1 s per episode) and there is no replay video for it (server render).

The twin is the single four-joint chain of digital_twin.sim.kinematics. Replays made with it
(``fk_dynamic.csv`` has ``qpos_slew``) are shown as recorded, with their replay.mp4 when rendered. Older
two-joint replays are re-solved from their actions.csv tip / pitch targets with the same IK on
first use (no video: the page renders on the server).

Labels (LabelStore): ``soil_success_labels.v2`` (a v1 file is migrated): per recording a verdict
(good | bad | skip) and an optional task prompt override; task prompt per episode = episode prompt > group
prompt > default_prompt. Recordings of a source whose config has a cohort selection
(``dataset/cohorts/<name>/selection.json``) must be in it (train, holdout or ``source.exclude``).

Time bases (all in seconds):
  * episode time t: VIO ``trajectory.csv`` t_s, i.e. raw camera start-of-exposure time minus
    ``summary.json time_origin.raw_camera_timestamp_ns_at_t0`` (the first kept frame).
  * replay time comes from actions.csv source_t_s -> t_s when present, including local slowdown.
    Older exports use t * time_scale. replay.mp4 is 20 fps CFR starting at replay time 0.
  * overlay.mp4 is 20 fps CFR, one frame per processed VIO frame. Frame k is VIO row k, so
    t -> (interp(t, frame_t, k) + 0.5) / 20.
  * original video.mp4 is VFR. Its frame pts equal the frames.csv ``camera_timestamp_ns`` minus the
    first recorded frame's timestamp, and all eight episodes match exactly. So
    t -> t + (t0_ns - first_ns) / 1e9.
"""
from __future__ import annotations

import csv
import json
import os
import re
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from digital_twin.paths import REPO
from policy.data import local_config
from policy.data.data_root import is_lfs_pointer
from policy.configs.schema import PROFILES, TASKS

JOINTS = ['slew', 'boom', 'stick', 'bucket']   # = kinematics.JOINTS (not imported: mujoco is loaded lazily)
SLEW = JOINTS.index('slew')                     # the slew is locked: always shown at 0 (facing forward)


def lock_slew(q) -> np.ndarray:
    """Joint rows ``q`` with the slew at 0: the machine never turns its base (display only)."""
    q = np.array(q, float)
    q[..., SLEW] = 0.0
    return q
CAMERAS = ('overview_close', 'overview', 'side', 'bucket_camera')
MODES = ('twin',)                                # one twin, one FK / IK
REPLAY_MEDIA = ('replay.mp4', 'joints.png')
MAX_DZ_M = 0.5                                   # height knob range (see dz above)
VIO_MEDIA = ('overlay.mp4', 'trajectory.png')
OVERLAY_FPS = 20.0
FRAME_W, FRAME_H = 640, 480
EPISODE_RE = re.compile(r'^[A-Za-z0-9_.-]{1,80}$')
LIVE_RETRY_S = 2.0
DEFAULT_LIVE_URL = local_config.pi_urls()[0] + '/preview.jpg'   # hosts.pi in configs/hub/local.json


# ------------------------------------------------------------------ time mapping (pure)
def overlay_video_time(t, frame_t, fps: float = OVERLAY_FPS):
    """Episode time -> overlay.mp4 time (mid-frame of the VIO frame at t; piecewise linear)."""
    frame_t = np.asarray(frame_t, float)
    k = np.interp(t, frame_t, np.arange(len(frame_t), dtype=float))
    return (k + 0.5) / fps


def overlay_episode_time(v, frame_t, fps: float = OVERLAY_FPS):
    """Inverse of :func:`overlay_video_time`."""
    frame_t = np.asarray(frame_t, float)
    return np.interp(np.asarray(v, float) * fps - 0.5, np.arange(len(frame_t), dtype=float), frame_t)


def original_video_time(t, offset_s: float):
    """Episode time -> original video.mp4 time (pts = camera timestamp - first frame timestamp)."""
    return np.asarray(t, float) + offset_s


def original_episode_time(v, offset_s: float):
    return np.asarray(v, float) - offset_s


def replay_video_time(t, time_scale: float, clock=None):
    """Episode time -> replay.mp4 / fk csv time."""
    if clock is not None:
        return np.interp(t, clock['source_t_s'], clock['replay_t_s'])
    return np.asarray(t, float) * time_scale


def replay_episode_time(r, time_scale: float, clock=None):
    if clock is not None:
        return np.interp(r, clock['replay_t_s'], clock['source_t_s'])
    return np.asarray(r, float) / time_scale


# ------------------------------------------------------------------ live URL handling (pure)
def validate_live_url(url: str) -> tuple[str, str]:
    """Check a live-camera URL and split off an access token.

    Accepts only absolute http(s) URLs with a host and no user:password. The umi-data-collection dashboard's
    private link carries ``?token=...``. That token is removed from the URL and returned
    separately, and the proxy sends it as the dashboard's ``umi_access`` cookie. Raises
    ValueError on anything else.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError('live URL is empty')
    url = url.strip()
    if any(c in url for c in '\r\n\t '):
        raise ValueError('live URL contains whitespace')
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https'):
        raise ValueError(f'live URL must be http:// or https:// (got {parts.scheme or "no scheme"!r})')
    if not parts.hostname:
        raise ValueError('live URL has no host')
    if parts.username or parts.password:
        raise ValueError('put credentials in ?token= (umi-data-collection) rather than user:password@')
    try:
        port = parts.port
    except ValueError as ex:
        raise ValueError(f'live URL has an invalid port: {ex}') from None
    if port == 0:
        raise ValueError('live URL has an invalid port: 0')
    query = parse_qsl(parts.query, keep_blank_values=True)
    token = ''.join(v for k, v in query if k == 'token')
    rest = urlencode([(k, v) for k, v in query if k != 'token'])
    path = parts.path or '/'
    return urlunsplit((parts.scheme, parts.netloc, path, rest, '')), token


def redact_url(url: str) -> str:
    return re.sub(r'([?&]token=)[^&#]*', r'\1***', url or '')


# ------------------------------------------------------------------ data loading
def _read_csv(path: Path) -> dict[str, list[str]]:
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    return {h: [r[i] for r in body] for i, h in enumerate(head)}


def _floats(col) -> np.ndarray:
    return np.array([float(x) for x in col], float)


def camera_first_timestamp_ns(tar_path: Path) -> tuple[int, int] | None:
    """(first camera_timestamp_ns, frame count) from recording.tar/frames.csv, or None."""
    if not tar_path.is_file() or is_lfs_pointer(tar_path):
        return None
    try:
        with tarfile.open(tar_path) as tf:
            f = tf.extractfile('frames.csv')
            rows = list(csv.DictReader(line.decode() for line in f))
        return int(rows[0]['camera_timestamp_ns']), len(rows)
    except (OSError, KeyError, IndexError, ValueError, tarfile.TarError):
        return None


@dataclass
class Episode:
    id: str
    replay_dir: Path
    vio_dir: Path
    metrics: dict
    summary: dict
    frame_t: np.ndarray                    # VIO frame times (episode s)
    status: list
    original_video: Path | None = None
    video_offset_s: float | None = None
    video_frames: int | None = None
    label: str = ''
    task: str = ''
    source: str = ''                       # the Source (policy config) it came from
    recording: str = ''                    # recording id in the data root when it differs from id
    stamp: tuple = ()                      # file mtimes the entry was built from (cache key)
    _fk: dict = field(default_factory=dict)
    _targets: np.ndarray | None = None
    _path_error: dict | None = None        # path_error() at dz = 0 (the overall summary reuses it)
    _replay_clock: dict | None = None

    @property
    def replay_clock(self):
        if self._replay_clock is None and (self.replay_dir / 'actions.csv').is_file():
            actions = _read_csv(self.replay_dir / 'actions.csv')
            if 'source_t_s' in actions:
                source, replay = _floats(actions['source_t_s']), _floats(actions['t_s'])
                if (len(source) < 2 or not np.isfinite([source, replay]).all()
                        or np.any(np.diff(source) <= 0) or np.any(np.diff(replay) <= 0)):
                    raise ValueError(f'{self.id}: invalid recording-to-replay clock')
                self._replay_clock = {'source_t_s': source.tolist(), 'replay_t_s': replay.tolist()}
            elif self.metrics.get('source_time', {}).get('nonuniform'):
                raise ValueError(f'{self.id}: nonuniform replay is missing source_t_s')
            else:
                # Empty dict caches a legacy export without a source clock.
                self._replay_clock = {}
        return self._replay_clock or None

    def replay_time(self, t):
        return replay_video_time(t, self.time_scale, self.replay_clock)

    @property
    def time_scale(self) -> float:
        return float(self.metrics.get('time_scale', 1.0))

    @property
    def duration(self) -> float:
        replay_end = float(replay_episode_time(self.metrics.get('duration_s', 0.0),
                                              self.time_scale, self.replay_clock))
        return float(max(self.frame_t[-1] if len(self.frame_t) else 0.0, replay_end))

    def media(self) -> dict[str, Path]:
        out = {n: self.replay_dir / n for n in REPLAY_MEDIA if (self.replay_dir / n).is_file()}
        out.update({n: self.vio_dir / n for n in VIO_MEDIA if (self.vio_dir / n).is_file()})
        if self.original_video is not None:
            out['video.mp4'] = self.original_video
        return out

    @property
    def profile(self) -> str:
        """Twin profile of this replay (labels.profile of its config); full4 for older replays without one."""
        p = self.metrics.get('profile')
        return p if p in PROFILES else 'full4'

    @property
    def recorded(self) -> bool:
        """Made with the four-joint twin: fk_dynamic.csv (and replay.mp4) show it as recorded."""
        f = self.replay_dir / 'fk_dynamic.csv'
        if not f.is_file():
            return False
        with open(f) as fh:
            return 'qpos_slew' in fh.readline().split(',')

    def has_mode(self, mode: str = 'twin') -> bool:
        return mode in MODES and (self.replay_dir / 'actions.csv').is_file()

    def fk(self, mode: str = 'twin', dz: float = 0.0) -> dict:
        """The twin's track; ``dz`` != 0: re-solved on the targets raised by dz metres."""
        key = int(round(dz * 1000))                     # mm
        if key:
            if key not in self._fk:
                from digital_twin.sim.replay import solve_actions
                c = _read_csv(self.replay_dir / 'actions.csv')
                acts = {k: _floats(v) for k, v in c.items()
                        if k in ('t_s', 'tip_target_x', 'tip_target_y', 'tip_target_z', 'pitch_target', 'tip_target_lateral_m')}
                acts['tip_target_z'] = acts['tip_target_z'] + key / 1000
                for k in [k for k in self._fk if k not in ('twin', key)]:
                    del self._fk[k]                     # keep the recorded track and the latest height only
                self._fk[key] = {**solve_actions(acts, self.profile), 'video': None}
            return self._fk[key]
        if 'twin' not in self._fk:
            if self.recorded:
                c = _read_csv(self.replay_dir / 'fk_dynamic.csv')
                self._fk['twin'] = {'t': _floats(c['t_s']),
                                    'q': np.stack([_floats(c['qpos_' + j]) for j in JOINTS], 1),
                                    'tip': np.stack([_floats(c['tip_' + a]) for a in 'xyz'], 1),
                                    'tip_err': _floats(c['tip_err_m']), 'video': 'replay.mp4'}
            else:                         # two-joint replay: re-solve its targets on the twin
                from digital_twin.sim.replay import solve_actions
                c = _read_csv(self.replay_dir / 'actions.csv')
                self._fk['twin'] = {**solve_actions({k: _floats(v) for k, v in c.items()
                                                     if k in ('t_s', 'tip_target_x', 'tip_target_y', 'tip_target_z',
                                                              'pitch_target', 'tip_target_lateral_m')},
                                                    self.profile), 'video': None}
        return self._fk['twin']

    def targets(self, dz: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._targets is None:
            c = _read_csv(self.replay_dir / 'actions.csv')
            self._targets = (_floats(c['t_s']), np.stack([_floats(c['tip_target_' + a]) for a in 'xyz'], 1),
                             np.array([int(float(x)) for x in c['reachable']], bool))
        if not round(dz * 1000):
            return self._targets
        t, tg, _ = self._targets
        f = self.fk('twin', dz)
        return t, tg + [0.0, 0.0, round(dz * 1000) / 1000], f['tip_err'] <= 2e-3

    def path_error(self, dz: float = 0.0) -> dict:
        """How far the twin's paths are from the SLAM path (the retargeted UMI tip targets and pitch
        targets in actions.csv): MSE / RMSE of the tip position (model mm, and UMI mm = model / retarget
        scale) and of the bucket angle (deg). ``ik``: the commanded IK path (fk_kinematic.csv, or the
        re-solved track); ``twin``: the actuated twin (fk_dynamic.csv), when it was recorded."""
        if not round(dz * 1000) and self._path_error is not None:
            return self._path_error
        from digital_twin.sim.kinematics import pitch_of
        c = _read_csv(self.replay_dir / 'actions.csv')
        _, tg, _ = self.targets(dz)
        phi = _floats(c['pitch_target']) if 'pitch_target' in c else None
        scale = float(self.metrics.get('retarget', {}).get('scale') or 0) or None

        def stats(tip, q):
            n = min(len(tip), len(tg))
            e2 = np.sum((tip[:n] - tg[:n]) ** 2, 1) * 1e6           # mm^2
            out = {'tip_mse_mm2': float(e2.mean()), 'tip_rmse_mm': float(np.sqrt(e2.mean())),
                   'tip_rmse_umi_mm': float(np.sqrt(e2.mean()) / scale) if scale else None, 'samples': int(n)}
            if phi is not None and q is not None:
                a2 = np.degrees(pitch_of(q[:n]) - phi[:n]) ** 2
                out.update(angle_mse_deg2=float(a2.mean()), angle_rmse_deg=float(np.sqrt(a2.mean())))
            return out

        if self.recorded and not round(dz * 1000):
            k = _read_csv(self.replay_dir / 'fk_kinematic.csv')
            ik = stats(np.stack([_floats(k['tip_' + a]) for a in 'xyz'], 1),
                       np.stack([_floats(k['q_' + j]) for j in JOINTS], 1))
            f = self.fk('twin')
            twin = stats(f['tip'], f['q'])
        else:
            f = self.fk('twin', dz)
            ik, twin = stats(f['tip'], f['q']), None
        out = {'ik': ik, 'twin': twin, 'retarget_scale': scale}
        if not round(dz * 1000):
            self._path_error = out
        return out

    def index_entry(self) -> dict:
        s = self.summary
        media = self.media()
        m = self.metrics
        return {
            'id': self.id, 'duration_s': round(self.duration, 4), 'time_scale': self.time_scale,
            'label': self.label, 'task': self.task, 'group': self.label or 'dig', 'source': self.source,
            'recording': self.recording or self.id,
            'retarget_scale': m.get('retarget', {}).get('scale'),
            'yaw_deg': m.get('yaw_deg', m.get('retarget', {}).get('yaw_deg')),
            'heading_source': m.get('heading_source'),
            'mount': m.get('mount') or ({'mount_source': 'not recorded (older replay)',
                                         'bucket_point_in_camera_m': m.get('camera_to_bucket', {}).get('t'),
                                         'rotation_quat_wxyz': m.get('camera_to_bucket', {}).get('q_wxyz')}
                                        if m.get('camera_to_bucket') else None),
            'frames': {'camera_total': s.get('frames_total'), 'vio_processed': s.get('frames_processed'),
                       'vio_tracked': s.get('frames_tracked'), 'vio_imu_only': s.get('frames_imu_only'),
                       'overlay': len(self.frame_t), 'replay': self.metrics.get('samples'),
                       'original_video': self.video_frames},
            'media': {n: f'media/{self.id}/{n}' for n in media},   # relative: the page may sit under a path prefix (hub/)
            'original_video_synced': self.video_offset_s is not None,
        }

    def detail(self, dz: float = 0.0) -> dict:
        # slew locked at 0; replay.mp4 has the recorded slew baked in, so frames are server-rendered instead
        fk = {m: {'t': f['t'].round(4).tolist(), 'q': lock_slew(f['q']).round(5).tolist(), 'joints': JOINTS,
                  'tip': f['tip'].round(5).tolist(), 'tip_err': f['tip_err'].round(5).tolist(),
                  'video': None, 'profile': self.profile}
              for m in MODES if self.has_mode(m) for f in [self.fk(m, dz)]}
        tz = self.targets(dz)[1][:, 2] if self.has_mode() else np.zeros(1)
        return {**self.index_entry(), 'dz': round(dz * 1000) / 1000,
                'path_error': self.path_error(dz) if self.has_mode() else None,
                'height': {'start_tip_z_m': float(tz[0]), 'min_tip_z_m': float(tz.min()), 'max_tip_z_m': float(tz.max()),
                           'max_dz_m': MAX_DZ_M}, 'frame_t': self.frame_t.round(6).tolist(), 'status': self.status,
                'overlay_fps': OVERLAY_FPS, 'video_offset_s': self.video_offset_s, 'fk': fk, 'joints': JOINTS,
                'replay_clock': self.replay_clock,
                'time_origin': self.summary.get('time_origin', {}),
                'retarget': {**{k: self.metrics.get('retarget', {}).get(k) for k in ('scale', 'yaw_deg', 'fill_fraction')},
                             'heading_source': self.metrics.get('heading_source')}}


def overall_path_error(episodes) -> dict:
    """Episode.path_error() pooled over every episode (dz = 0): MSE weighted by sample count, so the
    RMSE is over all samples of all episodes, plus the median and worst episode RMSE. ``bands``: the
    episode RMSE tertiles; good <= good_max < medium <= medium_max < bad."""
    per = {e.id: e.path_error() for e in episodes if e.has_mode()}

    def pool(mode):
        rows = [(i, p[mode]) for i, p in per.items() if p.get(mode)]
        if not rows:
            return None
        n = np.array([r['samples'] for _, r in rows], float)
        out = {'episodes': len(rows), 'samples': int(n.sum())}
        for key, unit in (('tip', 'mm'), ('tip_umi', 'mm'), ('angle', 'deg')):
            rm = 'tip_rmse_umi_mm' if key == 'tip_umi' else f'{key}_rmse_{unit}'
            ok = [(k, r[rm]) for k, (_, r) in enumerate(rows) if r.get(rm) is not None]
            if not ok:
                continue
            idx, v = np.array([k for k, _ in ok]), np.array([x for _, x in ok], float)
            mse = float(np.sum(v ** 2 * n[idx]) / n[idx].sum())
            worst = int(np.argmax(v))
            good_max, medium_max = np.percentile(v, [100 / 3, 200 / 3])
            out[key] = {f'mse_{unit}2': mse, f'rmse_{unit}': float(np.sqrt(mse)),
                        f'median_episode_rmse_{unit}': float(np.median(v)),
                        f'worst_episode_rmse_{unit}': float(v[worst]), 'worst_episode': rows[idx[worst]][0],
                        'bands': {'good_max': float(good_max), 'medium_max': float(medium_max)}}
        return out

    return {'ik': pool('ik'), 'twin': pool('twin')}


def parse_dz(value: str | None) -> float | None:
    """``dz`` query value in metres (default 0), or None when it is not a number within +-MAX_DZ_M."""
    try:
        dz = float(value or 0)
    except ValueError:
        return None
    return dz if np.isfinite(dz) and abs(dz) <= MAX_DZ_M else None


def recording_of(eid: str, ep) -> str:
    """The recording an episode id belongs to (labels are per recording)."""
    return getattr(ep, 'recording', '') or eid


def mtime_ns(p: Path) -> float:
    """``p``'s st_mtime_ns, -1 when it is missing (index rescan signatures)."""
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return -1


def _roots(data_root) -> list[Path]:
    """A data root, a sequence of them, or None -> list of Paths."""
    if data_root is None:
        return []
    return [Path(data_root)] if isinstance(data_root, (str, Path)) else [Path(r) for r in data_root]


def read_labels(data_root) -> dict[str, dict]:
    """{id: {'label', 'task'}} from each <data_root>/<id>/metadata.json over one or more data roots (the first
    root holding an id wins; missing roots are skipped). Recordings carry only a label; task stays empty."""
    labels = {}
    for root in reversed(_roots(data_root)):
        if not root.is_dir():
            continue
        for f in sorted(root.glob('*/metadata.json')):
            try:
                label = json.loads(f.read_text()).get('label')
            except (OSError, ValueError):
                continue
            labels[f.parent.name] = {'label': (label or '').strip(), 'task': ''}
    return labels


def load_episodes(replay_root: Path, vio_root: Path, data_root,
                  cache: dict[str, Episode] | None = None, labels: dict | None = None,
                  source: str = '', suffix: str = '') -> dict[str, Episode]:
    """Scan the replay root (``data_root``: one or more data roots, see read_labels). Entries in ``cache`` whose
    source files are unchanged are reused, so a rescan only parses new or rewritten episodes. Episode ids are the
    recording id + ``suffix`` (``.<config>`` for every source but the first), so a recording in several sources
    is listed once per source."""
    eps = {}
    if not replay_root.is_dir():
        return eps
    labels = read_labels(data_root) if labels is None else labels
    for d in sorted(replay_root.iterdir()):
        v = vio_root / d.name
        if not ((d / 'metrics.json').is_file() and (d / 'fk_dynamic.csv').is_file()
                and (v / 'trajectory.csv').is_file() and EPISODE_RE.match(d.name)):
            continue
        stamp = tuple(mtime_ns(x) for x in (d / 'metrics.json', d / 'fk_dynamic.csv', d / 'fk_kinematic.csv',
                                          d / 'actions.csv', v / 'trajectory.csv', v / 'summary.json'))
        lab = labels.get(d.name, {})
        eid = d.name + suffix
        old = (cache or {}).get(eid)
        if old is not None and old.stamp == stamp and old.source == source:
            old.label, old.task = lab.get('label', ''), lab.get('task', '')
            eps[eid] = old
            continue
        eps[eid] = make_episode(eid, d, v, data_root, recording=d.name if suffix else '', label=lab.get('label', ''),
                                task=lab.get('task', ''), source=source, stamp=stamp)
    return eps


def make_episode(eid: str, replay_dir: Path, vio_dir: Path, data_root, *, recording: str = '',
                 label: str = '', task: str = '', source: str = '', stamp: tuple = ()) -> Episode:
    """One Episode from its replay dir (metrics.json, fk_dynamic.csv, ...) and VIO dir (trajectory.csv,
    summary.json). The original video comes from <data_root>/<recording or eid>/ (the first data root that has
    the recording)."""
    tr = _read_csv(vio_dir / 'trajectory.csv')
    summary = json.loads((vio_dir / 'summary.json').read_text()) if (vio_dir / 'summary.json').is_file() else {}
    ep = Episode(eid, replay_dir, vio_dir, json.loads((replay_dir / 'metrics.json').read_text()), summary,
                 _floats(tr['t_s']), tr['status'], label=label, task=task, source=source,
                 recording=recording, stamp=stamp)
    src = next((r / (recording or eid) for r in _roots(data_root) if (r / (recording or eid)).is_dir()), None)
    if src is not None:
        # video-recovered.mp4: a viewing copy without undecodable frames (timestamp gaps kept) that
        # replaces video.mp4 for recordings flagged video_recovered
        vid = next((x for x in (src / 'video.mp4', src / 'video-recovered.mp4')
                    if x.is_file() and not is_lfs_pointer(x)), None)
        if vid is not None:
            ep.original_video = vid
            first = camera_first_timestamp_ns(src / 'recording.tar')
            t0 = summary.get('time_origin', {}).get('raw_camera_timestamp_ns_at_t0')
            if first is not None and t0 is not None:
                ep.video_offset_s = (int(t0) - first[0]) / 1e9
                ep.video_frames = first[1]
    return ep


@dataclass(frozen=True)
class Source:
    """One policy config's episodes: its twin IK replays (``replay_root``, the labels stage output), camera
    trajectories (``vio_root``, the reconstruct stage output) and recording data roots. ``split_file``: its cohort
    selection.json (LabelStore split). ``suffix``: appended to the episode ids ('' for the first source)."""
    name: str
    replay_root: Path
    vio_root: Path
    data_roots: tuple = ()
    split_file: Path | None = None
    method: str = ''
    suffix: str = ''


def config_source(spec, primary: bool = True) -> Source:
    """A configs/hub/apps.json ``sources`` entry -> Source, from the policy run config it names: ``"<config>"`` or
    ``{"config": "<config>"}`` (a name in configs/policy/ or a path; policy.configs.schema.load). Every path comes
    from the config's stage outputs (Config.dataset): replays = labels, trajectories = reconstructions, the split =
    ``<cohorts>/selection.json``; the data roots are its ``source.recordings`` roots."""
    from policy.configs import schema
    name = spec if isinstance(spec, str) else (spec or {}).get('config')
    if not isinstance(name, str) or not name:
        raise ValueError(f'twin viewer source {spec!r}: expected "<config>" or {{"config": "<config>"}}')
    if isinstance(spec, dict) and set(spec) - {'config'}:
        raise ValueError(f'twin viewer source {name!r}: unknown keys {sorted(set(spec) - {"config"})}')
    cfg = schema.load(name)
    if cfg.kind != 'cloak':
        raise ValueError(f'twin viewer source {name!r}: a {cfg.kind} config has no recordings')
    roots = []
    for rec in cfg.section('source').get('recordings') or []:
        r = _repo_path(rec['data_root'])
        if r not in roots:
            roots.append(r)
    return Source(cfg.name, _repo_path(cfg.dataset('labels')), _repo_path(cfg.dataset('reconstructions')),
                  tuple(roots), _repo_path(cfg.dataset('cohorts')) / 'selection.json', cfg.method,
                  '' if primary else f'.{cfg.name}')


def config_sources(specs) -> list[Source]:
    """config_source for each entry; the first is primary (no id suffix). Names must be unique."""
    out = [config_source(x, i == 0) for i, x in enumerate(specs or [])]
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f'twin viewer sources: duplicate configs {names}')
    return out


def _repo_path(p) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO / p


class EpisodeIndex:
    """Episode table rebuilt on demand. A cheap stat signature of every source's roots, every replay /
    VIO directory and the data roots decides whether to rescan; unchanged episodes are reused."""

    def __init__(self, sources):
        self.sources = list(sources)
        self._lock = threading.Lock()
        self._sig, self._eps = None, {}

    def _signature(self):
        sig = []
        for s in self.sources:
            sig += [s.name, mtime_ns(s.replay_root), mtime_ns(s.vio_root), *(mtime_ns(r) for r in s.data_roots)]
            if s.replay_root.is_dir():
                for d in sorted(s.replay_root.iterdir()):
                    sig.append((d.name, mtime_ns(d), mtime_ns(d / 'metrics.json'), mtime_ns(s.vio_root / d.name)))
        return tuple(sig)

    def get(self) -> dict[str, Episode]:
        with self._lock:
            sig = self._signature()
            if sig != self._sig:
                eps = {}
                for s in reversed(self.sources):          # primary last, so it wins on duplicate ids
                    eps.update(load_episodes(s.replay_root, s.vio_root, s.data_roots, cache=self._eps,
                                             source=s.name, suffix=s.suffix))
                self._eps = dict(sorted(eps.items()))
                self._sig = sig
            return self._eps

    def split_files(self) -> list[Path]:
        return [s.split_file for s in self.sources if s.split_file is not None]

    def split_exempt(self) -> set[str]:
        """Ids from sources without a cohort selection yet: the labels panel accepts them without the split check."""
        free = {s.name for s in self.sources if s.split_file is None or not s.split_file.is_file()}
        return {i for i, e in self.get().items() if e.source in free}

    def info(self) -> dict:
        """/api/index extras: the source picker when there is more than one source."""
        return {'sources': [{'id': s.name, 'name': f'{s.name} ({s.method})' if s.method else s.name}
                            for s in self.sources]} if len(self.sources) > 1 else {}


# ------------------------------------------------------------------ rendering
class SimRenderer:
    """Offscreen MuJoCo renders on one dedicated thread (GL contexts are thread-bound);
    one cached model + Renderer, additionally guarded by a lock."""

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mj-render')
        self._lock = threading.Lock()
        self._states = {}                  # profile -> (model, data, qadr, Renderer)

    def _init(self, profile):
        import mujoco
        from digital_twin.sim.kinematics import load_model
        m = load_model(profile)
        d = mujoco.MjData(m)
        qadr = m.jnt_qposadr[[m.joint(n).id for n in JOINTS]]
        self._states[profile] = (m, d, qadr, mujoco.Renderer(m, height=FRAME_H, width=FRAME_W))

    def render(self, ep: Episode, t: float, camera: str, mode: str = 'twin', dz: float = 0.0) -> bytes:
        return self._pool.submit(self._render, ep, t, camera, mode, dz).result(timeout=30)

    def _render(self, ep, t, camera, mode, dz=0.0):
        import cv2
        import mujoco
        from digital_twin.sim.viz import add_segment, hud, make_camera
        with self._lock:
            profile = ep.profile
            if profile not in self._states:
                self._init(profile)
            m, d, qadr, renderer = self._states[profile]
            fk = ep.fk(mode, dz)
            tt, tg, reach = ep.targets(dz)
            r = float(ep.replay_time(t))
            k = int(np.clip(np.searchsorted(fk['t'], r, 'right') - 1, 0, len(fk['t']) - 1))
            q = lock_slew([np.interp(r, fk['t'], fk['q'][:, i]) for i in range(fk['q'].shape[1])])
            ka = int(np.clip(np.searchsorted(tt, r, 'right') - 1, 0, len(tt) - 1))
            d.qpos[:] = m.qpos0; d.qvel[:] = 0
            d.qpos[qadr] = q
            if m.nmocap:
                d.mocap_pos[0] = tg[ka]
            mujoco.mj_forward(m, d)
            lookat = np.r_[0.5 * (tg.min(0) + tg.max(0))[:2] * 0.55, 0.15]   # as render_video
            renderer.update_scene(d, camera=make_camera(m, camera, lookat))
            sc = renderer.scene
            for j in range(len(tg) - 1):
                add_segment(sc, tg[j], tg[j + 1],
                            (.2, 1, .45, .35 if j >= ka else .9) if reach[j] else (1, .25, .25, .9), 0.0012)
            tip = fk['tip']
            for j in range(max(0, k - 60), k):
                add_segment(sc, tip[j], tip[j + 1], (1, .6, .1, 1), 0.0015)
            frame = np.ascontiguousarray(renderer.render())
        vi = int(np.clip(np.searchsorted(ep.frame_t, t, 'right') - 1, 0, len(ep.status) - 1))
        frame = hud(frame, [f'twin ({profile})  camera: {camera}' + (f'  height {dz:+.3f} m' if round(dz * 1000) else ''),
                            f't = {t:5.2f} s  replay {r:5.2f} s  VIO: {ep.status[vi]}',
                            f'tip error: {fk["tip_err"][k] * 1000:5.1f} mm'])
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG encode failed')
        return buf.tobytes()

    def close(self):
        def _close():
            for st in self._states.values():
                st[3].close()
            self._states.clear()
        try:
            self._pool.submit(_close).result(timeout=5)
        except Exception:
            pass
        self._pool.shutdown(wait=False)


# ------------------------------------------------------------------ success labels (annotation)
LABELS_SCHEMA = 'soil_success_labels.v2'
VERDICTS = {'good': True, 'bad': False, 'skip': None}
DEFAULT_PROMPT = TASKS['dirt']                 # a new labels file's default task prompt
PROMPT_GROUPS = ('dig', 'ball')                # groups the panel shows prompt fields for
MAX_PROMPT = 512
MAX_GROUPS = 32
LABELS_BODY_MAX = 64 << 10
_CONTROL_RE = re.compile(r'[\x00-\x1f\x7f-\x9f  ]')


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def empty_labels() -> dict:
    return {'schema': LABELS_SCHEMA, 'default_prompt': DEFAULT_PROMPT,
            'group_prompts': {g: '' for g in PROMPT_GROUPS}, 'annotated_at': None, 'labels': {}, 'pending': {}}


def check_prompt(v, key: str = 'prompt') -> str:
    """A task prompt: string, <= MAX_PROMPT chars, no control characters (newlines included)."""
    if v is None:
        v = ''
    if not isinstance(v, str):
        raise ValueError(f'"{key}" must be a string')
    if _CONTROL_RE.search(v):
        raise ValueError(f'"{key}" must not contain control characters or line breaks')
    v = ' '.join(v.split())
    if len(v) > MAX_PROMPT:
        raise ValueError(f'"{key}" is longer than {MAX_PROMPT} characters')
    return v


def _entry(verdict, prompt, updated_at) -> dict:
    e = {'success': VERDICTS[verdict], 'verdict': verdict} if verdict in ('good', 'bad') else {'verdict': verdict}
    if prompt:
        e['prompt'] = prompt
    e['updated_at'] = updated_at
    return e


def migrate_labels(raw) -> dict:
    """Any labels file (v1 or v2) -> v2. Keeps success/verdict/prompt/updated_at per episode and the
    default/group prompts; drops v1's reviewer, criterion, notes, flags, evidence and t_s."""
    data = empty_labels()
    if not isinstance(raw, dict):
        return data
    if isinstance(raw.get('default_prompt'), str):
        data['default_prompt'] = raw['default_prompt']
    if isinstance(raw.get('group_prompts'), dict):
        data['group_prompts'].update({g: p for g, p in raw['group_prompts'].items() if isinstance(p, str)})
    data['annotated_at'] = raw.get('annotated_at') or raw.get('reviewed_at')
    if raw.get('schema') == LABELS_SCHEMA:                  # keep v2 extras, e.g. the study's `imports`
        data.update({k: v for k, v in raw.items() if k not in data})
    for where in ('pending', 'labels'):                     # labels win if an id is in both
        for ep, v in (raw.get(where) or {}).items():
            if not isinstance(v, dict):
                continue
            prompt = v.get('prompt') if isinstance(v.get('prompt'), str) and v['prompt'].strip() else None
            if isinstance(v.get('success'), bool):
                verdict = 'good' if v['success'] else 'bad'
            elif v.get('verdict') == 'skip':
                verdict = 'skip'
            elif prompt:
                verdict = None                              # prompt override on an unlabelled episode
            else:
                continue
            data['labels'].pop(ep, None); data['pending'].pop(ep, None)
            entry = _entry(verdict, prompt, v.get('updated_at'))
            if isinstance(v.get('reason'), str) and v['reason'].strip():   # import exclusion reason
                entry['reason'] = v['reason']
            (data['labels'] if verdict in ('good', 'bad') else data['pending'])[ep] = entry
    return data


def resolve_prompt(data: dict, episode: str, group: str | None) -> tuple[str, str | None]:
    """(prompt, source) with episode prompt > group prompt > default_prompt; source None when empty."""
    entry = (data.get('labels') or {}).get(episode) or (data.get('pending') or {}).get(episode) or {}
    candidates = (('episode', entry.get('prompt') if isinstance(entry, dict) else None),
                  ('group', (data.get('group_prompts') or {}).get(group or 'dig')),
                  ('default', data.get('default_prompt')))
    for source, p in candidates:
        if isinstance(p, str) and p.strip():
            return p.strip(), source
    return '', None


class LabelStore:
    """Per-episode success flag + task prompt (``soil_success_labels.v2``).

    Good/Bad go under ``labels`` as {success, verdict, prompt?, updated_at}; Skip goes under
    ``pending`` as {verdict: "skip", prompt?, updated_at}. A prompt override on an episode without
    a verdict is kept under ``pending`` with ``verdict: null``. A v1 file is migrated on load and
    rewritten as v2 on the next save. Every write goes through a temp file + ``os.replace``.

    ``split_file``: one or more split files, merged: a cohort selection.json (policy.preprocessing.select:
    ``episodes`` with their ``split``, plus ``excluded_by_config``) or a plain ``{train, holdout}`` file.
    """

    def __init__(self, path: Path, split_file=None):
        self.path = Path(path)
        self.split_files = [Path(f) for f in ([] if split_file is None else
                                              [split_file] if isinstance(split_file, (str, Path)) else split_file)]
        self._lock = threading.Lock()

    # -- file io
    def load(self) -> dict:
        if not self.path.is_file():
            return empty_labels()
        return migrate_labels(json.loads(self.path.read_text()))

    def _save(self, data: dict):
        data['schema'] = LABELS_SCHEMA
        data['annotated_at'] = _now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f'.{self.path.name}.{os.getpid()}.{threading.get_ident()}.partial')
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def split(self) -> dict | None:
        """{train, holdout, excluded} merged over the readable split files; None when there is none."""
        out = None
        for f in self.split_files:
            try:
                s = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            out = out or {'train': [], 'holdout': [], 'excluded': []}
            if isinstance(s.get('episodes'), list):                  # cohort selection.json
                for e in s['episodes']:
                    if e.get('split') in ('train', 'holdout'):
                        out[e['split']].append(e['id'])
                out['excluded'] += list(s.get('excluded_by_config') or [])
            else:
                out['train'] += list(s.get('train') or [])
                out['holdout'] += list(s.get('holdout') or [])
        return out

    @staticmethod
    def _groups(episodes) -> dict[str, str]:
        """{id: group} from a {id: group} dict or a plain id iterable (-> 'dig')."""
        return dict(episodes) if isinstance(episodes, dict) else {i: 'dig' for i in episodes}

    def resolved_prompts(self, episodes, data: dict | None = None) -> dict[str, dict]:
        data = self.load() if data is None else data
        out = {}
        for i, g in self._groups(episodes).items():
            p, src = resolve_prompt(data, i, g)
            out[i] = {'prompt': p, 'source': src, 'group': g}
        return out

    # -- mutations
    def set_meta(self, body: dict) -> dict:
        """Body: {default_prompt?, group_prompts?: {group: prompt}} (group prompts merge per key)."""
        if not isinstance(body, dict):
            raise ValueError('body must be a JSON object')
        unknown = set(body) - {'default_prompt', 'group_prompts'}
        if unknown:
            raise ValueError(f'unknown fields: {sorted(unknown)}')
        if not body:
            raise ValueError('send "default_prompt" and/or "group_prompts"')
        default_prompt = check_prompt(body['default_prompt'], 'default_prompt') if 'default_prompt' in body else None
        groups = body.get('group_prompts')
        if 'group_prompts' in body:
            if not isinstance(groups, dict):
                raise ValueError('"group_prompts" must be an object {group: prompt}')
            if len(groups) > MAX_GROUPS or not all(isinstance(g, str) and EPISODE_RE.match(g) for g in groups):
                raise ValueError('"group_prompts" keys must be group names like "dig" or "ball"')
            groups = {g: check_prompt(v, f'group_prompts.{g}') for g, v in groups.items()}
        with self._lock:
            data = self.load()
            if default_prompt is not None:
                data['default_prompt'] = default_prompt
            if groups is not None:
                data['group_prompts'].update(groups)
                if len(data['group_prompts']) > MAX_GROUPS:
                    raise ValueError(f'at most {MAX_GROUPS} group prompts')
            self._save(data)
            return data

    def upsert(self, episode: str, body: dict, known_ids, split_exempt=()) -> dict | None:
        """Set one episode's verdict and/or prompt override.

        Body: {verdict?: good|bad|skip|null, prompt?: str}; at least one key. ``prompt`` "" clears
        the override; an omitted key keeps its current value. Returns the entry (None if removed).
        """
        if not isinstance(body, dict):
            raise ValueError('body must be a JSON object')
        if episode not in set(known_ids):
            raise LookupError('unknown episode')
        split = self.split()
        if (split is not None and (split['train'] or split['holdout'])
                and episode not in split['train'] + split['holdout'] + split['excluded']
                and episode not in set(split_exempt)):
            raise ValueError('episode is not in its config\'s cohort selection')
        unknown = set(body) - {'verdict', 'prompt'}
        if unknown:
            raise ValueError(f'unknown fields: {sorted(unknown)}')
        if not body:
            raise ValueError('send "verdict" (good, bad, skip or null) and/or "prompt"')
        if 'verdict' in body and body['verdict'] is not None and body['verdict'] not in VERDICTS:
            raise ValueError('"verdict" must be "good", "bad", "skip" or null')
        prompt = check_prompt(body['prompt']) if 'prompt' in body else None
        with self._lock:
            data = self.load()
            old = data['labels'].pop(episode, None) or data['pending'].pop(episode, None) or {}
            verdict = body['verdict'] if 'verdict' in body else old.get('verdict')
            keep = old.get('prompt') if prompt is None else prompt
            entry = None
            if verdict is not None or keep:
                entry = _entry(verdict, keep, _now_iso())
                if old.get('reason') and verdict == old.get('verdict'):
                    entry['reason'] = old['reason']
                (data['labels'] if verdict in ('good', 'bad') else data['pending'])[episode] = entry
            self._save(data)
            return entry

    def payload(self, episodes) -> dict:
        """GET /api/labels: the file plus ``resolved_prompts`` (computed, not stored)."""
        data = self.load()
        return {**data, 'resolved_prompts': self.resolved_prompts(episodes, data)}

    # -- summary
    def summary(self, episodes) -> dict:
        """``episodes``: {id: group} (or ids, all treated as group 'dig')."""
        data = self.load()
        groups = self._groups(episodes)
        ids = list(groups)
        resolved = self.resolved_prompts(groups, data)
        verdict = {**{k: v.get('verdict') for k, v in data['pending'].items()},
                   **{k: v['verdict'] for k, v in data['labels'].items()}}
        counts = {'good': 0, 'bad': 0, 'skip': 0, 'unlabeled': 0}
        for i in ids:
            counts[verdict.get(i) if verdict.get(i) in ('good', 'bad', 'skip') else 'unlabeled'] += 1
        out = {'total': len(ids), 'labeled': counts['good'] + counts['bad'], **counts,
               'verdicts': {i: verdict.get(i) for i in ids}, 'labels_file': str(self.path), 'split': None,
               'default_prompt': data['default_prompt'], 'group_prompts': data['group_prompts'],
               'resolved_prompts': {i: r['prompt'] for i, r in resolved.items()},
               'prompts': {'resolved': sum(bool(r['prompt']) for r in resolved.values()),
                           'missing': [i for i, r in resolved.items() if not r['prompt']]}}
        split = self.split()
        if split is not None:
            for name in ('train', 'holdout'):
                eps = split[name]
                out[name] = {'total': len(eps), 'labeled': sum(verdict.get(i) in ('good', 'bad') for i in eps),
                             'good': sum(verdict.get(i) == 'good' for i in eps),
                             'unlabeled': [i for i in eps if verdict.get(i) not in ('good', 'bad')]}
            out['split'] = [str(f) for f in self.split_files]
        out['train_ready'] = bool(split is not None and split['train'] and not out['train']['unlabeled'])
        out['all_ready'] = bool(out['train_ready'] and not out['holdout']['unlabeled'])
        out['prompts_ready'] = bool(split is not None and split['train']
                                    and all(resolve_prompt(data, i, groups.get(i, 'dig'))[0] for i in split['train']))
        return out
