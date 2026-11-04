"""Geometry-backed masks and bucket contact-pose retargeting.

``render_masks`` renders per-frame embodiment masks for a frozen export from a measured
geometry config. The handheld source needs its own mesh/mount calibration; the excavator's
unmeasured default mount is never accepted as a calibrated handheld source.

``BucketRetargeter`` (``serve --source-profile``) maps [boom, bucket] states and action chunks
between two twin profiles through the cutting-tip position and bucket frame, with slew and stick
locked. Those two joints cannot realize arbitrary 6D motion, so it reports residuals and rejects
chunks containing an unreachable pose.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy.data.cloak_dataset import JOINT_ORDERS, digest, load_arrays, unique_entries, validate_episode

from .masking import letterbox


def checked_config(path, embodiment=None):
    cfg = json.loads(Path(path).read_text())
    if cfg.get('measured') is not True or not cfg.get('source_embodiment'):
        raise ValueError('A measured mount and explicit source_embodiment are required')
    if embodiment is not None and cfg['source_embodiment'] != embodiment:
        raise ValueError('Geometry configuration does not match the recording embodiment')
    if not cfg.get('calibration_evidence'):
        raise ValueError('Provide calibration_evidence for the actual recording rig')
    return cfg


def render_masks(source, config, output, *, embodiment, image_geometry):
    """Render masks for an existing real export, preserving its frame IDs.

    config must additionally declare state_joints and state_contract matching
    the frozen export. For pre-letterboxed images, render at native camera size
    before applying the same pad transform. Never render a square-camera mask
    directly onto an already letterboxed wide-camera image.
    """
    from digital_twin.sim.self_mask import SelfMasker
    source, output = Path(source), Path(output)
    cfg = checked_config(config, embodiment)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['schema'] != 'actathon.pi05.real.v1':
        raise ValueError('Expected a frozen real-recording export')
    entries = unique_entries(manifest['episodes'])
    if image_geometry not in ('native', 'letterbox224'):
        raise ValueError('Explicit native or letterbox224 image geometry is required')
    if cfg.get('state_joints') not in JOINT_ORDERS:
        raise ValueError('Set state_joints to the model joints matching the source state')
    for e in entries.values():
        if cfg.get('state_contract') != e['action_contract'] or cfg['state_joints'] != e['action_contract']['joints']:
            raise ValueError('Mask-renderer joint coordinates must match the frozen source contract')
    output.mkdir(parents=True, exist_ok=False)
    renderer, records = SelfMasker(config=cfg), []
    try:
        for i, entry in enumerate(entries.values()):
            a = load_arrays(source, entry)
            validate_episode(entry, a, masks=False)
            h, w = a['images'].shape[1:3]
            nw, nh = cfg['intrinsics']['image_size']
            if image_geometry == 'letterbox224' and (h, w) != (224, 224):
                raise ValueError('letterbox224 source images must be 224x224')
            if image_geometry == 'native' and (w, h) != (nw, nh):
                raise ValueError('Native image size must match calibrated camera size')
            masks = []
            for q in a['state']:
                mask = renderer.mask(dict(zip(cfg['state_joints'], q)), (nh, nw))
                if image_geometry == 'letterbox224':
                    _, mask = letterbox(np.zeros((nh, nw, 3), np.uint8), mask)
                masks.append(mask)
            dest = output / f'masks-{i:05d}.npz'
            np.savez_compressed(dest, masks=np.stack(masks), source_frame=a['source_frame'], time_s=a['time_s'])
            records.append({'id': entry['id'], 'path': dest.name, 'sha256': digest(dest)})
    finally:
        renderer.close()
    result = {'schema': 'actathon.cloak.masks.v1', 'source_embodiment': embodiment,
              'source_manifest_sha256': digest(source / 'manifest.json'),
              'reviewed': False, 'episodes': records,
              'provenance': {'config': cfg, 'config_sha256': digest(config), 'image_geometry': image_geometry,
                             'method': 'MuJoCo FK segmentation and calibrated lens projection'}}
    # Alignment review remains explicit. Rendering alone does not certify a mask.
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


class BucketRetargeter:
    """Source FK -> target IK through the cutting-tip position and bucket frame.

    This is the rigid-bucket counterpart of CLOAK's two moving gripper tips.
    Profiles must describe the same world/base frame and metric scale. The
    target's measured joints must be used to compute the policy's source state.
    This legacy adapter uses [boom, bucket] states and actions, with slew and
    stick fixed. Four-joint policies use their declared full4 joint coordinates.
    """
    def __init__(self, source='full4', target='full4_calibrated', pos_tol=0.003, angle_tol_deg=2.0):
        from digital_twin.sim.kinematics import Arm
        if pos_tol <= 0 or angle_tol_deg <= 0:
            raise ValueError('Retargeting tolerances must be positive')
        self.source, self.target = Arm(source), Arm(target)
        self.pos_tol, self.angle_tol = pos_tol, angle_tol_deg

    def _map(self, q, seed, source, target):
        from digital_twin.sim.kinematics import full_chain_q, hardware_q, ik, locked_joints
        q, seed = np.asarray(q, float), np.asarray(seed, float)
        if q.shape != (2,) or seed.shape != (2,) or not np.isfinite([q, seed]).all():
            raise ValueError('Expected finite two-joint state and IK seed')
        qf = full_chain_q(q)
        if np.any(qf < source.lo) or np.any(qf > source.hi):
            raise ValueError('Source state exceeds its joint limits')
        position = source.fk(qf)
        result = ik(position, rotmat=source.bucket_rot(), q0=full_chain_q(seed), arm=target,
                    pos_tol=self.pos_tol, pitch_tol_deg=self.angle_tol, locked=locked_joints())
        if not result.reachable or result.orientation_error_deg > self.angle_tol:
            raise ValueError(f'Unreachable bucket pose: {result.as_dict()}')
        return hardware_q(result.q).copy(), result.as_dict()

    def target_state(self, q, source_seed):
        return self._map(q, source_seed, self.target, self.source)

    def source_actions(self, chunk, target_seed):
        chunk = np.asarray(chunk, float)
        if chunk.ndim != 2 or chunk.shape[1] != 2 or not len(chunk):
            raise ValueError('Expected a nonempty [horizon,2] action chunk')
        output, residuals = [], []
        for q in chunk:
            target_seed, report = self._map(q, target_seed, self.source, self.target)
            output.append(target_seed)
            residuals.append(report)
        return np.asarray(output, np.float32), residuals
