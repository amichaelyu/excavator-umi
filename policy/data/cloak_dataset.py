"""Frozen, episode-disjoint excavator data with frame-aligned geometric masks.

Preparation consumes the existing actathon.pi05.real.v1 export plus a mask
manifest. Mask NPZ files contain masks [N,H,W], source_frame [N], time_s [N].
The mask manifest schema is actathon.cloak.masks.v1, with source_manifest_sha256,
source_embodiment, reviewed=true, provenance, and episodes [{id,path,sha256}].
Review means the geometry and camera match the recording rig, not task success.
Reconstructed handheld joint targets retain their original provenance.

This is the format only (numpy, no cv2 / JAX): preprocessing writes it, training and evaluation read it.
The model's view of a frame (letterbox, CLOAK masks, π0.5 inputs) is policy.training.cloak.dataset.frame_inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

JOINT_ORDERS = (['boom', 'bucket'], ['slew', 'boom', 'stick', 'bucket'])


def contract_joints(contract):
    joints = contract.get('joints')
    if (joints not in JOINT_ORDERS or contract.get('units') != 'radians'
            or contract.get('type') != 'absolute next joint target'):
        raise ValueError('Expected absolute boom/bucket or slew/boom/stick/bucket targets in radians')
    return joints


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def local_path(root, path):
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Dataset paths must stay inside the dataset root')
    return result


def load_arrays(root, entry):
    path = local_path(root, entry['path'])
    if digest(path) != entry['sha256']:
        raise ValueError(f"Hash mismatch: {entry['id']}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def validate_episode(entry, arrays, *, masks=True):
    n = entry['frames']
    if not isinstance(n, int) or n < 1:
        raise ValueError('An episode must contain frames')
    if entry['split'] not in ('train', 'holdout') or not entry['prompt'].strip():
        raise ValueError('Expected frozen train/holdout membership and a task prompt')
    contract = entry['action_contract']
    joints = contract_joints(contract)
    for key in ('state', 'actions'):
        if arrays[key].shape != (n, len(joints)) or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid {key} shape or nonfinite values')
    images = arrays['images']
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[0] != n or images.shape[-1] != 3:
        raise ValueError('Expected N,H,W,3 uint8 RGB images')
    retimed = contract.get('time_base') == 'retimed_robot_execution'
    for key in ('source_frame', 'time_s', *(['source_time_s'] if retimed else [])):
        value = arrays[key]
        repeated = retimed and key != 'time_s'
        bad_order = np.any(np.diff(value) < 0 if repeated else np.diff(value) <= 0)
        if value.shape != (n,) or not np.isfinite(value).all() or bad_order:
            raise ValueError(f'{key} must be {"nondecreasing" if repeated else "strictly increasing"} and finite')
    if retimed:
        hz = contract.get('sampling_hz', 0)
        if hz <= 0 or not np.allclose(np.diff(arrays['time_s']), 1 / hz, atol=1e-6):
            raise ValueError('Retimed examples must follow the declared robot control clock')
    if arrays['source_frame'].dtype.kind not in 'iu' or np.any(arrays['source_frame'] < 0):
        raise ValueError('source_frame must contain nonnegative integers')
    if masks and (arrays['masks'].dtype != np.bool_ or arrays['masks'].shape != images.shape[:3]):
        raise ValueError('Boolean masks must align with every image')


def unique_entries(entries):
    result = {e['id']: e for e in entries}
    if len(result) != len(entries) or not result:
        raise ValueError('Episode IDs must be unique and nonempty')
    return result


def checked_content_box(box, shape=(224, 224)):
    if box is None or len(box) != 4 or any(not isinstance(x, (int, np.integer)) for x in box):
        raise ValueError('content_box must contain four integer pixel coordinates')
    x0, y0, x1, y1 = box
    h, w = shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('content_box must lie inside the image')
    return x0, y0, x1, y1


def prepare(source, masks, output):
    source, masks, output = map(Path, (source, masks, output))
    manifest = json.loads((source / 'manifest.json').read_text())
    mask_manifest = json.loads((masks / 'manifest.json').read_text())
    if manifest['schema'] != 'actathon.pi05.real.v1' or mask_manifest['schema'] != 'actathon.cloak.masks.v1':
        raise ValueError('Unexpected source or mask manifest schema')
    if mask_manifest['source_manifest_sha256'] != digest(source / 'manifest.json'):
        raise ValueError('Masks were generated for a different frozen source manifest')
    if mask_manifest.get('reviewed') is not True or not mask_manifest.get('provenance') or not mask_manifest.get('source_embodiment'):
        raise ValueError('Masks need reviewed geometry/camera provenance for the source embodiment')
    entries = unique_entries(manifest['episodes'])
    mask_entries = unique_entries(mask_manifest['episodes'])
    if entries.keys() != mask_entries.keys():
        raise ValueError('Mask manifest must cover exactly the frozen source episodes')
    check_splits(manifest, entries)
    if len({json.dumps(e['action_contract'], sort_keys=True) for e in entries.values()}) != 1:
        raise ValueError('Do not mix action coordinate systems in one source policy')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, (eid, entry) in enumerate(entries.items()):
        a = load_arrays(source, entry)
        validate_episode(entry, a, masks=False)
        m = load_arrays(masks, mask_entries[eid])
        for key in ('source_frame', 'time_s'):
            if not np.array_equal(a[key], m[key]):
                raise ValueError(f'{eid}: mask {key} does not align with source frames')
        a['masks'] = m['masks']
        validate_episode(entry, a)
        dest = output / f'episode-{i:05d}.npz'
        np.savez_compressed(dest, **a)
        records.append({**entry, 'path': dest.name, 'sha256': digest(dest), 'source_sha256': entry['sha256']})
    result = {**manifest, 'schema': 'actathon.cloak.v1', 'episodes': records,
              'source_manifest_sha256': digest(source / 'manifest.json'),
              'mask_manifest_sha256': digest(masks / 'manifest.json'),
              'source_embodiment': mask_manifest['source_embodiment'],
              'mask_provenance': mask_manifest['provenance'], 'masks_reviewed': True}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def no_holdout(manifest) -> bool:
    """``holdout: none``: every recording trains; there is no held-out evaluation or checkpoint selection."""
    return manifest.get('holdout') == 'none'


def check_splits(manifest, entries):
    want = {'train'} if no_holdout(manifest) else {'train', 'holdout'}
    if {e['split'] for e in entries.values()} != want:
        raise ValueError('Every episode must be train (holdout: none)' if no_holdout(manifest)
                         else 'Both train and holdout episodes are required')


def prepare_out_of_view(source, output, *, embodiment, evidence, content_box):
    """Freeze an explicitly confirmed invisible source tool without inventing geometry."""
    source, output = Path(source), Path(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['schema'] != 'actathon.pi05.real.v1':
        raise ValueError('Expected a frozen real-recording export')
    if not embodiment or not isinstance(evidence, dict) or not evidence.get('confirmation'):
        raise ValueError('An explicit source embodiment and visibility confirmation are required')
    if evidence.get('source_manifest_sha256') != digest(source / 'manifest.json'):
        raise ValueError('Visibility evidence must identify this frozen source manifest')
    x0, y0, x1, y1 = checked_content_box(content_box)
    entries = unique_entries(manifest['episodes'])
    check_splits(manifest, entries)
    if len({json.dumps(e['action_contract'], sort_keys=True) for e in entries.values()}) != 1:
        raise ValueError('Do not mix action coordinate systems')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    outside = np.ones((224, 224), bool)
    outside[y0:y1, x0:x1] = False
    for i, entry in enumerate(entries.values()):
        a = load_arrays(source, entry)
        validate_episode(entry, a, masks=False)
        if a['images'].shape[1:3] != (224, 224) or a['images'][:, outside].any():
            raise ValueError('Expected letterbox224 RGB with zero padding outside content_box')
        a['masks'] = np.zeros(a['images'].shape[:3], bool)
        dest = output / f'episode-{i:05d}.npz'
        np.savez_compressed(dest, **a)
        records.append({**entry, 'path': dest.name, 'sha256': digest(dest), 'source_sha256': entry['sha256']})
    result = {**manifest, 'schema': 'actathon.cloak.v1', 'episodes': records,
              'source_manifest_sha256': digest(source / 'manifest.json'),
              'source_embodiment': embodiment, 'source_visibility': 'out_of_view',
              'content_box': list(content_box), 'masks_reviewed': True,
              'mask_provenance': {'method': 'Confirmed source tool outside field of view', 'evidence': evidence},
              'masking_adaptation': 'edge_capsules_v1; empty actual source masks; no source roll-fill'}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


class CloakDataset:
    """One split of a prepared ``actathon.cloak.v1`` dataset: every episode validated and loaded, ``index`` =
    (episode, frame) pairs. ``policy.training.cloak.dataset.frame_inputs(dataset, i)`` turns a frame into model inputs."""

    def __init__(self, root, split='train', horizon=10):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('schema') != 'actathon.cloak.v1' or self.manifest.get('masks_reviewed') is not True:
            raise ValueError('Expected a prepared CLOAK dataset with reviewed masks')
        if horizon < 1 or split not in ('train', 'holdout'):
            raise ValueError('Invalid horizon or split')
        self.horizon, self.episodes, self.index = horizon, [], []
        all_entries = unique_entries(self.manifest['episodes'])
        if len({json.dumps(e['action_contract'], sort_keys=True) for e in all_entries.values()}) != 1:
            raise ValueError('Mixed action contracts')
        self.joints = contract_joints(next(iter(all_entries.values()))['action_contract'])
        for e in all_entries.values():
            if e['split'] != split:
                continue
            a = load_arrays(self.root, e)
            validate_episode(e, a)
            if self.manifest.get('source_visibility') == 'out_of_view':
                checked_content_box(self.manifest.get('content_box'))
                if a['masks'].any():
                    raise ValueError('An out-of-view source must have empty actual tool masks')
            self.index.extend((len(self.episodes), t) for t in range(e['frames']))
            self.episodes.append((e, a))
        if not self.index and not (split == 'holdout' and no_holdout(self.manifest)):
            raise ValueError(f'No {split} episodes')

    def __len__(self):
        return len(self.index)

    def stats(self, action_representation='absolute'):
        """Only this split's arrays; callers fit on the train dataset."""
        if action_representation not in ('absolute', 'delta'):
            raise ValueError('Unknown action representation')
        result = {k: np.concatenate([a[k] for _, a in self.episodes]) for k in ('state', 'actions')}
        if action_representation == 'delta':
            chunks = []
            for _, a in self.episodes:
                n = len(a['actions'])
                indices = np.minimum(np.arange(n)[:, None] + np.arange(self.horizon), n - 1)
                chunks.append((a['actions'][indices] - a['state'][:, None, :]).reshape(-1, len(self.joints)))
            # Every target in a chunk is relative to its observation's state.
            # One-step differences alone underestimate the horizon's range.
            result['actions'] = np.concatenate(chunks)
        return result
