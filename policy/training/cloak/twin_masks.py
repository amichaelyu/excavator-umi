"""Embodiment masks rendered from the excavator twin at each sample's labelled joint state.

At deployment the runner blanks the excavator out of the bucket camera with
``digital_twin.sim.self_mask.SelfMasker.mask(q, frame.shape)`` (pi05_tabletop_excavator) and the policy server
letterboxes frame and mask together (``masking.letterbox``). This renders the same mask for every
exported sample, at the sample's labelled joint state ``state`` = [slew, boom, stick, bucket], so
CLOAK trains on the tokens the robot drops at deployment instead of random edge capsules
(``cloak_oov``). The handheld recordings do not show the excavator: the mask marks where it *would*
be, which is what the policy has to tolerate missing.

    python -m policy.training.cloak.twin_masks --data <cohort>/export/source --output <masks dir> [--workers 8]
    python -m policy.training.cloak prepare --data <cohort>/export/source --masks <masks dir> --output <prepared>

The output is a ``actathon.cloak.masks.v1`` directory that ``dataset.prepare`` joins to the export.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from policy.data.camera import FRAME_SHAPE as NATIVE   # bucket camera frame (h, w); exports letterbox it to 224
from policy.data.cloak_dataset import digest, load_arrays

from .masking import letterbox_mask


_MASKER = None


def _episode(args):
    data, entry, output, config_path, joints = args
    global _MASKER
    if _MASKER is None:
        from digital_twin.sim.self_mask import SelfMasker
        _MASKER = SelfMasker(config_path=config_path)
    a = load_arrays(data, entry)
    masks = np.stack([letterbox_mask(_MASKER.mask(dict(zip(joints, map(float, q))), NATIVE)) for q in a['state']])
    path = output / f"{entry['id']}.npz"
    np.savez_compressed(path, masks=masks, source_frame=a['source_frame'], time_s=a['time_s'])
    return {'id': entry['id'], 'path': path.name, 'sha256': digest(path), 'frames': int(len(masks)),
            'mask_fraction': float(masks.mean())}


def render(data, output, config_path=None, workers=8):
    from digital_twin.sim.self_mask import SelfMasker
    data, output = Path(data), Path(output)
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest.get('schema') != 'actathon.pi05.real.v1':
        raise ValueError('Expected a frozen real-recording export (policy.preprocessing.export .../source)')
    joints = manifest['episodes'][0]['action_contract']['joints']
    if joints != ['slew', 'boom', 'stick', 'bucket']:
        raise ValueError('Four-joint labels are required')
    masker = SelfMasker(config_path=config_path)
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(data, e, output, config_path, joints) for e in manifest['episodes']]
    with ProcessPoolExecutor(workers) as ex:
        episodes = list(ex.map(_episode, jobs))
    frac = [e['mask_fraction'] for e in episodes]
    result = {'schema': 'actathon.cloak.masks.v1', 'source_manifest_sha256': digest(data / 'manifest.json'),
              'reviewed': True, 'source_embodiment': 'excavator-bucket-camera (twin self-mask)',
              'provenance': {'method': 'digital_twin.sim.self_mask.SelfMasker.mask at the labelled joint state, '
                                       'letterboxed like the images (masking.letterbox geometry)',
                             'config': str(masker.config_path), 'profile': masker.cfg['profile'],
                             'mount': masker.mount, 'mount_measured': bool(masker.cfg.get('measured', False)),
                             'native_frame': list(NATIVE), 'dilate_px': masker.cfg.get('dilate_px', 0),
                             'note': 'The handheld recordings do not contain the excavator; the mask is where the '
                                     'deployed runner blanks it. Camera mounts (handheld and robot) are unmeasured.'},
              'mask_fraction': {'mean': float(np.mean(frac)), 'min': float(np.min(frac)), 'max': float(np.max(frac))},
              'episodes': episodes}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return {k: result[k] for k in ('mask_fraction',)} | {'episodes': len(episodes)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True, help='policy.preprocessing.export .../source (actathon.pi05.real.v1)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, help='self-mask config (default configs/hardware/self_mask.json)')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    print(json.dumps(render(a.data, a.output, a.config, a.workers), indent=2))


if __name__ == '__main__':
    main()
