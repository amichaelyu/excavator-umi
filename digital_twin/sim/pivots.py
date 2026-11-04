"""Scan-frame geometry: the scan -> model coordinate transform and the anatomical pivot chain.

``robot.json`` gives each link's pivot (``scan_pivots``) and frame origin (``scan_frames``) in the raw
scan's coordinates; :func:`transform` maps them into the model frame (metres, z up) and :func:`fk`
chains them: slew about +z, then boom / stick / bucket about -y. No MuJoCo involved.
"""
import numpy as np
from scipy.spatial.transform import Rotation

LINKS = ['base', 'slew', 'boom', 'stick', 'bucket']


def transform(v, cfg):
    p = (np.asarray(v)-cfg['scan_origin'])*cfg['scan_scale']
    return p[..., [0, 2, 1]]*np.array([1, -1, 1])


def fk(q,cfg):
    p={k:transform(v,cfg) for k,v in cfg['scan_pivots'].items()}
    frames={k:transform(v,cfg) for k,v in cfg.get('scan_frames',cfg['scan_pivots']).items()}
    rotations=[np.eye(3)];translations=[np.array([0,0,.001])]
    prev=np.zeros(3)
    for i,name in enumerate(LINKS[1:]):
        axis=np.array([0,0,1]) if i==0 else np.array([0,-1,0])
        local_R=Rotation.from_rotvec(axis*q[i]).as_matrix()
        anchor=p[name]-frames[name]
        translations.append(translations[-1]+rotations[-1]@(frames[name]-prev+anchor-local_R@anchor))
        rotations.append(rotations[-1]@local_R)
        prev=frames[name]
    return rotations,translations
