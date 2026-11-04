"""Paper A.1 silhouette-IoU camera-mount fitting against supplied annotations.

The paper's dark-gripper heuristic does not describe handheld footage with soil
and hands. This adapter takes explicit silhouette labels instead. Labels only
calibrate the renderer; no learned segmentation model runs during inference.
"""
import copy

import numpy as np
from scipy.optimize import minimize


def fit_mount(renderer, states, masks, *, max_iterations=200):
    """Fit six mount coordinates without certifying the resulting calibration.

    renderer must have mask(q, shape), mount, and set_mount. The state dictionaries
    must be expressed in this geometry model's coordinates. Camera intrinsics
    and distortion remain fixed. Translation parameters are scaled to cm for
    Nelder-Mead conditioning; angles are scaled to 10 degrees.
    """
    masks = np.asarray(masks)
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    if masks.dtype != np.bool_ or masks.ndim != 3 or len(states) != len(masks) or not len(states):
        raise ValueError('Provide one boolean silhouette per source joint state')
    if not masks.any(axis=(1, 2)).all() or masks.all(axis=(1, 2)).any():
        raise ValueError('Calibration silhouettes must contain foreground and background')
    original = copy.deepcopy(renderer.mount)
    scale = np.array([.01] * 3 + [10.] * 3)
    initial = np.r_[original['pos_m'], original['rpy_deg']] / scale

    def objective(x):
        values = x * scale
        renderer.set_mount(pos_m=values[:3], rpy_deg=values[3:])
        losses = []
        for q, truth in zip(states, masks):
            predicted = renderer.mask(q, truth.shape)
            losses.append(1 - np.count_nonzero(predicted & truth) / np.count_nonzero(predicted | truth))
        return float(np.mean(losses))

    try:
        before = objective(initial)
        simplex = np.vstack([initial, initial + np.eye(6) * .1])
        fit = minimize(objective, initial, method='Nelder-Mead',
                       options={'maxiter': max_iterations, 'initial_simplex': simplex,
                                'xatol': 1e-4, 'fatol': 1e-5})
        best = fit.x if fit.fun < before else initial
        after = objective(best)
        values = best * scale
        return {'mount': {**original, 'pos_m': values[:3].tolist(), 'rpy_deg': values[3:].tolist()},
                'initial_iou': 1 - before, 'fitted_iou': 1 - after,
                'optimizer_converged': bool(fit.success), 'iterations': int(fit.nit),
                'measured': False, 'review_required': True,
                'note': 'Fit to supplied silhouettes; validate on independent frames before accepting.'}
    finally:
        renderer.set_mount(pos_m=original['pos_m'], rpy_deg=original['rpy_deg'])
