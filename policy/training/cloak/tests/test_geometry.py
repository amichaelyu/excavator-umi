import numpy as np
import pytest

from policy.training.cloak.geometry import BucketRetargeter, checked_config


def mid(arm):
    """Middle of the boom / bucket ranges (the retargeter's [boom, bucket] state)."""
    from digital_twin.sim.kinematics import hardware_q
    return hardware_q((arm.lo + arm.hi) / 2)


def test_unmeasured_excavator_mount_is_not_handheld_calibration():
    with pytest.raises(ValueError, match='measured'):
        checked_config('configs/sim/self_mask.json', 'handheld')


def test_identity_tip_retargeting():
    r = BucketRetargeter('full4', 'full4')
    q = mid(r.source)
    chunk = np.stack([q, q + [.01, -.02], q + [.02, -.03]])
    actual, errors = r.source_actions(chunk, q)
    np.testing.assert_allclose(actual, chunk, atol=.004)
    assert max(x['position_residual_m'] for x in errors) < .003
    back, _ = r.target_state(actual[-1], q)
    np.testing.assert_allclose(back, chunk[-1], atol=.004)


def test_retargeting_refuses_invalid_and_unreachable_chunk():
    r = BucketRetargeter('full4', 'full4')
    q = mid(r.source)
    with pytest.raises(ValueError, match='finite'):
        r.source_actions([[float('nan'), 0]], q)
    with pytest.raises(ValueError, match='limits'):
        from digital_twin.sim.kinematics import hardware_q
        r.source_actions([hardware_q(r.source.hi) + 1], q)
    # Same joints but a displaced target base cannot reach the source tip.
    r.target.model.body_pos[r.target.model.body('base').id] += [10, 0, 0]
    with pytest.raises(ValueError, match='Unreachable'):
        r.source_actions([q], q)


def test_detached_bucket_mask_excludes_absent_excavator_links():
    from digital_twin.sim.self_mask import SelfMasker, load_config
    cfg = load_config()
    bucket = SelfMasker(config={**cfg, 'mask_bodies': ['bucket']})
    full = SelfMasker(config=cfg)
    q = {'boom': .5, 'bucket': -.7}
    try:
        a = bucket.mask(q, (90, 160))
        b = full.mask(q, (90, 160))
        assert not (a & ~b).any()
        assert (b & ~a).any()
        np.testing.assert_array_equal(a, bucket.mask({'boom': -.1, 'bucket': .3}, (90, 160)))
    finally:
        bucket.close(); full.close()


def test_silhouette_fit_restores_renderer_and_does_not_certify_calibration():
    import copy
    from policy.training.cloak.calibration import fit_mount
    class Renderer:
        mount = {'body': 'bucket', 'pos_m': [.01, .02, .03], 'rpy_deg': [0, 0, 0]}
        def set_mount(self, pos_m, rpy_deg):
            self.mount = {**self.mount, 'pos_m': list(pos_m), 'rpy_deg': list(rpy_deg)}
        def mask(self, q, shape):
            yy, xx = np.indices(shape)
            return (xx >= round(self.mount['pos_m'][0] * 1000)) & (xx < 30) & (yy > 12)
    renderer = Renderer(); before = copy.deepcopy(renderer.mount)
    truth = np.stack([renderer.mask({}, (40, 40))])
    fit = fit_mount(renderer, [{}], truth, max_iterations=3)
    assert fit['fitted_iou'] == 1
    assert fit['review_required'] and not fit['measured']
    assert renderer.mount == before
