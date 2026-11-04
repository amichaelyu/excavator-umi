"""Pure metrics, and policy.eval.cloak.evaluate on a synthetic frozen cohort (ported from the cloak tests)."""
import numpy as np
import pytest

from policy.eval import metrics
from policy.eval.cloak import evaluate
from policy.data.cloak_dataset import digest, prepare, prepare_out_of_view


def test_score_errors_per_joint_and_normalized():
    targets = np.zeros((2, 3, 2))
    prediction = np.broadcast_to([1., -2.], targets.shape)
    s = metrics.score(prediction, targets, scale=np.array([1., 4.]))
    assert s['mae_rad'] == [1., 2.] and s['rmse_rad'] == [1., 2.]
    assert s['normalized_mae'] == pytest.approx((1 + .5) / 2)


def test_normalization_floors_constant_joints():
    mean, scale = metrics.normalization([[1., 5.], [3., 5.]])
    assert mean.tolist() == [2., 5.] and scale.tolist() == [1., 1e-6]


def test_episode_mean_weighs_episodes_equally_and_strata_only_when_present():
    def ep(p, warnings=()):
        return {name: {'normalized_mae': p} for name in metrics.BASELINES} | {'motion_warnings': list(warnings)}
    episodes = [ep(1.), ep(3.), ep(8., ['fast slew'])]
    assert metrics.episode_mean(episodes) == {name: 4. for name in metrics.BASELINES}
    strata = metrics.motion_quality_strata(episodes)
    assert strata['no_motion_warnings']['episodes'] == 2
    assert strata['no_motion_warnings']['episode_mean_normalized_mae']['policy'] == 2.
    assert strata['motion_warnings']['episode_mean_normalized_mae']['policy'] == 8.
    assert list(metrics.motion_quality_strata(episodes[:2])) == ['no_motion_warnings']


def test_evaluation_uses_only_holdout_and_complete_chunks(frozen):
    source, masks, output = frozen
    prepare(source, masks, output)

    class Repeat:
        def infer(self, obs):
            assert obs['state'].tolist() == [100, 100]
            assert obs['embodiment_mask'].any()
            assert 'actions' not in obs
            return {'actions': np.repeat(obs['state'][None], 2, axis=0)}
    result = evaluate(Repeat(), output, horizon=2)
    assert len(result['episodes']) == 1
    e = result['episodes'][0]
    assert e['id'] == '1' and e['windows'] == 1
    assert e['policy'] == e['repeat_state']
    assert e['policy']['mae_rad'] == [.5, .5]
    assert e['training_mean']['normalized_mae'] > e['policy']['normalized_mae']
    assert result['split'] == 'holdout' and result['manifest_sha256'] == digest(output / 'manifest.json')
    assert result['episode_mean_normalized_mae']['policy'] == e['policy']['normalized_mae']
    assert result['scope'] == metrics.SCOPE


def test_mask_stress_evaluation_changes_only_visibility(frozen, repeat):
    source, _, output = frozen
    prepare_out_of_view(source, output, embodiment='handheld-test', content_box=[0, 0, 224, 224],
                        evidence={'confirmation': 'Synthetic test source is out of view',
                                  'source_manifest_sha256': digest(source / 'manifest.json')})
    policy = repeat()
    clean = evaluate(policy, output, horizon=2)
    masked = evaluate(policy, output, horizon=2, mask_seed=7)
    seen = policy.seen
    assert clean['condition'] == 'recorded_view' and masked['condition'] == 'synthetic_edge_mask'
    np.testing.assert_array_equal(seen[0]['rgb'], seen[1]['rgb'])
    np.testing.assert_array_equal(seen[0]['state'], seen[1]['state'])
    assert not seen[0]['embodiment_mask'].any() and seen[1]['embodiment_mask'].any()
    assert clean['episodes'] == masked['episodes']


def test_mask_stress_needs_an_out_of_view_source(frozen, repeat):
    source, masks, output = frozen
    prepare(source, masks, output)
    with pytest.raises(ValueError, match='out-of-view'):
        evaluate(repeat(), output, horizon=2, mask_seed=7)


def test_train_split_and_saved_predictions(frozen, repeat, tmp_path):
    source, masks, output = frozen
    prepare(source, masks, output)
    path = tmp_path / 'chunks.npz'
    result = evaluate(repeat(), output, horizon=2, split='train', predictions_path=path)
    assert result['split'] == 'train' and [e['id'] for e in result['episodes']] == ['0']
    with np.load(path) as z:
        assert z['0/policy'].shape == (1, 2, 2) and z['0/target'].shape == (1, 2, 2)
    with pytest.raises(ValueError, match='split'):
        evaluate(repeat(), output, horizon=2, split='test')


def test_rejects_wrong_action_shape(frozen, repeat):
    source, masks, output = frozen
    prepare(source, masks, output)
    with pytest.raises(ValueError, match='horizon'):
        evaluate(repeat(horizon=3), output, horizon=2)


def test_evaluate_checkpoint_logs_its_run_reports_to_wandb(frozen, repeat, tmp_path, monkeypatch):
    """Each eval-*.json written into the run directory goes to wandb_run.log_eval with its step / split; a report
    elsewhere or of a foreign dataset does not."""
    from policy.eval import cloak
    from policy.training.cloak import policy as cloak_policy, wandb_run
    source, masks, output = frozen
    prepare(source, masks, output)
    model = repeat()
    model.metadata = {'action_horizon': 2}
    monkeypatch.setattr(cloak_policy, 'load_policy', lambda *args, **kwargs: model)
    logged = []
    monkeypatch.setattr(wandb_run, 'log_eval', lambda run_dir, report, **kw: logged.append((run_dir, kw)))
    run = tmp_path / 'run'
    cloak.evaluate_checkpoint('backend', output, run, run / 'eval-250.json', checkpoint='c', progress=lambda e: None)
    cloak.evaluate_checkpoint('backend', output, run, run / 'eval-initial-train.json', pretrained=True,
                              split='train', progress=lambda e: None)
    assert logged == [(run, {'step': 250, 'masked': False, 'split': 'holdout'}),
                      (run, {'step': 0, 'masked': False, 'split': 'train'})]
    cloak.evaluate_checkpoint('backend', output, run, tmp_path / 'elsewhere' / 'eval-250.json', checkpoint='c',
                              progress=lambda e: None)
    cloak.evaluate_checkpoint('backend', output, run, run / 'eval-500.json', checkpoint='c', policy_data=output,
                              progress=lambda e: None)
    assert len(logged) == 2
