"""Report names and aggregation into a summary table."""
import json

from policy.eval import report


def test_report_names():
    for name, step, masked in (('eval-initial.json', 0, False), ('eval-250.json', 250, False),
                               ('eval-750-masked.json', 750, True)):
        parsed = report.parse_name(name)
        assert (parsed['step'], parsed['masked'], parsed['split']) == (step, masked, 'holdout')
    assert report.parse_name('eval-250-train.json') == {'label': '250', 'step': 250, 'masked': False,
                                                        'split': 'train'}
    assert report.parse_name('eval-250-predictions.npz') is None and report.parse_name('result.json') is None


def test_aggregate_rows_best_and_selected_step(tmp_path, report_file):
    for name, policy, seed in [('initial', .1, None), ('250', .4, None), ('500', .3, None),
                               ('500-masked', .05, 7), ('250-train', .01, None)]:
        report_file(tmp_path / f'eval-{name}.json', 'abc', policy, seed)
    (tmp_path / 'eval-9-predictions.npz').write_bytes(b'')
    (tmp_path / 'result.json').write_text(json.dumps({'selected_step': 500}))
    s = report.aggregate(tmp_path)
    assert [r['name'] for r in s['rows']] == ['eval-initial', 'eval-250', 'eval-500', 'eval-500-masked',
                                              'eval-250-train']
    # The pretrained model, the stress test and the train split never win the selection.
    assert s['best'] == {'name': 'eval-500', 'step': 500, 'policy': .3}
    assert s['selected_step'] == 500 and s['datasets'] == ['abc']
    assert s['rows'][3]['masked'] and s['rows'][3]['mask_seed'] == 7 and s['rows'][4]['split'] == 'train'
    table = report.format_table(s)
    assert 'eval-500 *' in table and '(selected step 500)' in table and '0.300' in table


def test_empty_run_and_mixed_datasets(tmp_path, report_file):
    assert report.aggregate(tmp_path)['rows'] == []
    assert 'no eval-*.json reports' in report.format_table(report.aggregate(tmp_path))
    report_file(tmp_path / 'eval-250.json', 'abc', .4)
    report_file(tmp_path / 'eval-500.json', 'def', .3)
    assert 'different datasets' in report.format_table(report.aggregate(tmp_path))


def test_rl_summary_rows(tmp_path):
    (tmp_path / 'evaluation.json').write_text(json.dumps({
        'profile': 'arm2', 'actual_steps': 1000,
        'rl': {'success_rate': .8, 'unstable_episodes': 0, 'episodes': [{}]},
        'random': {'success_rate': .1, 'unstable_episodes': 2, 'episodes': [{}]}}))
    s = report.rl_summary(tmp_path)
    assert s['profile'] == 'arm2' and [r['name'] for r in s['rows']] == ['rl', 'random']
    assert s['rows'][0] == {'name': 'rl', 'success_rate': .8, 'unstable_episodes': 0}
    assert 'success_rate' in report.format_rl_table(s)
