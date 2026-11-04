"""Read a run directory's eval-*.json reports (CLOAK) into a summary.

The CLOAK rows carry ``episode_mean_normalized_mae`` per baseline (policy, repeat_state = hold the current
position, training_mean), the same numbers a run that tracks with W&B logs per report
(policy.training.cloak.wandb_run.eval_metrics).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

NAME = re.compile(r'eval-(initial|\d+)(-masked)?(?:-(train))?\.json')


def parse_name(name):
    """``eval-initial.json`` -> step 0; ``eval-750-masked.json`` -> step 750, masked; ``-train`` suffix -> split
    train. None for other files (e.g. predictions)."""
    m = NAME.fullmatch(Path(name).name)
    if not m:
        return None
    return {'label': m.group(1), 'step': 0 if m.group(1) == 'initial' else int(m.group(1)),
            'masked': bool(m.group(2)), 'split': m.group(3) or 'holdout'}


def load(run_dir):
    """{report name (file stem): report} for every eval-*.json in ``run_dir``."""
    out = {}
    for path in Path(run_dir).glob('eval-*.json'):
        if parse_name(path.name):
            try:
                out[path.stem] = json.loads(path.read_text())
            except ValueError:                  # mid-write
                continue
    return out


def row(name, report):
    """One table row: identity, condition and the episode-mean normalized MAE per baseline."""
    parsed = parse_name(name + '.json') or {}
    mae = report.get('episode_mean_normalized_mae') or {}
    episodes = report.get('episodes') or []
    return {'name': name, 'step': parsed.get('step'), 'initial': parsed.get('label') == 'initial',
            'masked': parsed.get('masked', False), 'split': parsed.get('split') or report.get('split', 'holdout'),
            'condition': report.get('condition'), 'mask_seed': report.get('mask_seed'),
            'episodes': len(episodes), 'windows': sum(e.get('windows', 0) for e in episodes),
            'policy': mae.get('policy'), 'repeat_state': mae.get('repeat_state'),
            'training_mean': mae.get('training_mean'), 'manifest_sha256': report.get('manifest_sha256')}


def _order(r):
    return (r['split'] != 'holdout', r['masked'], r['step'] if r['step'] is not None else math.inf, r['name'])


def aggregate(run_dir):
    """Rows sorted by split, condition and step; ``best`` = lowest policy error among the trained, unmasked
    holdout checkpoints (experiment.py's selection rule); ``selected_step`` from result.json when present."""
    run_dir = Path(run_dir)
    rows = sorted((row(n, r) for n, r in load(run_dir).items()), key=_order)
    candidates = [r for r in rows if r['split'] == 'holdout' and not r['masked'] and not r['initial']
                  and r['policy'] is not None]
    best = min(candidates, key=lambda r: r['policy']) if candidates else None
    result = run_dir / 'result.json'
    selected = json.loads(result.read_text()).get('selected_step') if result.is_file() else None
    return {'run_dir': str(run_dir), 'rows': rows,
            'best': {'name': best['name'], 'step': best['step'], 'policy': best['policy']} if best else None,
            'selected_step': selected,
            'datasets': sorted({r['manifest_sha256'] for r in rows if r['manifest_sha256']})}


def _fmt(v):
    return f'{v:.3f}' if isinstance(v, (int, float)) and not isinstance(v, bool) else '-'


def format_table(summary):
    """Plain-text table of ``aggregate`` output (one or several runs)."""
    runs = summary if isinstance(summary, list) else [summary]
    lines = []
    for s in runs:
        lines.append(f'{s["run_dir"]}' + (f'  (selected step {s["selected_step"]})' if s.get('selected_step') else ''))
        if not s['rows']:
            lines.append('  no eval-*.json reports')
            continue
        lines.append(f'  {"report":26s} {"episodes":>8s} {"policy":>7s} {"hold":>7s} {"mean":>7s}  condition')
        for r in s['rows']:
            mark = ' *' if s.get('best') and r['name'] == s['best']['name'] else ''
            lines.append(f'  {r["name"] + mark:26s} {r["episodes"]:8d} {_fmt(r["policy"]):>7s} '
                         f'{_fmt(r["repeat_state"]):>7s} {_fmt(r["training_mean"]):>7s}  {r["condition"] or "-"}')
        if len(s.get('datasets', [])) > 1:
            lines.append(f'  warning: reports from {len(s["datasets"])} different datasets')
    lines.append('normalized MAE (episode mean); hold = repeat current state, mean = training-mean action; '
                 '* = best trained holdout step')
    return '\n'.join(lines)

