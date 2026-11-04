"""Local checkpoint catalog (configs/hub/local_checkpoints.json): the CLOAK π0.5 servers the runner can drive."""
from datetime import date
import json
import os
from pathlib import Path
import re
import secrets

from policy.configs.schema import POLICY_HZ

from ..paths import LOCAL_CHECKPOINTS
from .cloak import POLICY_SPACE, CloakPolicy

DEFAULT_REGISTRY = LOCAL_CHECKPOINTS               # configs/hub/local_checkpoints.json
# The one (backend, profile, policy_space) the registry accepts: 4-joint CLOAK π0.5 servers (raw frame plus
# embodiment mask, calibrated-twin labels) on the runner's cloak_full4 profile (digital_twin.policy.cloak).
CONTRACT = ('cloak', 'cloak_full4', POLICY_SPACE)


def entries(path=DEFAULT_REGISTRY):
    if path is None or not Path(path).is_file():
        return []
    data = json.loads(Path(path).read_text())
    rows = []
    for row in data.get('checkpoints', []):
        row = dict(row)
        row['gateway_url'] = os.environ.get(row.get('gateway_env', ''), row['gateway_url'])
        if (row['backend'], row['profile'], row['policy_space']) != CONTRACT:
            raise ValueError(f"Unsupported local checkpoint contract for {row.get('id')}: expected {CONTRACT}")
        rows.append(row)
    return rows


def runs(path=DEFAULT_REGISTRY):
    return [{'name': e['id'], 'key': e['id'], 'label': e['label'], 'version': None,
             'policy_space': e['policy_space'], 'backend': e['backend'], 'profile': e['profile'],
             'state': e['state'], 'num_steps': e['optimizer_updates'], 'base_model': e['base_model'],
             'source': 'local checkpoint registry', 'mode': e.get('mode'),
             'preferred': e.get('default', False), 'added': e.get('added'), 'task': e.get('task'), 'group': e.get('group'), 'prompt': e.get('prompt'), 'order': i,
             'checkpoints': [{'checkpoint_id': e['id'], 'step': e['step'], 'state': 'configured',
                              'gateway_url': e['gateway_url']}]}
            for i, e in enumerate(entries(path))]


def find(checkpoint_id, path=DEFAULT_REGISTRY):
    return next((e for e in entries(path) if e['id'] == checkpoint_id), None)


def _write(path, data):
    tmp = Path(path).with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def add(label, gateway_url, prompt, path=DEFAULT_REGISTRY, connect_timeout=5.0):
    """Register a Lambda CLOAK policy server after checking its contract (CloakPolicy.metadata). Returns the
    new entry."""
    label, gateway_url, prompt = (str(x or '').strip() for x in (label, gateway_url, prompt))
    if not 1 <= len(label) <= 80:
        raise ValueError('Label must be 1..80 characters')
    if not 1 <= len(prompt) <= 512:
        raise ValueError('Prompt must be 1..512 characters')
    rows = entries(path)
    same = next((e for e in rows if e['gateway_url'] == gateway_url), None)
    if same:
        raise ValueError(f"{gateway_url} is already registered as {same['label']!r}")
    policy = CloakPolicy(gateway_url, prompt, connect_timeout=connect_timeout)    # validates the URL
    try:
        md = policy.metadata()                                                      # validates the contract
    finally:
        policy.close()
    data = json.loads(Path(path).read_text()) if Path(path).is_file() else {'checkpoints': []}
    slug = re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')[:40] or 'endpoint'
    backend, profile, space = CONTRACT
    entry = {'id': f'{slug}-{secrets.token_hex(2)}', 'label': label, 'backend': backend, 'profile': profile,
             'policy_space': space, 'gateway_url': gateway_url, 'state': 'configured', 'step': None,
             'optimizer_updates': None, 'base_model': 'unknown', 'action_horizon': md['action_horizon'],
             'hz': POLICY_HZ, 'prompt': prompt, 'mode': md['mode'], 'joints': md['joints'], 'default': False,
             'added': date.today().isoformat()}
    data['checkpoints'].append(entry)
    _write(path, data)
    return entry


def remove(checkpoint_id, path=DEFAULT_REGISTRY):
    data = json.loads(Path(path).read_text()) if Path(path).is_file() else {'checkpoints': []}
    keep = [e for e in data['checkpoints'] if e['id'] != checkpoint_id]
    if len(keep) == len(data['checkpoints']):
        raise KeyError(checkpoint_id)
    data['checkpoints'] = keep
    _write(path, data)
