"""policy.data.local_config: the hosts block of configs/hub/local.json, its defaults and the committed example."""
import json
from pathlib import Path

import pytest

from policy.data import local_config as lc

REPO = Path(lc.__file__).resolve().parents[2]


def write(tmp_path, doc):
    path = tmp_path / 'local.json'
    path.write_text(json.dumps(doc))
    return path


def test_missing_file_or_block_gives_the_defaults(tmp_path):
    for path in (tmp_path / 'absent.json', write(tmp_path, {'apps': {'inference': {}}})):
        assert lc.hosts(path) == {}
        assert lc.pi_urls(path) == list(lc.DEFAULT_PI_URLS)
        assert lc.pi_token_file(path) == REPO / lc.DEFAULT_PI_TOKEN_FILE
        assert lc.lambda_host(path) is None
        assert lc.lambda_root(path) == '~/umi-excavator'


def test_values_from_the_file(tmp_path):
    path = write(tmp_path, {'hosts': {'pi': {'urls': 'http://a:8000, http://b:8000', 'token_file': '~/tok'},
                                      'lambda': {'host': 'ubuntu@203.0.113.9', 'root': '/srv/x'}}})
    assert lc.pi_urls(path) == ['http://a:8000', 'http://b:8000']
    assert lc.pi_token_file(path) == Path('~/tok').expanduser()
    assert (lc.lambda_host(path), lc.lambda_root(path)) == ('ubuntu@203.0.113.9', '/srv/x')
    assert lc.pi_token_file(write(tmp_path, {'hosts': {'pi': {'token_file': 'rel/tok'}}})) == REPO / 'rel/tok'


def test_environment_names_the_file(tmp_path, monkeypatch):
    path = write(tmp_path, {'hosts': {'pi': {'urls': ['http://pi.example:8000']}}})
    monkeypatch.setenv(lc.ENV_VAR, str(path))
    assert lc.path() == path and lc.pi_urls() == ['http://pi.example:8000']
    monkeypatch.delenv(lc.ENV_VAR)
    assert lc.path() == lc.LOCAL == REPO / 'configs/hub/local.json'


def test_bad_shapes_are_refused(tmp_path):
    with pytest.raises(ValueError, match='hosts'):
        lc.hosts(write(tmp_path, {'hosts': ['pi']}))
    with pytest.raises(ValueError, match='hosts.pi'):
        lc.pi_urls(write(tmp_path, {'hosts': {'pi': 'raspberrypi.local'}}))


def test_the_example_parses_and_uses_placeholders():
    example = REPO / 'configs/hub/local.example.json'
    assert lc.pi_urls(example)[1] == 'http://raspberrypi.local:8000'
    assert lc.lambda_host(example) == 'ubuntu@203.0.113.10'          # TEST-NET-3, not a real box
    assert 'configs/hub/local.json' in (REPO / '.gitignore').read_text().splitlines()


def test_ensure_local_creates_the_machine_copy_and_writers_refuse_the_template(tmp_path, monkeypatch):
    from policy.data.local_config import EXAMPLES_ENV, check_writable, ensure_local
    example = tmp_path / 'hardware_map.example.json'
    example.write_text('{"joints": {}}')
    local = tmp_path / 'hardware_map.json'
    monkeypatch.setenv(EXAMPLES_ENV, '1')                       # tests read the template...
    assert ensure_local(local) == example and not local.exists()
    assert ensure_local(local, examples=False) == local         # ...but shipping always makes the machine copy
    assert local.read_text() == example.read_text()
    monkeypatch.delenv(EXAMPLES_ENV)
    local.write_text('{"joints": {"boom": {}}}')
    assert ensure_local(local) == local and 'boom' in local.read_text()    # never overwritten once it exists
    assert not list(tmp_path.glob('.*.tmp'))
    assert check_writable(local) == local
    with pytest.raises(ValueError, match='tracked template'):
        check_writable(example)
