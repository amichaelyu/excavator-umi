"""digital_twin.sim.pins: each GPIO lives once in the hardware map, on its joint; helpers and label edits."""
import json
import re

import pytest

from digital_twin.sim import pins
from digital_twin.sim.commands import validate_hardware_map
from digital_twin.paths import hardware_map

MAP = {
    'joints': {'slew': {'positive': {'pin': 19, 'label': 'base left'}, 'negative': {'pin': 21, 'label': 'base right'}, 'sign': 1},
               'boom': {'positive': {'pin': 33, 'label': 'boom up'}, 'negative': {'pin': 32, 'label': 'boom down'},
                        'sign': 1, 'sign_verified': False},
               'stick': {'positive': {'pin': 23, 'label': 'arm up'}, 'negative': {'pin': 22, 'label': 'arm down'}, 'sign': 1},
               'bucket': {'positive': {'pin': 25, 'label': 'bucket up'}, 'negative': {'pin': 26, 'label': 'bucket down'}, 'sign': 1}},
    'homing': {'order': ['boom'], 'joints': {'boom': {'home_direction': 1, 'home_value_rad': 0.5, 'home_duration_s': 4.9}}},
}


def current():
    return json.loads(json.dumps(MAP))


def test_helpers():
    hw = current()
    assert pins.pin(hw, 'boom', 1) == 33 and pins.pin(hw, 'boom', 'negative') == 32 and pins.pin(hw, 'bucket', -2) == 26
    assert pins.entry_pin(hw['joints']['stick'], -1) == 22
    assert pins.side(1) == 'positive' and pins.side(-1) == 'negative' and pins.side('negative') == 'negative'
    with pytest.raises(ValueError):
        pins.side('up')
    with pytest.raises(KeyError):
        pins.pin(hw, 'tracks', 1)
    assert pins.labels(hw) == {19: 'base left', 21: 'base right', 22: 'arm down', 23: 'arm up', 25: 'bucket up',
                               26: 'bucket down', 32: 'boom down', 33: 'boom up'}
    assert pins.label(hw, 26) == 'bucket down' and pins.label(hw, '33') == 'boom up' and pins.label(hw, 27) == ''


def test_pin_labels_and_save_write_into_the_joints(tmp_path):
    path = tmp_path / 'hardware_map.json'
    path.write_text(json.dumps(current(), indent=2) + '\n')
    assert pins.pin_labels(path)[33] == 'boom up'
    new = {**pins.pin_labels(path), 33: 'big arm up'}
    out = pins.save_pin_labels(new, path)
    hw = json.loads(path.read_text())
    assert out['33'] == 'big arm up' and hw['joints']['boom']['positive'] == {'pin': 33, 'label': 'big arm up'}
    assert hw['homing']['joints']['boom']['home_value_rad'] == 0.5      # the rest of the map is kept
    before = path.read_text()
    with pytest.raises(ValueError, match='not a joint pin'):
        pins.save_pin_labels({**new, 4: 'spare'}, path)
    with pytest.raises(ValueError, match='needs a name'):
        pins.save_pin_labels({**new, 32: ''}, path)
    assert path.read_text() == before                                   # refused edits write nothing
    assert pins.pin_labels(tmp_path / 'missing.json') is None


def test_the_checked_in_map_has_each_gpio_once():
    text = hardware_map().read_text()               # tests: configs/hardware/hardware_map.example.json (conftest.py)
    hw = json.loads(text)
    validate_hardware_map(hw)
    numbers = [int(n) for n in re.findall(r'(?<![\w.\-])(\d+)(?![\w.])', text)]
    for gpio in pins.labels(hw):
        assert numbers.count(gpio) == 1, gpio
