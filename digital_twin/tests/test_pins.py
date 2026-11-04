"""digital_twin.sim.pins: each GPIO lives once in the hardware map, on its joint; helpers, migration and label edits."""
import json
import re

import pytest

from digital_twin.sim import pins
from digital_twin.sim.commands import validate_hardware_map
from digital_twin.paths import DEFAULT_HARDWARE_MAP

LEGACY = {
    'pins': {'19': 'base left', '21': 'base right', '32': 'boom down', '33': 'boom up', '25': 'bucket up',
             '26': 'bucket down', '22': 'arm down', '23': 'arm up'},
    'joints': {'slew': {'function': 'base turn', 'pin_positive': 19, 'pin_negative': 21, 'sign': 1},
               'boom': {'function': 'big arm', 'pin_positive': 33, 'pin_negative': 32, 'sign': 1, 'sign_verified': False},
               'stick': {'function': 'stick', 'pin_positive': 23, 'pin_negative': 22, 'sign': 1},
               'bucket': {'function': 'gripper', 'pin_positive': 25, 'pin_negative': 26, 'sign': 1}},
    'homing': {'order': ['boom'], 'joints': {'boom': {'home_direction': 1, 'home_gpio': 33, 'home_value_rad': 0.5,
                                                      'home_duration_s': 4.9, 'speed_rad_s': 0.52, 'range_rad': [-0.8, 0.5]}}},
    'calibration': {'arm_imus': {'checks': {'joints': {'boom': {
        'pins': {'33': {'direction': 1, 'moved': True, 'results': ['end']},
                 '32': {'direction': -1, 'moved': False, 'results': ['no_motion']}},
        'visits': [], 'range_spread_deg': 0.1}}}}},
    'coupling': {'channels': {'22': {'stick_deg_s': -30.0, 'stick_stop': 'down'},
                              '23': {'stick_deg_s': 25.0, 'stick_stop': 'up'}}, 'pins_never_pulsed': []},
    'slew': {'pin_positive': 19, 'pin_negative': 21, 'sign': 1},
}


def legacy():
    return json.loads(json.dumps(LEGACY))


def test_migrate_moves_every_pin_onto_its_joint():
    hw = pins.migrate(legacy())
    assert 'pins' not in hw and hw['slew'] == {'sign': 1}
    assert hw['joints']['boom'] == {'positive': {'pin': 33, 'label': 'boom up'},
                                    'negative': {'pin': 32, 'label': 'boom down'}, 'sign': 1, 'sign_verified': False}
    assert hw['homing']['joints']['boom'] == {'home_direction': 1, 'home_value_rad': 0.5, 'home_duration_s': 4.9}
    c = hw['calibration']['arm_imus']['checks']['joints']['boom']
    assert list(c) == ['drives', 'visits', 'range_spread_deg']
    assert c['drives'] == {'positive': {'moved': True, 'results': ['end']},
                           'negative': {'moved': False, 'results': ['no_motion']}}
    assert hw['coupling'] == {'channels': {'negative': {'stick_deg_s': -30.0, 'stick_stop': 'down'},
                                           'positive': {'stick_deg_s': 25.0, 'stick_stop': 'up'}}}
    validate_hardware_map(hw)
    assert pins.migrate(json.loads(json.dumps(hw))) == hw               # a current map is unchanged


def test_old_wiring_coupling_channels_are_the_stick_directions():
    hw = legacy(); hw['joints'].pop('stick')
    hw['coupling']['channels'] = {'27': {'yaw_deg_s': 63.0}, '22': {'yaw_deg_s': -63.0}}
    assert set(pins.migrate(hw)['coupling']['channels']) == {'positive', 'negative'}
    assert hw['coupling']['channels']['positive'] == {'yaw_deg_s': 63.0}


def test_helpers():
    hw = pins.migrate(legacy())
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
    path.write_text(json.dumps(legacy(), indent=2) + '\n')              # an older map is migrated on its first save
    assert pins.pin_labels(path)[33] == 'boom up'
    new = {**pins.pin_labels(path), 33: 'big arm up'}
    out = pins.save_pin_labels(new, path)
    hw = json.loads(path.read_text())
    assert out['33'] == 'big arm up' and hw['joints']['boom']['positive'] == {'pin': 33, 'label': 'big arm up'}
    assert 'pins' not in hw and 'pin_positive' not in hw['joints']['boom']
    assert hw['homing']['joints']['boom']['home_value_rad'] == 0.5      # the rest of the map is kept
    before = path.read_text()
    with pytest.raises(ValueError, match='not a joint pin'):
        pins.save_pin_labels({**new, 4: 'spare'}, path)
    with pytest.raises(ValueError, match='needs a name'):
        pins.save_pin_labels({**new, 32: ''}, path)
    assert path.read_text() == before                                   # refused edits write nothing
    assert pins.pin_labels(tmp_path / 'missing.json') is None


def test_the_checked_in_map_has_each_gpio_once():
    text = DEFAULT_HARDWARE_MAP.read_text()
    hw = json.loads(text)
    assert pins.migrate(json.loads(text)) == hw                         # already in the current shape
    validate_hardware_map(hw)
    numbers = [int(n) for n in re.findall(r'(?<![\w.\-])(\d+)(?![\w.])', text)]
    for gpio in pins.labels(hw):
        assert numbers.count(gpio) == 1, gpio
