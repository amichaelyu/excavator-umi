"""Entry point: ``python -m digital_twin.hardware.calibrate`` (the package docstring is its help); :func:`run`
is the same work for in-process callers (the inference dashboard's calibration job)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from digital_twin.sim import pins

from ...paths import HARDWARE_MAP, REPO
from ..pi import DEFAULT_PI_HOST, DEFAULT_TOKEN_FILE
from .hardware_map import _jsonable, write_hardware_map
from .rig import CalibrationError, LiveArm, SimArm, utc_stamp
from .routine import run_calibration, watch

ARTIFACTS = REPO / 'artifacts' / 'arm-imu-calibration'


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog='python -m digital_twin.hardware.calibrate', description=sys.modules[__package__].__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--live', action='store_true', help='drive the real machine (default: simulated arm, nothing sent)')
    p.add_argument('--i-understand-open-loop', action='store_true')
    p.add_argument('--watch', action='store_true', help='print live joint angles from the calibration in hardware_map.json')
    p.add_argument('--pi-host', default=DEFAULT_PI_HOST)
    p.add_argument('--token-file', default=str(DEFAULT_TOKEN_FILE))
    p.add_argument('--bridge-host', default='localhost:8888', help='GPIO bridge: the hub host:port')
    p.add_argument('--hardware-map', default=str(HARDWARE_MAP))
    p.add_argument('--artifacts', default=str(ARTIFACTS), help='runs go to <dir>/<stamp>[-sim]/')
    p.add_argument('--visits', type=int, default=2, help='round trips per joint (each end is averaged over them)')
    p.add_argument('--no-slew', dest='slew', action='store_false', help='leave the base turn out')
    p.add_argument('--no-write-map', action='store_true', help='live: only write the artifact, not hardware_map.json')
    p.add_argument('--force', action='store_true', help='write the map despite failed checks')
    p.add_argument('--yes', action='store_true', help='answer yes to the clearance questions (boom down, base turn)')
    p.add_argument('--seed', type=int, default=0, help='simulated arm: IMU mounts and noise')
    a = p.parse_args(argv)
    a.mode = 'live' if a.live else 'sim'
    return a


def main(argv=None) -> int:
    return run(parse_args(argv))


def run(args, log=None, bridge=None, stop=None) -> int:
    """The CLI's work on parsed ``args``. ``log``: called per line (default print). ``bridge``: a
    :class:`hardware.gpio.GpioBridge` used instead of ``args.bridge_host`` (the hub's, in-process); ``stop``: a
    threading.Event that interrupts the run (live or simulated). 0: done (merged when live), 1: failed or stopped,
    2: refused."""
    hw_path = Path(args.hardware_map)
    hw = pins.read(hw_path)
    bridge = bridge if bridge is not None else args.bridge_host
    if args.watch:
        io = LiveArm(hw, args.pi_host, args.token_file, bridge, stream=False, stop=stop)
        try:
            watch(io, hw)
        except KeyboardInterrupt:
            pass
        finally:
            io.close()
        return 0
    if args.live and not args.i_understand_open_loop:
        (log or (lambda m: print(m, file=sys.stderr)))('--live needs --i-understand-open-loop (every joint is driven end to end)')
        return 2
    stamp = utc_stamp()
    out_dir = Path(args.artifacts) / (stamp + ('' if args.live else '-sim'))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    out = log or (lambda m: print(m, flush=True))

    def log(m):
        out(m); lines.append(m)
    result = {'status': 'error'}
    io = None
    try:
        io = (LiveArm(hw, args.pi_host, args.token_file, bridge, yes=args.yes, log=log, stop=stop) if args.live
              else SimArm(hw, seed=args.seed, stop=stop))
        result = run_calibration(io, hw, args, log=log)
    except KeyboardInterrupt:
        log('interrupted: releasing all pins'); result = {'status': 'interrupted'}
    except CalibrationError as ex:
        log(f'STOPPED: {ex}'); result = {'status': 'stopped', 'error': str(ex)}
    finally:
        if io is not None:
            io.close()
        if isinstance(io, SimArm):
            result['sim_truth'] = {'ends_rad': io.ends, 'speeds_rad_s': io.speeds, 'slew_speeds_rad_s': io.slew_speeds,
                                   'onset_s': io.onset_s, 'final_q_rad': io.q, 'final_slew_rad': io.slew,
                                   'pulses': len(io.pulses)}
        (out_dir / 'calibration.json').write_text(json.dumps(_jsonable(result), indent=2) + '\n')
        (out_dir / 'log.txt').write_text('\n'.join(lines) + '\n')
    log(f'artifact: {out_dir / "calibration.json"}')
    cal = result.get('calibration')
    write = args.live and not args.no_write_map
    if result.get('status') == 'done' or (cal and args.force and result.get('status') != 'stopped'):
        if write:
            write_hardware_map(hw_path, cal, stamp)
            log(f'merged into {hw_path}: calibration.arm_imus, ranges, speeds (dead times), base turn, pin signs; the policy '
                'runner and the inference twin panel use it (policy.runner --imu auto); watch it with --watch')
        elif not args.live:
            log('simulated: hardware_map.json left untouched')
        return 0
    rev = [j for j, s in ((cal or {}).get('signs') or {}).items() if s.get('sign') == -1]
    if write and rev:
        write_hardware_map(hw_path, cal, stamp, signs_only=True)
        log(f'only marked {", ".join(rev)} sign_verified false in {hw_path} (reversed pins); nothing else was merged')
    return 1


if __name__ == '__main__':
    sys.exit(main())
