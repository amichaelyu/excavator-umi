"""Hardware calibration of the RC excavator from its three link IMUs (boom, stick, bucket).

Each arm link carries an IMU read by the Pi dashboard (umi-data-collection ``imus.json``: bucket on I2C
bus 1, stick on bus 3, boom on bus 5). All arm joints turn about parallel horizontal axes, so a
link's pitch is the rotation of the gravity it reads about the joint axis in its own frame, and
each joint is the difference of two neighbouring link pitches. The routine measures what
:class:`digital_twin.hardware.state_estimator.ArmIMUEstimator` and the policy runner need:

* every IMU's joint axis (its frame), from the gyro while its own joint moves;
* every IMU's at-rest gravity reading with every joint at its home end (stick up, boom up, bucket
  in) = the reference pose, at the model's home angles (hardware_map ``homing``);
* each joint's end-to-end range, its speed in each direction and its press-to-motion dead time (press to
  half the steady rate), where both of its pins move it;
* the pin signs: the axes are oriented by each joint's own + drives, so the IMUs see relative reversals (a
  child IMU turning the opposite way while its parent moves); the reversed joint gets
  ``joints.<j>.sign_verified: false``, a joint both of whose pins move it consistently is verified;
* the base turn (slew), with the base facing forward at the start: 300 / 600 ms presses left and right,
  each undone by the other pin; the yaw rate is every link IMU's gyro projected on its at-rest gravity.
  Steady turn rate, press-to-motion onset and pin sign per direction (``--no-slew`` leaves it out).

The joints have no hard stops: at the end of its travel a joint's motor keeps driving and the link
chatters (gyro +-40 deg/s, accel 0.3-2.6 g) without net motion, and the motor can run on for up to
~1 s after the press ends. So the routine streams every IMU at full rate (Pi
``/api/imu/recent?imu=<name>``) and, per drive, presses a joint in 0.5 s chunks while classifying its
own IMU's gyro in 0.25 s windows: *moving* (mean rate vector > ``MOVING_DPS``), *chatter* (no net
rate but vibration), *quiet*. The end of travel is the first two non-moving windows after motion. It
then waits until every IMU is still (``wait_still``) before reading gravity. A pin that never moves
its joint is reported (``no motion``) instead of producing a bogus range.

Sequence: check -> home every joint -> reference -> ``--visits`` round trips per joint (away end and
back home; the boom only after the clearance question, with the stick and bucket out) -> home -> base
turn -> fit -> merge into configs/hardware/hardware_map.json: ``calibration.arm_imus``, ``calibration.ranges`` /
``speeds`` (range_measured_rad, speed_measured_rad_s, dead_time_s: digital_twin.sim.joint_config, the runner's
motor onsets), the top-level ``slew`` (yaw_deg_s, onset_s, sign) + ``calibration.slew``, and
``joints.<j>.sign_verified``. A joint whose away pin does not move it keeps its axis and reference (enough
for the state estimate) but has no range. A run that fails its checks merges nothing but reversed pins.
The stick / base coupling (top-level ``coupling``) is not measured here and is kept. Every run writes
``artifacts/arm-imu-calibration/<stamp>[-sim]/calibration.json`` and ``log.txt``. The inference dashboard runs
the same routine in-process (hub.apps.inference.calibration_jobs).

Modules: ``rig`` (SimArm, LiveArm, ImuStream), ``routine`` (measurements, base turn, fit, run_calibration),
``hardware_map`` (the merge), ``cli`` (parse_args, run).

    python -m digital_twin.hardware.calibrate                                  # simulated arm, nothing sent
    python -m digital_twin.hardware.calibrate --live --i-understand-open-loop  # real machine
    python -m digital_twin.hardware.calibrate --watch                          # live joint angles from the calibration
"""
from .cli import ARTIFACTS, main, parse_args, run
from .hardware_map import merge_calibration, update_hardware_map, write_hardware_map
from .rig import CalibrationError, ImuStream, LiveArm, SimArm, joint_pins, utc_stamp
from .routine import (SLEW_MS, classify, drive, fit_calibration, measure_joints, measure_slew, pin_signs,
                      run_calibration, servo_joint, wait_still, watch, window)

__all__ = ['ARTIFACTS', 'SLEW_MS', 'CalibrationError', 'ImuStream', 'LiveArm', 'SimArm', 'classify', 'drive',
           'fit_calibration', 'joint_pins', 'main', 'measure_joints', 'measure_slew', 'merge_calibration', 'parse_args',
           'pin_signs', 'run', 'run_calibration', 'servo_joint', 'update_hardware_map', 'utc_stamp', 'wait_still',
           'watch', 'window', 'write_hardware_map']
