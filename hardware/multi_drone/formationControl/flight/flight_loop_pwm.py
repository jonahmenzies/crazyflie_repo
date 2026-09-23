"""Fly with send_setpoint — the paper's u applied on all four channels.

Roll, pitch, yaw rate and thrust all come from the control law. Thrust
needs a per-drone newton-to-PWM calibration, which is the weak point: see
setpoint_map_pwm.py for why.

    python3 flight/flight_loop_PWM.py 5
"""

import sys

import setpoint_map_pwm
from flight_loop import fly, dry

if __name__ == '__main__':
	args = [a for a in sys.argv[1:] if not a.startswith('-')]
	interval = float(args[0]) if args else 5.0

	if '--dry' in sys.argv:
		dry(interval)
	else:
		fly(interval, note=f'{interval}s replan, send_setpoint (PWM)',
		    mapper=setpoint_map_pwm)
