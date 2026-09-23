"""Fly with send_full_state_setpoint — firmware closes the position loop.

No thrust calibration needed, but the firmware regenerates its own attitude
from position error, so u[0], u[1] and u[3] are computed and discarded. u
enters only as z acceleration feedforward. Least faithful of the three, but
the most likely to stay in the air.

    python3 flight/flight_loop_full.py 5
"""

import sys

import setpoint_map
from flight_loop import fly, dry

if __name__ == '__main__':
	args = [a for a in sys.argv[1:] if not a.startswith('-')]
	interval = float(args[0]) if args else 5.0

	if '--dry' in sys.argv:
		dry(interval)
	else:
		fly(interval, note=f'{interval}s replan, send_full_state_setpoint',
		    mapper=setpoint_map)
