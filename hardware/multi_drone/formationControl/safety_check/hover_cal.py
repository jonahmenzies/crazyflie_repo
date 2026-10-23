# Thrust calibration from steady hovers, no pulses.
#
# Holds the drone in position hold and averages the firmware's own thrust
# command (stabilizer.thrust), which the onboard integral term drives to
# exactly what's needed to hover. Run it with and without a small known
# weight taped under the drone: the change in hover thrust for a known
# change in weight gives the slope directly, without the drone's own mass
# or any differentiated velocity.
#
#     python3 safety_check/hover_cal.py <uri> <added grams>
#
# Do three runs per drone, back to back: 0, then the weight, then 0 again.
# The two 0 g runs bracket the weighted one, so battery drain cancels out.
# Every run is saved to logs/cal/hover_cal.json, and once a drone has
# 0 g runs either side of a weighted one, its slope is printed.
#
# Place the drone near the origin: the hover is at x = 0, y = 0.
# Run from formationControl/

import json
import os
import sys
import time

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from common import preflight, reset_estimator, climb, hold, land, label

if len(sys.argv) < 3:
	sys.exit('usage: python3 safety_check/hover_cal.py <uri> <added grams>')

URI      = sys.argv[1]
ADDED_G  = float(sys.argv[2])
Z        = 0.7        # hover height, middle of the formation heights, m
SETTLE   = 4.0        # hover before recording, s
RECORD   = 15.0       # recording time, s
G        = 9.81
OUT      = 'logs/cal/hover_cal.json'

readings = []         # (thrust, vbat, z) every 100 ms while recording
recording = {'on': False}


def on_data(timestamp, data, logconf):
	if recording['on']:
		readings.append((data['stabilizer.thrust'], data['pm.vbat'], data['stateEstimate.z']))


# ---- Fly ----

cflib.crtp.init_drivers()
print(f'Connecting to {URI} (drone {label(URI)}), {ADDED_G:g} g added...')

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
	cf = scf.cf
	preflight(scf)

	log = LogConfig(name='Cal', period_in_ms=100)
	log.add_variable('stabilizer.thrust', 'float')
	log.add_variable('pm.vbat', 'float')
	log.add_variable('stateEstimate.z', 'float')
	cf.log.add_config(log)
	log.data_received_cb.add_callback(on_data)

	reset_estimator(cf)
	input(f'Hover at {Z} m for {SETTLE + RECORD:.0f} s. Enter to arm, Ctrl-C to abort: ')

	try:
		cf.commander.send_setpoint(0, 0, 0, 0)    # unlock
		log.start()
		climb(cf, Z, 3.0)
		hold(cf, Z, SETTLE)

		recording['on'] = True
		hold(cf, Z, RECORD)
		recording['on'] = False

		land(cf, Z, 3.0)
		log.stop()

	except KeyboardInterrupt:
		print('\naborted')
		cf.commander.send_stop_setpoint()
		sys.exit(1)


# ---- This run ----

if len(readings) < 50:
	sys.exit(f'only {len(readings)} readings, not saved')

r = np.array(readings)
run = {
	'drone':    label(URI),
	'uri':      URI,
	'added_g':  ADDED_G,
	'thrust':   float(r[:, 0].mean()),
	'thrust_sd': float(r[:, 0].std()),
	'vbat':     float(r[:, 1].mean()),
	'z_mean':   float(r[:, 2].mean()),
	'n':        len(r),
	'time':     time.time(),
}

print(f'\ndrone {run["drone"]}, {ADDED_G:g} g added')
print(f'  hover thrust  {run["thrust"]:.0f}  (sd {run["thrust_sd"]:.0f}, {run["n"]} readings)')
print(f'  battery       {run["vbat"]:.2f} V under load')
print(f'  height held   {run["z_mean"]:.3f} m')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
runs = json.load(open(OUT)) if os.path.exists(OUT) else []
runs.append(run)
with open(OUT, 'w') as f:
	json.dump(runs, f, indent=1)


# ---- Slope, once a weighted run has 0 g runs before and after it ----

mine = [x for x in runs if x['uri'] == URI]
for i, w in enumerate(mine):
	if w['added_g'] <= 0:
		continue
	before = [x for x in mine[:i] if x['added_g'] == 0]
	after = [x for x in mine[i+1:] if x['added_g'] == 0]
	if not (before and after):
		continue

	base = 0.5 * (before[-1]['thrust'] + after[0]['thrust'])
	base_v = 0.5 * (before[-1]['vbat'] + after[0]['vbat'])
	d_pwm = w['thrust'] - base
	slope = (w['added_g'] / 1000 * G) / d_pwm

	print(f'\nslope from the {w["added_g"]:g} g run: {slope:.3e} N/count '
	      f'({d_pwm:.0f} counts for {w["added_g"]:g} g)')
	print(f'hover at {base_v:.2f} V under load: {base:.0f}')
	print(f'  drift between the two 0 g runs: '
	      f'{after[0]["thrust"] - before[-1]["thrust"]:+.0f} counts, '
	      f'{after[0]["vbat"] - before[-1]["vbat"]:+.2f} V')
	print(f"\nfor setpoint_map_pwm.THRUST:\n  '{URI}': ({base:.0f}, {slope:.3e}),")
