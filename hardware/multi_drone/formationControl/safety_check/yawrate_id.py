# Measures a_r: how fast yaw rate follows its command.
#
# The drone hovers on position setpoints, then gets short yaw rate steps.
# Each step is fitted to  rate(t) = cmd * (1 - exp(-a*t)),  tau = 1/a
#
#     python3 safety_check/yawrate_id.py [uri]      default is drone 1
#
# Run from formationControl/

import sys
import time

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from scipy.optimize import curve_fit

from common import uri_from_args, preflight, reset_estimator, climb, hold, land

URI      = uri_from_args()
Z        = 0.6      # hover height, m
STEP_DPS = 60.0     # commanded yaw rate, deg/s
STEP_S   = 0.5      # how long each step lasts, s
RECOVER  = 2.5      # position hold between steps, s
N_STEPS  = 6        # number of steps

samples = []              # (time, yaw rate)
hover = {'pwm': None}     # thrust the firmware uses to hover


def on_rate(timestamp, data, logconf):
	samples.append((time.time(), data['gyro.z']))


def on_thrust(timestamp, data, logconf):
	hover['pwm'] = data['stabilizer.thrust']


# ---- Fly the steps ----

cflib.crtp.init_drivers()
print(f'Connecting to {URI}...')

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
	cf = scf.cf
	preflight(scf)

	# Yaw rate every 10 ms
	rate = LogConfig(name='Rate', period_in_ms=10)
	rate.add_variable('gyro.z', 'float')
	cf.log.add_config(rate)
	rate.data_received_cb.add_callback(on_rate)

	# Thrust, to find the hover PWM
	thr = LogConfig(name='Thr', period_in_ms=100)
	thr.add_variable('stabilizer.thrust', 'float')
	cf.log.add_config(thr)
	thr.data_received_cb.add_callback(on_thrust)

	reset_estimator(cf)

	print(f'\n{N_STEPS} yaw-rate steps at {STEP_DPS} deg/s.')
	print('The drone spins in place — keep props clear.')
	input('Enter to arm, Ctrl-C to abort: ')

	events = []    # (start time, sign) of each step

	try:
		cf.commander.send_setpoint(0, 0, 0, 0)    # unlock
		climb(cf, Z, 3.0)

		# Read the hover thrust (waits up to 2 s)
		thr.start()
		t0 = time.time()
		while hover['pwm'] is None and time.time() - t0 < 2.0:
			cf.commander.send_position_setpoint(0, 0, Z, 0)
			time.sleep(0.05)
		thr.stop()

		pwm = int(hover['pwm']) if hover['pwm'] else 38000
		print(f'hover pwm {pwm}, stepping...')

		rate.start()
		for n in range(N_STEPS):
			sign = 1.0 if n % 2 == 0 else -1.0    # alternate direction

			# Settle on position setpoints
			hold(cf, Z, RECOVER)

			# Step, on attitude setpoints
			events.append((time.time(), sign))
			t0 = time.time()
			while time.time() - t0 < STEP_S:
				cf.commander.send_setpoint(0, 0, sign * STEP_DPS, pwm)
				time.sleep(0.01)

		rate.stop()
		time.sleep(0.3)
		land(cf, Z, 3.0)

	except KeyboardInterrupt:
		print('\naborted')
		cf.commander.send_stop_setpoint()


# ---- Fit each step ----

if not samples:
	sys.exit('no rate samples')

d = np.array(samples)
print(f'\n{len(d)} samples, mean period {1000*np.diff(d[:,0]).mean():.1f} ms')


# First-order step response
def rise(t, a, amp, off):
	return off + amp * (1.0 - np.exp(-a * t))


good = []    # good values of a

for t_start, sign in events:
	# Samples during this step
	in_step = (d[:, 0] >= t_start) & (d[:, 0] < t_start + STEP_S)
	seg = d[in_step]
	if len(seg) < 20:
		continue

	tt = seg[:, 0] - t_start
	yy = seg[:, 1]

	try:
		popt, _ = curve_fit(rise, tt, yy, p0=[10.0, sign * STEP_DPS, yy[0]], maxfev=8000)
	except Exception:
		continue

	# How well the fit matches (R^2)
	a_hat = popt[0]
	pred = rise(tt, *popt)
	r2 = 1 - ((yy - pred)**2).sum() / ((yy - yy.mean())**2).sum()

	# Keep only good fits
	if a_hat > 0 and r2 > 0.7:
		good.append(a_hat)

	tau = 1/a_hat if a_hat > 0 else float('nan')
	print(f'  yaw {sign:+.0f}  a = {a_hat:6.2f}  '
	      f'tau = {tau:5.3f}s  R2 = {r2:5.3f}  n = {len(seg)}')


# ---- Result ----

if good:
	a = np.array(good)
	print(f'\nyaw   a_r = {a.mean():6.2f} +/- {a.std():5.2f}   '
	      f'tau = {1/a.mean():5.3f}s   ({len(a)} good fits)')
	print(f'\nparams.hpp:\n  a_r = {a.mean():.2f};')
else:
	print('\nno usable fits')
