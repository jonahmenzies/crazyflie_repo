# Measures a_theta and a_phi: how fast pitch and roll follow their commands.
#
# The drone hovers on position setpoints, then gets short attitude steps.
# Each step is fitted to  angle(t) = cmd * (1 - exp(-a*t)),  tau = 1/a
#
#     python3 safety_check/attitude_id.py [uri]      default is drone 1
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
STEP_DEG = 8.0      # commanded lean, degrees
STEP_S   = 0.5      # how long each step lasts, s
RECOVER  = 2.5      # position hold between steps, s
N_STEPS  = 4        # steps per axis

samples = []              # (time, roll, pitch)
hover = {'pwm': None}     # thrust the firmware uses to hover


def on_attitude(timestamp, data, logconf):
	samples.append((time.time(), data['stabilizer.roll'], data['stabilizer.pitch']))


def on_thrust(timestamp, data, logconf):
	hover['pwm'] = data['stabilizer.thrust']


# ---- Fly the steps ----

cflib.crtp.init_drivers()
print(f'Connecting to {URI}...')

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
	cf = scf.cf
	preflight(scf)

	# Attitude every 10 ms. Only two variables so the packets keep up.
	att = LogConfig(name='Att', period_in_ms=10)
	att.add_variable('stabilizer.roll', 'float')
	att.add_variable('stabilizer.pitch', 'float')
	cf.log.add_config(att)
	att.data_received_cb.add_callback(on_attitude)

	# Thrust, to find the hover PWM
	thr = LogConfig(name='Thr', period_in_ms=100)
	thr.add_variable('stabilizer.thrust', 'float')
	cf.log.add_config(thr)
	thr.data_received_cb.add_callback(on_thrust)

	reset_estimator(cf)

	print(f'\n{N_STEPS} steps per axis at {STEP_DEG} degrees.')
	print('The drone WILL lurch on each step. Needs ~1.5 m clear space.')
	input('Enter to arm, Ctrl-C to abort: ')

	events = []    # (start time, axis, sign) of each step

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

		att.start()
		for axis in ('roll', 'pitch'):
			for n in range(N_STEPS):
				sign = 1.0 if n % 2 == 0 else -1.0    # alternate direction

				# Settle on position setpoints
				hold(cf, Z, RECOVER)

				# Step, on attitude setpoints
				events.append((time.time(), axis, sign))
				roll = sign * STEP_DEG if axis == 'roll' else 0.0
				pitch = sign * STEP_DEG if axis == 'pitch' else 0.0
				t0 = time.time()
				while time.time() - t0 < STEP_S:
					cf.commander.send_setpoint(roll, pitch, 0, pwm)
					time.sleep(0.01)

		att.stop()
		time.sleep(0.3)
		land(cf, Z, 3.0)

	except KeyboardInterrupt:
		print('\naborted')
		cf.commander.send_stop_setpoint()


# ---- Fit each step ----

if not samples:
	sys.exit('no attitude samples')

d = np.array(samples)
print(f'\n{len(d)} samples, mean period {1000*np.diff(d[:,0]).mean():.1f} ms')


# First-order step response
def rise(t, a, amp, off):
	return off + amp * (1.0 - np.exp(-a * t))


fits = {'roll': [], 'pitch': []}    # good values of a for each axis

for t_start, axis, sign in events:
	# Samples during this step
	col = 1 if axis == 'roll' else 2
	in_step = (d[:, 0] >= t_start) & (d[:, 0] < t_start + STEP_S)
	seg = d[in_step]
	if len(seg) < 20:
		continue

	tt = seg[:, 0] - t_start
	yy = seg[:, col]

	try:
		popt, _ = curve_fit(rise, tt, yy, p0=[10.0, sign * STEP_DEG, yy[0]], maxfev=8000)
	except Exception:
		continue

	# How well the fit matches (R^2)
	a_hat = popt[0]
	pred = rise(tt, *popt)
	r2 = 1 - ((yy - pred)**2).sum() / ((yy - yy.mean())**2).sum()

	# Keep only good fits
	if a_hat > 0 and r2 > 0.7:
		fits[axis].append(a_hat)

	tau = 1/a_hat if a_hat > 0 else float('nan')
	print(f'  {axis:5s} {sign:+.0f}  a = {a_hat:6.2f}  '
	      f'tau = {tau:5.3f}s  R2 = {r2:5.3f}  n = {len(seg)}')


# ---- Results ----

print()
result = {}
for axis in ('roll', 'pitch'):
	if not fits[axis]:
		print(f'{axis}: no usable fits')
		continue
	a = np.array(fits[axis])
	result[axis] = a.mean()
	print(f'{axis:5s}  a = {a.mean():6.2f} +/- {a.std():5.2f}   '
	      f'tau = {1/a.mean():5.3f}s   ({len(a)} good fits)')

if 'pitch' in result and 'roll' in result:
	print(f'\nparams.hpp:')
	print(f'  a_theta = {result["pitch"]:.2f};')
	print(f'  a_phi   = {result["roll"]:.2f};')
