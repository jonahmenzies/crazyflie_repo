# Measures the thrust channel: how much force each PWM value gives.
#
# Commands a series of PWM pulses around hover and measures the vertical
# acceleration each gives. Since F = m*(g + a_z), every pulse is a (pwm, force)
# point. Also measures hover PWM several times to see how much it drifts.
# The slope and hover PWM go into THRUST in flight/setpoint_map_pwm.py.
#
#     python3 safety_check/thrust_id.py [uri]      default is drone 1
#
# Run from formationControl/

import sys
import time

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from common import uri_from_args, preflight, reset_estimator, climb, hold, land

URI     = uri_from_args()
MASS    = 0.027        # kg
G       = 9.81         # m/s^2
Z       = 0.8          # hover height, high enough to sink during a pulse, m
PULSE_S = 0.40         # how long each test PWM is held, s
RECOVER = 3.0          # position hold between pulses, s
OFFSETS = [-6000, -4000, -2000, 0, 2000, 4000, 6000]    # PWM relative to hover
REPEATS = 3            # number of hover measurements

samples = []                                 # (time, z, vz) every 10 ms
state = {'thrust': None, 'vbat': None}       # latest slow log values


def on_fast(timestamp, data, logconf):
	samples.append((time.time(), data['stateEstimate.z'], data['stateEstimate.vz']))


def on_slow(timestamp, data, logconf):
	state['thrust'] = data['stabilizer.thrust']
	state['vbat'] = data['pm.vbat']


# ---- Fly the pulses ----

cflib.crtp.init_drivers()
print(f'Connecting to {URI}...')

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
	cf = scf.cf
	preflight(scf)

	# Height and vertical speed every 10 ms
	fast = LogConfig(name='Fast', period_in_ms=10)
	fast.add_variable('stateEstimate.z', 'float')
	fast.add_variable('stateEstimate.vz', 'float')
	cf.log.add_config(fast)
	fast.data_received_cb.add_callback(on_fast)

	# Thrust and battery every 100 ms
	slow = LogConfig(name='Slow', period_in_ms=100)
	slow.add_variable('stabilizer.thrust', 'float')
	slow.add_variable('pm.vbat', 'float')
	cf.log.add_config(slow)
	slow.data_received_cb.add_callback(on_slow)

	reset_estimator(cf)

	print(f'\n{len(OFFSETS)} thrust pulses at {Z} m, plus {REPEATS} hover '
	      f'measurements.')
	print('The drone will climb and sink noticeably on each pulse.')
	print('Needs ~2 m of ceiling and clear floor.')
	input('Enter to arm, Ctrl-C to abort: ')

	hovers = []    # (hover pwm, battery voltage) for each measurement
	events = []    # (start time, offset, pwm, battery voltage) for each pulse

	try:
		cf.commander.send_setpoint(0, 0, 0, 0)    # unlock
		slow.start()
		climb(cf, Z, 4.0)

		# ---- Hover PWM, measured REPEATS times ----
		for rep in range(REPEATS):
			hold(cf, Z, 2.0)

			# Average the firmware's thrust over 2 s of hover
			readings = []
			t0 = time.time()
			while time.time() - t0 < 2.0:
				cf.commander.send_position_setpoint(0, 0, Z, 0)
				if state['thrust'] is not None:
					readings.append(state['thrust'])
				time.sleep(0.05)

			h = float(np.mean(readings))
			hovers.append((h, state['vbat']))
			print(f'  hover {rep+1}: {h:8.0f}  vbat {state["vbat"]:.2f} V')

		hover_pwm = float(np.mean([h for h, _ in hovers]))
		print(f'\nusing hover_pwm {hover_pwm:.0f} as the pulse centre\n')

		# ---- Thrust pulses ----
		fast.start()
		for off in OFFSETS:
			hold(cf, Z, RECOVER)

			pwm = int(np.clip(hover_pwm + off, 0, 65535))
			events.append((time.time(), off, pwm, state['vbat']))
			t0 = time.time()
			while time.time() - t0 < PULSE_S:
				cf.commander.send_setpoint(0, 0, 0, pwm)
				time.sleep(0.01)

		fast.stop()
		slow.stop()
		time.sleep(0.3)

		hold(cf, Z, 1.0)
		land(cf, Z, 3.5)

	except KeyboardInterrupt:
		print('\naborted')
		cf.commander.send_stop_setpoint()


# ---- Hover repeatability ----

print('\n' + '=' * 62)

h = np.array([x[0] for x in hovers])
print(f'\nhover repeatability: mean {h.mean():.0f}  sd {h.std():.0f}  '
      f'spread {h.max()-h.min():.0f}  ({100*(h.max()-h.min())/h.mean():.1f}%)')

if not samples:
	sys.exit('\nno fast samples — cannot fit the curve')

d = np.array(samples)
print(f'{len(d)} samples, mean period {1000*np.diff(d[:,0]).mean():.1f} ms')

# ---- Force from each pulse ----

print(f'\n{"offset":>8} {"pwm":>7} {"a_z":>8} {"F":>9} {"F/hover":>9} '
      f'{"vbat":>6} {"n":>4}')

pts = []    # (pwm, force) for each usable pulse
for t_start, off, pwm, vbat in events:
	# Samples during the pulse, skipping the first 100 ms while the motors spin up
	in_pulse = (d[:, 0] >= t_start + 0.10) & (d[:, 0] < t_start + PULSE_S)
	seg = d[in_pulse]
	if len(seg) < 10:
		print(f'{off:8d} {pwm:7d}   too few samples')
		continue

	tt = seg[:, 0] - seg[0, 0]
	vz = seg[:, 2]

	# Slope of vz is the vertical acceleration
	a_z = np.polyfit(tt, vz, 1)[0]
	F = MASS * (G + a_z)

	pts.append((pwm, F))
	print(f'{off:+8d} {pwm:7d} {a_z:+8.2f} {F:9.5f} '
	      f'{F/(MASS*G):9.3f} {vbat:6.2f} {len(seg):4d}')

# ---- Straight line fit F = k*pwm + c ----

if len(pts) >= 3:
	p = np.array(pts)
	k, c = np.polyfit(p[:, 0], p[:, 1], 1)
	pred = k * p[:, 0] + c
	r2 = 1 - ((p[:,1]-pred)**2).sum() / ((p[:,1]-p[:,1].mean())**2).sum()

	print(f'\nlinear fit  F = {k:.3e} * pwm + {c:+.5f}   R2 = {r2:.4f}')
	print(f'  implied hover pwm  {(MASS*G - c)/k:.0f}')
	print(f'  measured hover pwm {h.mean():.0f}')

	# Compare with a model that assumes force is proportional to PWM
	print(f'\ncurrent model assumes F = hover_N * pwm / hover_pwm, '
	      f'i.e. slope {MASS*G/h.mean():.3e} through the origin')
	print(f'  measured slope is {k/(MASS*G/h.mean()):.2f}x that')
else:
	print('\nnot enough usable pulses to fit')
