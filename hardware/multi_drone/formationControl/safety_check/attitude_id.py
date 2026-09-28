"""Measure a_theta and a_phi properly.

Attitude mode, real steps, 10 ms logging, fit the exponential rise
directly rather than differentiating. The drone hovers on position
setpoints between steps so it stays put.

    theta(t) = theta_cmd * (1 - exp(-a*t))    ->    tau = 1/a
"""

import sys
import time
import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.console import Console
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from scipy.optimize import curve_fit


def _tolerant_incoming(self, packet):
	try:
		text = packet.data.decode('UTF-8', errors='replace')
	except Exception:
		return
	self.receivedChar.call(text)


Console._incoming = _tolerant_incoming

URI      = sys.argv[1] if len(sys.argv) > 1 else 'radio://0/10/2M/E7E7E7E701'
Z        = 0.6
STEP_DEG = 8.0      # commanded lean, degrees. Big enough to see clearly.
STEP_S   = 0.5      # how long to hold each step
RECOVER  = 2.5      # position-hold between steps, to stop drift
N_STEPS  = 4        # steps per axis

CAN_FLY    = 1 << 3
IS_TUMBLED = 1 << 5
IS_LOCKED  = 1 << 6

samples = []
sup     = {'info': None}
hover   = {'pwm': None}


def att_cb(ts, msg, conf):
	samples.append((time.time(),
	                msg['stabilizer.roll'], msg['stabilizer.pitch']))


def thr_cb(ts, msg, conf):
	hover['pwm'] = msg['stabilizer.thrust']


def sup_cb(ts, msg, conf):
	sup['info'] = msg['supervisor.info']


cflib.crtp.init_drivers()
print(f'Connecting to {URI}...')

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
	cf = scf.cf

	lg = LogConfig(name='Sup', period_in_ms=100)
	lg.add_variable('supervisor.info', 'uint16_t')
	cf.log.add_config(lg)
	lg.data_received_cb.add_callback(sup_cb)
	lg.start()
	t0 = time.time()
	while sup['info'] is None and time.time() - t0 < 5.0:
		time.sleep(0.05)
	lg.stop(); time.sleep(0.2)

	info = sup['info']
	if info is None:
		sys.exit('no supervisor data')
	if (info & IS_LOCKED) or (info & IS_TUMBLED) or not (info & CAN_FLY):
		sys.exit('drone is locked/tumbled — power cycle it')
	print('preflight ok')

	# Attitude at 10 ms. Two variables only, to keep the packet small
	# enough to actually arrive at that rate.
	att = LogConfig(name='Att', period_in_ms=10)
	att.add_variable('stabilizer.roll',  'float')
	att.add_variable('stabilizer.pitch', 'float')
	cf.log.add_config(att)
	att.data_received_cb.add_callback(att_cb)

	thr = LogConfig(name='Thr', period_in_ms=100)
	thr.add_variable('stabilizer.thrust', 'float')
	cf.log.add_config(thr)
	thr.data_received_cb.add_callback(thr_cb)

	cf.param.set_value('kalman.resetEstimation', '1')
	time.sleep(0.2)
	cf.param.set_value('kalman.resetEstimation', '0')
	time.sleep(2.0)

	print(f'\n{N_STEPS} steps per axis at {STEP_DEG} degrees.')
	print('The drone WILL lurch on each step. Needs ~1.5 m clear space.')
	input('Enter to arm, Ctrl-C to abort: ')

	events = []      # (t_start, axis, sign)

	try:
		cf.commander.send_setpoint(0, 0, 0, 0)

		for s in range(int(3.0 / 0.05)):
			cf.commander.send_position_setpoint(
				0, 0, Z * (s + 1) / (3.0 / 0.05), 0)
			time.sleep(0.05)

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
				sign = 1.0 if n % 2 == 0 else -1.0

				# recover on position setpoints
				t0 = time.time()
				while time.time() - t0 < RECOVER:
					cf.commander.send_position_setpoint(0, 0, Z, 0)
					time.sleep(0.05)

				# step, on attitude
				events.append((time.time(), axis, sign))
				t0 = time.time()
				while time.time() - t0 < STEP_S:
					r = sign * STEP_DEG if axis == 'roll'  else 0.0
					p = sign * STEP_DEG if axis == 'pitch' else 0.0
					cf.commander.send_setpoint(r, p, 0, pwm)
					time.sleep(0.01)

		att.stop(); time.sleep(0.3)

		for s in range(int(3.0 / 0.05)):
			cf.commander.send_position_setpoint(
				0, 0, max(Z * (1 - (s + 1) / (3.0 / 0.05)), 0.05), 0)
			time.sleep(0.05)
		cf.commander.send_stop_setpoint()

	except KeyboardInterrupt:
		print('\naborted')
		cf.commander.send_stop_setpoint()

# --- fit ---
if not samples:
	sys.exit('no attitude samples')

d = np.array(samples)
print(f'\n{len(d)} samples, mean period '
      f'{1000*np.diff(d[:,0]).mean():.1f} ms')


def rise(t, a, amp, off):
	return off + amp * (1.0 - np.exp(-a * t))


fits = {'roll': [], 'pitch': []}

for t_start, axis, sign in events:
	col = 1 if axis == 'roll' else 2
	m = (d[:, 0] >= t_start) & (d[:, 0] < t_start + STEP_S)
	seg = d[m]
	if len(seg) < 20:
		continue

	tt = seg[:, 0] - t_start
	yy = seg[:, col]

	try:
		popt, _ = curve_fit(rise, tt, yy,
		                    p0=[10.0, sign * STEP_DEG, yy[0]],
		                    maxfev=8000)
	except Exception:
		continue

	a_hat = popt[0]
	pred  = rise(tt, *popt)
	r2    = 1 - ((yy - pred)**2).sum() / ((yy - yy.mean())**2).sum()

	if a_hat > 0 and r2 > 0.7:
		fits[axis].append((a_hat, r2, len(seg)))
	print(f'  {axis:5s} {sign:+.0f}  a = {a_hat:6.2f}  '
	      f'tau = {1/a_hat if a_hat > 0 else float("nan"):5.3f}s  '
	      f'R2 = {r2:5.3f}  n = {len(seg)}')

print()
out = {}
for axis in ('roll', 'pitch'):
	if not fits[axis]:
		print(f'{axis}: no usable fits')
		continue
	a = np.array([f[0] for f in fits[axis]])
	out[axis] = a.mean()
	print(f'{axis:5s}  a = {a.mean():6.2f} +/- {a.std():5.2f}   '
	      f'tau = {1/a.mean():5.3f}s   ({len(a)} good fits)')

if 'pitch' in out and 'roll' in out:
	print(f'\nparams.hpp:')
	print(f'  a_theta = {out["pitch"]:.2f};')
	print(f'  a_phi   = {out["roll"]:.2f};')
