"""Measure a_r. Steps yaw rate in attitude mode, fits the exponential rise
in gyro.z. Same first-order structure as roll and pitch: tau = 1/a_r.
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
STEP_DPS = 60.0     # commanded yaw rate, deg/s
STEP_S   = 0.5
RECOVER  = 2.5
N_STEPS  = 6

CAN_FLY    = 1 << 3
IS_TUMBLED = 1 << 5
IS_LOCKED  = 1 << 6

samples = []
sup     = {'info': None}
hover   = {'pwm': None}


def rate_cb(ts, msg, conf):
	samples.append((time.time(), msg['gyro.z']))


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

	rate = LogConfig(name='Rate', period_in_ms=10)
	rate.add_variable('gyro.z', 'float')
	cf.log.add_config(rate)
	rate.data_received_cb.add_callback(rate_cb)

	thr = LogConfig(name='Thr', period_in_ms=100)
	thr.add_variable('stabilizer.thrust', 'float')
	cf.log.add_config(thr)
	thr.data_received_cb.add_callback(thr_cb)

	cf.param.set_value('kalman.resetEstimation', '1')
	time.sleep(0.2)
	cf.param.set_value('kalman.resetEstimation', '0')
	time.sleep(2.0)

	print(f'\n{N_STEPS} yaw-rate steps at {STEP_DPS} deg/s.')
	print('The drone spins in place — keep props clear.')
	input('Enter to arm, Ctrl-C to abort: ')

	events = []

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

		rate.start()

		for n in range(N_STEPS):
			sign = 1.0 if n % 2 == 0 else -1.0

			t0 = time.time()
			while time.time() - t0 < RECOVER:
				cf.commander.send_position_setpoint(0, 0, Z, 0)
				time.sleep(0.05)

			events.append((time.time(), sign))
			t0 = time.time()
			while time.time() - t0 < STEP_S:
				cf.commander.send_setpoint(0, 0, sign * STEP_DPS, pwm)
				time.sleep(0.01)

		rate.stop(); time.sleep(0.3)

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
	sys.exit('no rate samples')

d = np.array(samples)
print(f'\n{len(d)} samples, mean period '
      f'{1000*np.diff(d[:,0]).mean():.1f} ms')


def rise(t, a, amp, off):
	return off + amp * (1.0 - np.exp(-a * t))


good = []
for t_start, sign in events:
	m = (d[:, 0] >= t_start) & (d[:, 0] < t_start + STEP_S)
	seg = d[m]
	if len(seg) < 20:
		continue

	tt = seg[:, 0] - t_start
	yy = seg[:, 1]

	try:
		popt, _ = curve_fit(rise, tt, yy,
		                    p0=[10.0, sign * STEP_DPS, yy[0]],
		                    maxfev=8000)
	except Exception:
		continue

	a_hat = popt[0]
	pred  = rise(tt, *popt)
	r2    = 1 - ((yy - pred)**2).sum() / ((yy - yy.mean())**2).sum()

	if a_hat > 0 and r2 > 0.7:
		good.append(a_hat)
	print(f'  yaw {sign:+.0f}  a = {a_hat:6.2f}  '
	      f'tau = {1/a_hat if a_hat > 0 else float("nan"):5.3f}s  '
	      f'R2 = {r2:5.3f}  n = {len(seg)}')

if good:
	a = np.array(good)
	print(f'\nyaw   a_r = {a.mean():6.2f} +/- {a.std():5.2f}   '
	      f'tau = {1/a.mean():5.3f}s   ({len(a)} good fits)')
	print(f'\nparams.hpp:\n  a_r = {a.mean():.2f};')
else:
	print('\nno usable fits')
