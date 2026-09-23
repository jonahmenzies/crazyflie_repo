"""Characterise the thrust channel properly.

Rather than assuming F is linear about one hover point, this commands a
series of PWM values and measures the vertical acceleration each produces.
Since zdd = F/m - g, every pulse gives a direct (pwm, force) pair.

Also repeats the hover measurement several times to quantify how much the
"constant" actually moves within one session.
"""

import sys
import time
import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.console import Console
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


def _tolerant_incoming(self, packet):
	try:
		text = packet.data.decode('UTF-8', errors='replace')
	except Exception:
		return
	self.receivedChar.call(text)


Console._incoming = _tolerant_incoming

URI     = sys.argv[1] if len(sys.argv) > 1 else 'radio://0/10/2M/E7E7E7E701'
MASS    = 0.027
G       = 9.81
Z       = 0.8          # pulse from high enough to have room to sink
PULSE_S = 0.40         # how long to hold each test PWM
RECOVER = 3.0          # position hold between pulses
OFFSETS = [-6000, -4000, -2000, 0, 2000, 4000, 6000]
REPEATS = 3            # hover re-measurements, to see session drift

CAN_FLY    = 1 << 3
IS_TUMBLED = 1 << 5
IS_LOCKED  = 1 << 6

samples = []
state   = {'thrust': None, 'vbat': None, 'z': None}
sup     = {'info': None}


def fast_cb(ts, msg, conf):
	samples.append((time.time(),
	                msg['stateEstimate.z'], msg['stateEstimate.vz']))


def slow_cb(ts, msg, conf):
	state['thrust'] = msg['stabilizer.thrust']
	state['vbat']   = msg['pm.vbat']


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

	fast = LogConfig(name='Fast', period_in_ms=10)
	fast.add_variable('stateEstimate.z',  'float')
	fast.add_variable('stateEstimate.vz', 'float')
	cf.log.add_config(fast)
	fast.data_received_cb.add_callback(fast_cb)

	slow = LogConfig(name='Slow', period_in_ms=100)
	slow.add_variable('stabilizer.thrust', 'float')
	slow.add_variable('pm.vbat', 'float')
	cf.log.add_config(slow)
	slow.data_received_cb.add_callback(slow_cb)

	cf.param.set_value('kalman.resetEstimation', '1')
	time.sleep(0.2)
	cf.param.set_value('kalman.resetEstimation', '0')
	time.sleep(2.0)

	print(f'\n{len(OFFSETS)} thrust pulses at {Z} m, plus {REPEATS} hover '
	      f'measurements.')
	print('The drone will climb and sink noticeably on each pulse.')
	print('Needs ~2 m of ceiling and clear floor.')
	input('Enter to arm, Ctrl-C to abort: ')

	hovers = []
	events = []

	def hold(seconds):
		t0 = time.time()
		while time.time() - t0 < seconds:
			cf.commander.send_position_setpoint(0, 0, Z, 0)
			time.sleep(0.05)

	try:
		cf.commander.send_setpoint(0, 0, 0, 0)
		slow.start()

		for s in range(int(4.0 / 0.05)):
			cf.commander.send_position_setpoint(
				0, 0, Z * (s + 1) / (4.0 / 0.05), 0)
			time.sleep(0.05)

		# --- repeated hover measurement ---
		for rep in range(REPEATS):
			hold(2.0)
			acc = []
			t0 = time.time()
			while time.time() - t0 < 2.0:
				cf.commander.send_position_setpoint(0, 0, Z, 0)
				if state['thrust'] is not None:
					acc.append(state['thrust'])
				time.sleep(0.05)
			h = float(np.mean(acc))
			hovers.append((h, state['vbat']))
			print(f'  hover {rep+1}: {h:8.0f}  vbat {state["vbat"]:.2f} V')

		hover_pwm = float(np.mean([h for h, _ in hovers]))
		print(f'\nusing hover_pwm {hover_pwm:.0f} as the pulse centre\n')

		fast.start()

		# --- thrust pulses ---
		for off in OFFSETS:
			hold(RECOVER)
			pwm = int(np.clip(hover_pwm + off, 0, 65535))
			events.append((time.time(), off, pwm, state['vbat']))
			t0 = time.time()
			while time.time() - t0 < PULSE_S:
				cf.commander.send_setpoint(0, 0, 0, pwm)
				time.sleep(0.01)

		fast.stop(); slow.stop(); time.sleep(0.3)

		hold(1.0)
		for s in range(int(3.5 / 0.05)):
			cf.commander.send_position_setpoint(
				0, 0, max(Z * (1 - (s + 1) / (3.5 / 0.05)), 0.05), 0)
			time.sleep(0.05)
		cf.commander.send_stop_setpoint()

	except KeyboardInterrupt:
		print('\naborted')
		cf.commander.send_stop_setpoint()

# --- analysis ---
print('\n' + '=' * 62)

h = np.array([x[0] for x in hovers])
print(f'\nhover repeatability: mean {h.mean():.0f}  sd {h.std():.0f}  '
      f'spread {h.max()-h.min():.0f}  ({100*(h.max()-h.min())/h.mean():.1f}%)')

if not samples:
	sys.exit('\nno fast samples — cannot fit the curve')

d = np.array(samples)
print(f'{len(d)} samples, mean period {1000*np.diff(d[:,0]).mean():.1f} ms')

print(f'\n{"offset":>8} {"pwm":>7} {"a_z":>8} {"F":>9} {"F/hover":>9} '
      f'{"vbat":>6} {"n":>4}')

pts = []
for t_start, off, pwm, vbat in events:
	# skip the first 100 ms, so the motors have spun up
	m = (d[:, 0] >= t_start + 0.10) & (d[:, 0] < t_start + PULSE_S)
	seg = d[m]
	if len(seg) < 10:
		print(f'{off:8d} {pwm:7d}   too few samples')
		continue

	tt = seg[:, 0] - seg[0, 0]
	vz = seg[:, 2]

	# slope of vz is vertical acceleration
	a_z = np.polyfit(tt, vz, 1)[0]
	F   = MASS * (G + a_z)

	pts.append((pwm, F))
	print(f'{off:+8d} {pwm:7d} {a_z:+8.2f} {F:9.5f} '
	      f'{F/(MASS*G):9.3f} {vbat:6.2f} {len(seg):4d}')

if len(pts) >= 3:
	p = np.array(pts)
	# linear fit F = k*pwm + c
	k, c = np.polyfit(p[:, 0], p[:, 1], 1)
	pred = k * p[:, 0] + c
	r2   = 1 - ((p[:,1]-pred)**2).sum() / ((p[:,1]-p[:,1].mean())**2).sum()

	print(f'\nlinear fit  F = {k:.3e} * pwm + {c:+.5f}   R2 = {r2:.4f}')
	print(f'  implied hover pwm  {(MASS*G - c)/k:.0f}')
	print(f'  measured hover pwm {h.mean():.0f}')

	# what the current linear-about-hover model assumes
	print(f'\ncurrent model assumes F = hover_N * pwm / hover_pwm, '
	      f'i.e. slope {MASS*G/h.mean():.3e} through the origin')
	print(f'  measured slope is {k/(MASS*G/h.mean()):.2f}x that')
else:
	print('\nnot enough usable pulses to fit')
