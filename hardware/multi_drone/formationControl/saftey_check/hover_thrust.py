"""Measure hover thrust command for one drone.

Hovers using the onboard position controller and logs what thrust the
firmware asks for. That value is the anchor for thrust_to_pwm.
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

URI       = sys.argv[1] if len(sys.argv) > 1 else 'radio://0/10/2M/E7E7E7E701'
HOVER_Z   = 0.50    # metres
CLIMB     = 3.0     # seconds to reach hover height
HOLD      = 8.0     # seconds of steady hover to average over
SETTLE    = 2.0     # seconds discarded at the start of the hold

CAN_FLY    = 1 << 3
IS_TUMBLED = 1 << 5
IS_LOCKED  = 1 << 6

samples = []
state   = {'z': None, 'vbat': None, 'info': None}


def log_cb(ts, msg, conf):
	state['z']      = msg['stateEstimate.z']
	state['vbat']   = msg['pm.vbat']
	samples.append((time.time(), msg['stabilizer.thrust'], msg['pm.vbat']))


def sup_cb(ts, msg, conf):
	state['info'] = msg['supervisor.info']


cflib.crtp.init_drivers()
print(f'Connecting to {URI}...')

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
	cf = scf.cf

	# --- preflight ---
	sup = LogConfig(name='Supervisor', period_in_ms=100)
	sup.add_variable('supervisor.info', 'uint16_t')
	cf.log.add_config(sup)
	sup.data_received_cb.add_callback(sup_cb)
	sup.start()

	t0 = time.time()
	while state['info'] is None and time.time() - t0 < 5.0:
		time.sleep(0.05)
	sup.stop()
	time.sleep(0.2)

	info = state['info']
	if info is None:
		sys.exit('no supervisor data')
	if (info & IS_LOCKED) or (info & IS_TUMBLED) or not (info & CAN_FLY):
		sys.exit('drone is locked/tumbled — power cycle it')
	print('preflight ok')

	# --- logging ---
	lg = LogConfig(name='Thrust', period_in_ms=50)
	lg.add_variable('stabilizer.thrust', 'float')
	lg.add_variable('stateEstimate.z', 'float')
	lg.add_variable('pm.vbat', 'float')
	cf.log.add_config(lg)
	lg.data_received_cb.add_callback(log_cb)
	lg.start()

	t0 = time.time()
	while state['vbat'] is None and time.time() - t0 < 3.0:
		time.sleep(0.05)
	print(f'battery {state["vbat"]:.2f} V')

	# --- estimator ---
	cf.param.set_value('kalman.resetEstimation', '1')
	time.sleep(0.2)
	cf.param.set_value('kalman.resetEstimation', '0')
	time.sleep(2.0)

	input(f'\nProp will spin, hovering at {HOVER_Z} m. Enter to arm, Ctrl-C to abort: ')

	try:
		cf.commander.send_setpoint(0, 0, 0, 0)   # unlock

		# climb
		steps = int(CLIMB / 0.05)
		for s in range(steps):
			z = HOVER_Z * (s + 1) / steps
			cf.commander.send_position_setpoint(0.0, 0.0, z, 0.0)
			time.sleep(0.05)

		# hold
		print(f'holding {HOLD:.0f}s...')
		hold_start = time.time()
		while time.time() - hold_start < HOLD:
			cf.commander.send_position_setpoint(0.0, 0.0, HOVER_Z, 0.0)
			time.sleep(0.05)
		hold_end = time.time()

		# land
		print('landing...')
		steps = int(2.5 / 0.05)
		for s in range(steps):
			z = HOVER_Z * (1.0 - (s + 1) / steps)
			cf.commander.send_position_setpoint(0.0, 0.0, max(z, 0.05), 0.0)
			time.sleep(0.05)
		cf.commander.send_stop_setpoint()

	except KeyboardInterrupt:
		print('\naborted')
		cf.commander.send_stop_setpoint()

	lg.stop()
	time.sleep(0.3)

# --- analysis ---
window = [s for s in samples
          if hold_start + SETTLE <= s[0] <= hold_end]

if len(window) < 10:
	sys.exit('not enough hover samples — did it reach height?')

thrust = np.array([s[1] for s in window])
vbat   = np.array([s[2] for s in window])

print(f'\nhover samples: {len(thrust)} over {hold_end - hold_start - SETTLE:.1f}s')
print(f'  thrust  mean {thrust.mean():8.0f}  sd {thrust.std():6.0f}  '
      f'min {thrust.min():.0f}  max {thrust.max():.0f}')
print(f'  vbat    start {vbat[0]:.2f} V  end {vbat[-1]:.2f} V')
print(f'\nHOVER_PWM = {thrust.mean():.0f}')
