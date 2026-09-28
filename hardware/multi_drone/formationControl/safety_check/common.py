# Shared setup for the safety_check scripts. Run them from formationControl/
# Importing this also applies swarm.py's fix for cflib's console noise.

import sys
import time

import yaml
from cflib.crazyflie.log import LogConfig

sys.path.insert(0, 'flight')
from swarm import label, CAN_FLY, IS_TUMBLED, IS_LOCKED

DEFAULT_URI = 'radio://0/10/2M/E7E7E7E701'    # drone 1
SEND_DT = 0.05                                 # time between position setpoints, s


# Drones listed in the config file
def load_uris(config='config/uris.yaml'):
	with open(config) as f:
		return yaml.safe_load(f)['uris']


# Drone URI from the command line, or drone 1 if none given
def uri_from_args():
	return sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URI


# Read one log variable from one drone. Returns None if nothing arrives.
def read_once(scf, var, vtype, timeout=5.0):
	result = {'value': None}    # dict so the callback can write into it

	def on_data(timestamp, data, logconf):
		result['value'] = data[var]

	try:
		log = LogConfig(name='Once', period_in_ms=100)
		log.add_variable(var, vtype)
		scf.cf.log.add_config(log)
		log.data_received_cb.add_callback(on_data)
		log.start()

		start = time.time()
		while result['value'] is None and time.time() - start < timeout:
			time.sleep(0.05)

		log.stop()
		time.sleep(0.2)
		scf.cf.log.delete_config(log)
	except Exception:
		pass

	return result['value']


# Exit if the drone is locked or tumbled from a previous crash
def preflight(scf):
	info = read_once(scf, 'supervisor.info', 'uint16_t')
	if info is None:
		sys.exit('no supervisor data')
	if (info & IS_LOCKED) or (info & IS_TUMBLED) or not (info & CAN_FLY):
		sys.exit('drone is locked/tumbled — power cycle it')
	print('preflight ok')


# Reset the position estimator and wait for it to settle
def reset_estimator(cf):
	cf.param.set_value('kalman.resetEstimation', '1')
	time.sleep(0.2)
	cf.param.set_value('kalman.resetEstimation', '0')
	time.sleep(2.0)


# Climb straight up from the ground to height z over the given time
def climb(cf, z, seconds):
	steps = int(seconds / SEND_DT)
	for s in range(steps):
		cf.commander.send_position_setpoint(0, 0, z * (s + 1) / steps, 0)
		time.sleep(SEND_DT)


# Hover at height z for the given time
def hold(cf, z, seconds):
	t0 = time.time()
	while time.time() - t0 < seconds:
		cf.commander.send_position_setpoint(0, 0, z, 0)
		time.sleep(SEND_DT)


# Descend from height z over the given time, then cut the motors
def land(cf, z, seconds):
	steps = int(seconds / SEND_DT)
	for s in range(steps):
		cf.commander.send_position_setpoint(0, 0, max(z * (1 - (s + 1) / steps), 0.05), 0)
		time.sleep(SEND_DT)
	cf.commander.send_stop_setpoint()
