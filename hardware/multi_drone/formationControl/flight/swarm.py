import time
import logging
import yaml
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.console import Console
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


# ---- Silence cflib noise on connect ----

# The drone prints debug text while booting. cflib crashes trying to decode
# half a packet of it, which is harmless but prints a traceback every connect.
# This replacement decodes it without crashing.
def _tolerant_incoming(self, packet):
	try:
		console_text = packet.data.decode('UTF-8', errors='replace')
	except Exception:
		return
	self.receivedChar.call(console_text)


Console._incoming = _tolerant_incoming
logging.getLogger('cflib').setLevel(logging.CRITICAL)


# ---- Constants ----

# Bits in the supervisor.info log variable
CAN_FLY    = 1 << 3
IS_TUMBLED = 1 << 5
IS_LOCKED  = 1 << 6

# Minimum resting battery voltage. The cell sags 0.4-0.5 V under load.
VBAT_MIN = 3.83


# Drone number painted on the airframe, taken from the end of its URI
# e.g. radio://0/60/2M/E7E7E7E706 -> 6
def label(uri):
	return int(uri[-2:])


# ---- Swarm ----

# Connects to every drone in config/uris.yaml. Use it as:
#     with Swarm() as sw:
#         ...
# The links are closed and the motors stopped when the with block ends.
class Swarm:

	def __init__(self, config='config/uris.yaml', cache='./cache', settle=3.0):
		with open(config) as f:
			cfg = yaml.safe_load(f)

		self.uris = cfg['uris']                                   # every drone we try to connect to
		self.required = cfg.get('swarm', {}).get('required', len(self.uris))  # how many must connect
		self.cache = cache      # cflib's cache of each drone's log/param tables
		self.settle = settle    # seconds to wait for drones to finish booting

		self.scf = []     # open connections
		self.live = []    # URIs of the drones that connected, same order as scf

	# Runs at the start of the with block
	def __enter__(self):
		cflib.crtp.init_drivers()

		# Give just-powered drones time to finish booting
		if self.settle > 0:
			time.sleep(self.settle)

		# Try to connect to each drone, skipping any that don't respond
		for uri in self.uris:
			scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache=self.cache))
			try:
				scf.open_link()
			except Exception as ex:
				print(f'  {uri}  no response ({type(ex).__name__})')
				continue

			self.scf.append(scf)
			self.live.append(uri)
			print(f'  {uri}  drone {label(uri)}  (list {len(self.live) - 1})')

		# Stop if not enough drones connected
		if len(self.scf) != self.required:
			self.close()
			raise RuntimeError(f'need {self.required} drones, got {len(self.scf)}')

		self.N = len(self.scf)
		return self

	# Runs at the end of the with block, even after an error
	def __exit__(self, exc_type, exc, tb):
		self.stop_all()
		self.close()
		return False

	# Drone numbers in the current list order (this follows any reorder)
	def labels(self):
		return [label(uri) for uri in self.live]

	# Close every connection
	def close(self):
		for scf in self.scf:
			try:
				scf.close_link()
			except Exception:
				pass
		self.scf = []

	# Read one log variable from one drone. Returns None if nothing arrives.
	def _read_once(self, scf, var, vtype, timeout=5.0):
		result = {'value': None}    # dict so the callback can write into it

		def on_data(timestamp, data, logconf):
			result['value'] = data[var]

		try:
			# Start logging the variable
			log = LogConfig(name='Once', period_in_ms=100)
			log.add_variable(var, vtype)
			scf.cf.log.add_config(log)
			log.data_received_cb.add_callback(on_data)
			log.start()

			# Wait for the first value or the timeout
			start = time.time()
			while result['value'] is None and time.time() - start < timeout:
				time.sleep(0.05)

			# Stop logging and remove the config from the drone
			log.stop()
			time.sleep(0.2)
			scf.cf.log.delete_config(log)
		except Exception:
			pass

		return result['value']

	# Refuse to fly on low batteries. Returns {uri: voltage} for the log.
	def battery_check(self, minimum=VBAT_MIN):
		print('\nBattery check:')
		volts = {}
		ok = True

		for scf, uri in zip(self.scf, self.live):
			v = self._read_once(scf, 'pm.vbat', 'float')
			volts[uri] = v

			if v is None:
				print(f'  drone {label(uri):2d}  NO DATA')
				ok = False
			elif v < minimum:
				print(f'  drone {label(uri):2d}  {v:.2f} V  LOW — charge before flying')
				ok = False
			else:
				print(f'  drone {label(uri):2d}  {v:.2f} V  ok')

		if not ok:
			raise RuntimeError(f'battery check failed (minimum {minimum:.2f} V)')
		print('All batteries ok.\n')
		return volts

	# Refuse to fly if any drone is locked from a previous crash or stop.
	# A locked drone needs a power cycle.
	def preflight_check(self, timeout=5.0):
		print('\nPreflight check:')
		ok = True

		for scf, uri in zip(self.scf, self.live):
			info = self._read_once(scf, 'supervisor.info', 'uint16_t', timeout)

			if info is None:
				print(f'  drone {label(uri):2d}  NO SUPERVISOR DATA')
				ok = False
				continue

			# Check each status bit
			problems = []
			if info & IS_LOCKED:
				problems.append('LOCKED')
			if info & IS_TUMBLED:
				problems.append('TUMBLED')
			if not (info & CAN_FLY):
				problems.append('CANNOT FLY')

			if problems:
				print(f'  drone {label(uri):2d}  {", ".join(problems)} '
				      f'— power cycle required')
				ok = False
			else:
				print(f'  drone {label(uri):2d}  ready')

		if not ok:
			raise RuntimeError('preflight check failed — power cycle affected drones')
		print('All drones ready.\n')

	# Reset each drone's position estimator, then wait for it to settle
	def reset_estimators(self):
		for scf in self.scf:
			scf.cf.param.set_value('kalman.resetEstimation', '1')
		time.sleep(0.2)

		for scf in self.scf:
			scf.cf.param.set_value('kalman.resetEstimation', '0')
		time.sleep(2.0)

	# Cut the motors on every drone
	def stop_all(self):
		for scf in self.scf:
			try:
				scf.cf.commander.send_stop_setpoint()
			except Exception:
				pass


# Self test: python3 flight/swarm.py (run from formationControl/, needs the drones)
if __name__ == '__main__':
	print('Connecting swarm...')
	with Swarm() as swarm:
		print(f'\nConnected: {swarm.N}/{swarm.required} drones')
		for i, uri in enumerate(swarm.live):
			print(f'  list {i}: drone {label(uri)}  {uri}')

		swarm.preflight_check()
		swarm.battery_check()

		print('Holding link open for 3s, then disconnecting...')
		time.sleep(3.0)
	print('Disconnected cleanly.')
