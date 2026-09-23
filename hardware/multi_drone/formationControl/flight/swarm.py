import time
import logging
import yaml
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.console import Console
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


# --- silence cflib boot-console noise -------------------------------------
# The Crazyflie streams debug text on the console port during boot. cflib
# decodes it as strict UTF-8; catching a partial packet mid-boot raises
# UnicodeDecodeError inside the callback thread. Harmless, but it dumps a
# traceback on every connect. Decode leniently instead.
def _tolerant_incoming(self, packet):
	try:
		console_text = packet.data.decode('UTF-8', errors='replace')
	except Exception:
		return
	self.receivedChar.call(console_text)


Console._incoming = _tolerant_incoming
logging.getLogger('cflib').setLevel(logging.CRITICAL)
# --------------------------------------------------------------------------

# supervisor.info bit positions
CAN_FLY    = 1 << 3
IS_TUMBLED = 1 << 5
IS_LOCKED  = 1 << 6

# Resting voltage. Under motor load a 1S cell sags 0.4-0.5 V, so 4.00 V at
# rest is already close to the point where results start degrading.
VBAT_MIN = 3.83


def label(uri):
	"""Physical drone number from a URI.

	radio://0/60/2M/E7E7E7E706 -> 6,  radio://1/100/2M/E7E7E7E710 -> 10.

	This is the number painted on the airframe, not a list position or a
	formation slot. Used so logs and console output can name the actual
	hardware after the Hungarian assignment has shuffled everything.
	"""
	return int(uri[-2:])


class Swarm:
	def __init__(self, config='config/uris.yaml', cache='./cache', settle=3.0):
		with open(config) as f:
			cfg = yaml.safe_load(f)
		self.uris     = cfg['uris']
		self.required = cfg.get('swarm', {}).get('required', len(self.uris))
		self.cache    = cache
		self.settle   = settle
		self.scf  = []
		self.live = []

	def __enter__(self):
		cflib.crtp.init_drivers()

		# Let any just-powered drones finish booting before opening links,
		# so we aren't reading half-written console output.
		if self.settle > 0:
			time.sleep(self.settle)

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

		if len(self.scf) != self.required:
			self.close()
			raise RuntimeError(f'need {self.required} drones, got {len(self.scf)}')

		self.N = len(self.scf)
		return self

	def __exit__(self, exc_type, exc, tb):
		self.stop_all()
		self.close()
		return False

	def labels(self):
		"""Physical drone numbers, in current list order.

		Call after any reorder — flight_loop permutes self.live so that
		list position i is formation slot i, so this tracks the reorder.
		"""
		return [label(uri) for uri in self.live]

	def close(self):
		for scf in self.scf:
			try:
				scf.close_link()
			except Exception:
				pass
		self.scf = []

	def _read_once(self, scf, var, vtype, timeout=5.0):
		"""Read one log variable once. Returns None on failure."""
		state = {'v': None}

		def cb(ts, msg, conf, _s=state):
			_s['v'] = msg[var]

		try:
			lg = LogConfig(name='Once', period_in_ms=100)
			lg.add_variable(var, vtype)
			scf.cf.log.add_config(lg)
			lg.data_received_cb.add_callback(cb)
			lg.start()

			start = time.time()
			while state['v'] is None and time.time() - start < timeout:
				time.sleep(0.05)

			lg.stop()
			time.sleep(0.2)
			scf.cf.log.delete_config(lg)
		except Exception:
			pass

		return state['v']

	def battery_check(self, minimum=VBAT_MIN):
		"""Refuse to fly on low packs.

		Thrust per PWM count falls as the cell sags, and there is no
		altitude feedback in the model to absorb it, so a low battery shows
		up directly as vertical tracking error. Returns the per-drone
		voltages so they can be recorded with the flight.
		"""
		print('\nBattery check:')
		volts = {}
		ok = True

		for i, scf in enumerate(self.scf):
			uri = self.live[i]
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

	def preflight_check(self, timeout=5.0):
		"""Refuse to fly if any drone is latched from a previous stop.

		After an emergency stop or a tumble the supervisor locks the drone
		until power cycle. Ask each drone directly rather than tracking
		reboots by hand.
		"""
		print('\nPreflight check:')
		ok = True

		for i, scf in enumerate(self.scf):
			uri = self.live[i]
			info = self._read_once(scf, 'supervisor.info', 'uint16_t', timeout)

			if info is None:
				print(f'  drone {label(uri):2d}  NO SUPERVISOR DATA')
				ok = False
				continue

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

	def reset_estimators(self):
		for scf in self.scf:
			scf.cf.param.set_value('kalman.resetEstimation', '1')
		time.sleep(0.2)
		for scf in self.scf:
			scf.cf.param.set_value('kalman.resetEstimation', '0')
		time.sleep(2.0)

	def stop_all(self):
		for scf in self.scf:
			try:
				scf.cf.commander.send_stop_setpoint()
			except Exception:
				pass


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
