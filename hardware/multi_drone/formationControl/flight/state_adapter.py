import math
import time

import numpy as np
from cflib.crazyflie.log import LogConfig

# How often each drone sends its state, in ms (10 Hz)
LOG_PERIOD_MS = 100


# ---- One drone ----

# Holds the latest logged data for one drone.
# cflib calls _pos_cb and _att_cb in the background whenever a packet
# arrives, so these values are always the newest received.
class DroneState:

	def __init__(self, scf, index):
		self.scf = scf
		self.index = index    # which block of the state vector this drone fills

		# Position and velocity, m and m/s
		self.x = 0.0
		self.y = 0.0
		self.z = 0.0
		self.vx = 0.0
		self.vy = 0.0
		self.vz = 0.0

		# Attitude in degrees, yaw rate in degrees/s
		self.roll = 0.0
		self.pitch = 0.0
		self.yaw = 0.0
		self.rate_yaw = 0.0

		# Time the last packet of each kind arrived (0 = never)
		self.t_pos = 0.0
		self.t_att = 0.0

		# Log block 1: position and velocity
		self.log_pos = LogConfig(name=f'pos{index}', period_in_ms=LOG_PERIOD_MS)
		self.log_pos.add_variable('stateEstimate.x',  'float')
		self.log_pos.add_variable('stateEstimate.y',  'float')
		self.log_pos.add_variable('stateEstimate.z',  'float')
		self.log_pos.add_variable('stateEstimate.vx', 'float')
		self.log_pos.add_variable('stateEstimate.vy', 'float')
		self.log_pos.add_variable('stateEstimate.vz', 'float')

		# Log block 2: attitude and yaw rate
		self.log_att = LogConfig(name=f'att{index}', period_in_ms=LOG_PERIOD_MS)
		self.log_att.add_variable('stabilizer.roll',  'float')
		self.log_att.add_variable('stabilizer.pitch', 'float')
		self.log_att.add_variable('stabilizer.yaw',   'float')
		self.log_att.add_variable('gyro.z',           'float')

	# Start both log blocks on the drone
	def start(self):
		self.scf.cf.log.add_config(self.log_pos)
		self.log_pos.data_received_cb.add_callback(self._pos_cb)
		self.log_pos.start()

		self.scf.cf.log.add_config(self.log_att)
		self.log_att.data_received_cb.add_callback(self._att_cb)
		self.log_att.start()

	# Stop both log blocks and remove them from the drone
	def stop(self):
		for log in (self.log_pos, self.log_att):
			try:
				log.stop()
				self.scf.cf.log.delete_config(log)
			except Exception:
				pass

	# Called by cflib when a position packet arrives
	def _pos_cb(self, timestamp, data, logconf):
		self.x = data['stateEstimate.x']
		self.y = data['stateEstimate.y']
		self.z = data['stateEstimate.z']
		self.vx = data['stateEstimate.vx']
		self.vy = data['stateEstimate.vy']
		self.vz = data['stateEstimate.vz']
		self.t_pos = time.time()

	# Called by cflib when an attitude packet arrives
	def _att_cb(self, timestamp, data, logconf):
		self.roll = data['stabilizer.roll']
		self.pitch = data['stabilizer.pitch']
		self.yaw = data['stabilizer.yaw']
		self.rate_yaw = data['gyro.z']
		self.t_att = time.time()

	# Seconds since the older of the two log blocks last updated
	def age(self):
		now = time.time()
		return max(now - self.t_pos, now - self.t_att)

	# True once both log blocks have sent at least one packet
	def received(self):
		return self.t_pos > 0.0 and self.t_att > 0.0


# ---- Whole swarm ----

# Builds the full state vector from every drone.
# Each drone fills 10 elements, in the order the generator uses:
#     [x, vx, pitch, y, vy, roll, z, vz, yaw, yaw rate]
class StateAdapter:

	def __init__(self, swarm):
		self.drones = [DroneState(scf, i) for i, scf in enumerate(swarm.scf)]
		self.N = len(self.drones)    # number of drones
		self.n = 10 * self.N         # state vector length

	# Start logging on every drone and wait until they have all sent data
	def start(self, timeout=5.0):
		for d in self.drones:
			d.start()

		t0 = time.time()
		while time.time() - t0 < timeout:
			if all(d.received() for d in self.drones):
				return
			time.sleep(0.05)

		missing = [d.index for d in self.drones if not d.received()]
		raise RuntimeError(f'no log data from drones {missing}')

	# Stop logging on every drone
	def stop(self):
		for d in self.drones:
			d.stop()

	# Reorder the drones so list position i is formation slot i.
	# order[j] is the current position of the drone that becomes slot j.
	def reorder(self, order):
		self.drones = [self.drones[i] for i in order]
		for j, d in enumerate(self.drones):
			d.index = j

	# Current state vector. Raises if any drone's data is older than max_age.
	def read(self, max_age=1.0):
		# Check no drone has stopped sending data
		stale = []
		for d in self.drones:
			age = d.age()
			if age > max_age:
				stale.append(f'drone {d.index} {age:.2f}s')
		if stale:
			raise RuntimeError('stale log data: ' + ', '.join(stale))

		# Fill each drone's 10-element block, angles converted to radians
		x = np.zeros(self.n)
		for d in self.drones:
			b = d.index * 10
			x[b + 0] = d.x
			x[b + 1] = d.vx
			x[b + 2] = math.radians(d.pitch)
			x[b + 3] = d.y
			x[b + 4] = d.vy
			x[b + 5] = math.radians(d.roll)
			x[b + 6] = d.z
			x[b + 7] = d.vz
			x[b + 8] = math.radians(d.yaw)
			x[b + 9] = math.radians(d.rate_yaw)
		return x

	# (N, 3) array of just x, y, z for each drone
	def positions(self):
		return np.array([[d.x, d.y, d.z] for d in self.drones])

	# Print a table of every drone's current state
	def report(self):
		print(f'{"i":>2}  {"x":>7} {"y":>7} {"z":>7}  '
		      f'{"vx":>6} {"vy":>6} {"vz":>6}  '
		      f'{"roll":>6} {"pitch":>6} {"yaw":>6}  {"age":>5}')
		for d in self.drones:
			print(f'{d.index:2d}  {d.x:+7.3f} {d.y:+7.3f} {d.z:+7.3f}  '
			      f'{d.vx:+6.2f} {d.vy:+6.2f} {d.vz:+6.2f}  '
			      f'{d.roll:+6.1f} {d.pitch:+6.1f} {d.yaw:+6.1f}  '
			      f'{d.age():5.2f}')


# Self test: python3 flight/state_adapter.py (run from formationControl/, needs the drones)
# Prints every drone's state once a second until Ctrl-C
if __name__ == '__main__':
	from swarm import Swarm

	with Swarm() as sw:
		sw.reset_estimators()
		adapter = StateAdapter(sw)
		adapter.start()
		try:
			while True:
				adapter.report()
				print()
				time.sleep(1.0)
		except KeyboardInterrupt:
			pass
		finally:
			adapter.stop()
