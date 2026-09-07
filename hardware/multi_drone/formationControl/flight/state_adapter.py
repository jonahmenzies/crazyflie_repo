import math
import time

import numpy as np
from cflib.crazyflie.log import LogConfig


class DroneState:
	"""Latest log data for one drone.

	cflib calls the callbacks on its own thread whenever a packet lands,
	so these attributes are always the most recent values received.
	"""

	def __init__(self, scf, index):
		self.scf   = scf
		self.index = index

		self.x  = 0.0; self.y  = 0.0; self.z  = 0.0
		self.vx = 0.0; self.vy = 0.0; self.vz = 0.0
		self.roll = 0.0; self.pitch = 0.0; self.yaw = 0.0   # degrees
		self.rate_yaw = 0.0                                  # degrees/s

		self.t_pos = 0.0    # wall clock of last packet, for staleness
		self.t_att = 0.0

		self.log_pos = LogConfig(name=f'pos{index}', period_in_ms=50)
		self.log_pos.add_variable('stateEstimate.x',  'float')
		self.log_pos.add_variable('stateEstimate.y',  'float')
		self.log_pos.add_variable('stateEstimate.z',  'float')
		self.log_pos.add_variable('stateEstimate.vx', 'float')
		self.log_pos.add_variable('stateEstimate.vy', 'float')
		self.log_pos.add_variable('stateEstimate.vz', 'float')

		self.log_att = LogConfig(name=f'att{index}', period_in_ms=50)
		self.log_att.add_variable('stabilizer.roll',  'float')
		self.log_att.add_variable('stabilizer.pitch', 'float')
		self.log_att.add_variable('stabilizer.yaw',   'float')
		self.log_att.add_variable('gyro.z',           'float')

	def start(self):
		self.scf.cf.log.add_config(self.log_pos)
		self.log_pos.data_received_cb.add_callback(self._pos_cb)
		self.log_pos.start()

		self.scf.cf.log.add_config(self.log_att)
		self.log_att.data_received_cb.add_callback(self._att_cb)
		self.log_att.start()

	def stop(self):
		for log in (self.log_pos, self.log_att):
			try:
				log.stop()
				self.scf.cf.log.delete_config(log)
			except Exception:
				pass

	def _pos_cb(self, ts, msg, conf):
		self.x  = msg['stateEstimate.x']
		self.y  = msg['stateEstimate.y']
		self.z  = msg['stateEstimate.z']
		self.vx = msg['stateEstimate.vx']
		self.vy = msg['stateEstimate.vy']
		self.vz = msg['stateEstimate.vz']
		self.t_pos = time.time()

	def _att_cb(self, ts, msg, conf):
		self.roll     = msg['stabilizer.roll']
		self.pitch    = msg['stabilizer.pitch']
		self.yaw      = msg['stabilizer.yaw']
		self.rate_yaw = msg['gyro.z']
		self.t_att = time.time()

	def age(self):
		"""Seconds since the oldest of the two log streams last updated."""
		now = time.time()
		return max(now - self.t_pos, now - self.t_att)

	def received(self):
		return self.t_pos > 0.0 and self.t_att > 0.0


class StateAdapter:
	"""Assembles all drones into the 50x1 state vector.

	Element order per drone matches params::setDroneState:
	    [x, xd, theta, y, yd, phi, z, zd, psi, r]
	"""

	def __init__(self, swarm):
		self.drones = [DroneState(scf, i) for i, scf in enumerate(swarm.scf)]
		self.N = len(self.drones)
		self.n = 10 * self.N

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

	def stop(self):
		for d in self.drones:
			d.stop()

	def read(self, max_age=0.5):
		"""Snapshot of the current state. Raises if any drone is stale."""
		stale = [(d.index, d.age()) for d in self.drones if d.age() > max_age]
		if stale:
			raise RuntimeError('stale log data: ' +
				', '.join(f'drone {i} {a:.2f}s' for i, a in stale))

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

	def positions(self):
		"""(N, 3) of xyz only. For assign.py and pre-arm checks."""
		return np.array([[d.x, d.y, d.z] for d in self.drones])

	def report(self):
		print(f'{"i":>2}  {"x":>7} {"y":>7} {"z":>7}  '
		      f'{"vx":>6} {"vy":>6} {"vz":>6}  '
		      f'{"roll":>6} {"pitch":>6} {"yaw":>6}  {"age":>5}')
		for d in self.drones:
			print(f'{d.index:2d}  {d.x:+7.3f} {d.y:+7.3f} {d.z:+7.3f}  '
			      f'{d.vx:+6.2f} {d.vy:+6.2f} {d.vz:+6.2f}  '
			      f'{d.roll:+6.1f} {d.pitch:+6.1f} {d.yaw:+6.1f}  '
			      f'{d.age():5.2f}')


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
