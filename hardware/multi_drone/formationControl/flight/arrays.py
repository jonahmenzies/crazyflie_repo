import json
import os
import numpy as np


# Number of timesteps in interval, or None if it isn't a whole number of steps
def steps_in(interval, dt):
	k = interval / dt
	return round(k) if abs(k - round(k)) < 1e-6 else None


# True if the arrays in path can be used for this replan interval:
# they were made after the generator binary was last built (so they match
# the current params), and their interval divides this one. A ping's arrays
# only depend on its start time, so the 0.1 s set contains the 1 s, 3 s,
# 5 s, ... sets as subsets of its pings.
def reusable(path, interval, binary):
	manifest_path = f'{path}/manifest.json'
	if not os.path.exists(manifest_path):
		return False
	if os.path.getmtime(manifest_path) < os.path.getmtime(binary):
		return False

	with open(manifest_path) as f:
		manifest = json.load(f)
	k_base = steps_in(manifest['replan_interval'], manifest['dt'])
	k_want = steps_in(interval, manifest['dt'])
	return bool(k_base and k_want and k_want % k_base == 0)


# Loads the arrays made by generator/generate.
# Each ping (replan) has its own c, e, cu and eu arrays.
# x_meas must be the state measured at the start of the ping it is used with.
# interval picks a coarser replan interval out of a finer set of arrays
# (e.g. 3 s out of the 0.1 s set). None uses every ping in the set.
class Arrays:

	def __init__(self, path='arrays', interval=None):
		# Read the manifest that describes the arrays
		with open(f'{path}/manifest.json') as f:
			manifest = json.load(f)

		# Sizes
		self.n = manifest['n']      # state length, 10 per drone
		self.m = manifest['m']      # control length, 4 per drone
		self.N = self.n // 10       # number of drones

		# Timing
		self.dt       = manifest['dt']               # timestep, s
		self.stride   = manifest['stride']           # arrays keep every stride-th step
		self.n_steps  = manifest['n_steps']          # timesteps in the mission
		self.tT       = manifest['tT']               # mission length, s
		self.base     = manifest['replan_interval']  # interval the arrays were made for, s
		self.interval = interval if interval is not None else self.base  # interval flown, s

		# Physical constants
		self.mass = manifest['mass']
		self.g    = manifest['g']

		# Formation shape and the formation centre at t = 0.
		# The manifest's xd0 is drone 0's position (centre + its offset),
		# so take drone 0's offset off to get the centre.
		self.formation_offsets = np.array(manifest['formation_offsets'])  # (N, 3)
		self.xd0 = np.array(manifest['xd0']) - self.formation_offsets[0]  # (3,)

		# Reference trajectory, one row per timestep
		self.xd = np.fromfile(f'{path}/xd.bin').reshape(-1, self.n)

		# Lists with one entry per ping
		self.c  = []
		self.e  = []
		self.cu = []
		self.eu = []
		self.k0 = []    # first timestep of each ping

		# Ping start steps wanted for this interval
		k_base = steps_in(self.base, self.dt)
		k_want = steps_in(self.interval, self.dt)
		if not (k_base and k_want and k_want % k_base == 0):
			raise ValueError(f'{self.interval}s pings are not in the {self.base}s arrays')
		wanted = set(range(0, self.n_steps, k_want))

		for i, replan in enumerate(manifest['replans']):
			if replan['k0'] not in wanted:
				continue

			# Load this ping's four arrays
			c  = np.fromfile(f'{path}/c_{i:03d}.bin').reshape(-1, self.n, self.n)
			e  = np.fromfile(f'{path}/e_{i:03d}.bin').reshape(-1, self.n)
			cu = np.fromfile(f'{path}/cu_{i:03d}.bin').reshape(-1, self.m, self.n)
			eu = np.fromfile(f'{path}/eu_{i:03d}.bin').reshape(-1, self.m)

			# Check each array has the number of rows the manifest says
			for name, arr in (('c', c), ('e', e), ('cu', cu), ('eu', eu)):
				if arr.shape[0] != replan['steps']:
					raise ValueError(
						f'replan {i} (tau={replan["tau"]}): manifest says '
						f'{replan["steps"]} rows, {name} has {arr.shape[0]}')

			self.c.append(c)
			self.e.append(e)
			self.cu.append(cu)
			self.eu.append(eu)
			self.k0.append(replan['k0'])

		self.n_pings = len(self.c)

		# Every wanted ping must be in the set
		missing = sorted(wanted - set(self.k0))
		if missing:
			raise ValueError(f'arrays have no ping starting at k={missing}')

	# Convert global timestep k into a row number in this ping's arrays
	def row(self, k, ping):
		r = (k - self.k0[ping]) // self.stride
		rows = self.c[ping].shape[0]

		if r < 0:
			raise IndexError(f'k={k} is before ping {ping} starts (k0={self.k0[ping]})')
		if r >= rows:
			raise IndexError(f'k={k} is past the end of ping {ping} ({rows} rows)')
		return r

	# Predicted state: x_pred = c @ x_meas + e  (paper eq 10d)
	def predict(self, k, ping, x_meas):
		r = self.row(k, ping)
		return self.c[ping][r] @ x_meas + self.e[ping][r]

	# Control input: u = cu @ x_meas + eu  (paper eq 10a)
	# Not clamped here, setpoint_map_pwm clamps it before sending
	def control(self, k, ping, x_meas):
		r = self.row(k, ping)
		return self.cu[ping][r] @ x_meas + self.eu[ping][r]

	# Summary line printed with print(a)
	def __repr__(self):
		all_arrays = self.c + self.e + self.cu + self.eu
		mb = sum(arr.nbytes for arr in all_arrays) / 1e6
		return (f'Arrays(n={self.n}, m={self.m}, dt={self.dt}, '
		        f'stride={self.stride}, pings={self.n_pings}, '
		        f'interval={self.interval}s, tT={self.tT:.2f}s, {mb:.1f} MB'
		        + (f', from {self.base}s set)' if self.interval != self.base else ')'))


# Self test: python3 flight/arrays.py (run from formationControl/)
if __name__ == '__main__':
	a = Arrays()
	print(a)

	# At the start of the first ping, c should be identity and e zero
	assert np.allclose(a.c[0][0], np.eye(a.n)), 'c(tau,0) should be identity'
	assert np.allclose(a.e[0][0], 0),           'e(0) should be zero'
	print('c(tau,0) = I  ok')
	print('e(0) = 0      ok')

	# Where each drone should be at t = 0
	print(f'\nxd0 = ({a.xd0[0]:+.2f}, {a.xd0[1]:+.2f}, {a.xd0[2]:+.2f})')
	print('formation targets at t=0:')
	for i, offset in enumerate(a.formation_offsets):
		x, y, z = a.xd0 + offset
		print(f'  {i}: ({x:+.2f}, {y:+.2f}, {z:+.2f})')

	# Time span covered by each ping
	for p in range(a.n_pings):
		k0   = a.k0[p]
		rows = a.c[p].shape[0]
		k_end = k0 + (rows - 1) * a.stride
		print(f'  ping {p}: k0={k0:5d}  rows={rows:4d}  '
		      f't={k0*a.dt:5.2f} to {k_end*a.dt:5.2f}s')
