import json
import numpy as np


class Arrays:
	"""Precomputed state (c, e) and control (cu, eu) arrays.

	Indexed by ping number, not tau. Ping 0 is the first replan (tau = 0),
	ping 1 the second, and so on. Each array starts at its own global
	timestep k0 and stores every `stride`-th step.

	The state passed to predict() and control() must be anchored at the
	ping's tau — measured at the moment that ping began. There is no way to
	re-anchor a mid-horizon measurement onto an older ping: c(t) contracts
	as the closed loop converges, so inverting it is catastrophically
	ill-conditioned (cond ~1e12 by 3s). To get closer to closed-loop
	behaviour, shorten replan_interval instead.
	"""

	def __init__(self, path='arrays'):
		with open(f'{path}/manifest.json') as f:
			man = json.load(f)

		self.n        = man['n']
		self.m        = man['m']
		self.dt       = man['dt']
		self.stride   = man['stride']
		self.n_steps  = man['n_steps']
		self.tT       = man['tT']
		self.interval = man['replan_interval']
		self.mass     = man['mass']
		self.g        = man['g']

		self.formation_offsets = np.array(man['formation_offsets'])   # (N,3)
		self.xd0               = np.array(man['xd0'])                 # (3,)

		self.xd = np.fromfile(f'{path}/xd.bin').reshape(-1, self.n)
		self.c  = []
		self.e  = []
		self.cu = []
		self.eu = []
		self.k0 = []

		# Files are named by replan index, not tau — see generate.cpp.
		for idx, r in enumerate(man['replans']):
			c  = np.fromfile(f'{path}/c_{idx:03d}.bin').reshape(-1, self.n, self.n)
			e  = np.fromfile(f'{path}/e_{idx:03d}.bin').reshape(-1, self.n)
			cu = np.fromfile(f'{path}/cu_{idx:03d}.bin').reshape(-1, self.m, self.n)
			eu = np.fromfile(f'{path}/eu_{idx:03d}.bin').reshape(-1, self.m)

			for name, arr in (('c', c), ('e', e), ('cu', cu), ('eu', eu)):
				if arr.shape[0] != r['steps']:
					raise ValueError(
						f'replan {idx} (tau={r["tau"]}): manifest says '
						f'{r["steps"]} rows, {name} has {arr.shape[0]}')

			self.c.append(c)
			self.e.append(e)
			self.cu.append(cu)
			self.eu.append(eu)
			self.k0.append(r['k0'])

		self.n_pings = len(self.c)
		self.N = self.n // 10

	def row(self, k, ping):
		"""Global timestep k -> row index within this ping's array."""
		r = (k - self.k0[ping]) // self.stride
		if r < 0:
			raise IndexError(f'k={k} is before ping {ping} starts (k0={self.k0[ping]})')
		if r >= self.c[ping].shape[0]:
			raise IndexError(f'k={k} is past the end of ping {ping} '
			                 f'({self.c[ping].shape[0]} rows)')
		return r

	def predict(self, k, ping, x_meas):
		"""x_pred = c[k] @ x_meas + e[k], eq 10d."""
		r = self.row(k, ping)
		return self.c[ping][r] @ x_meas + self.e[ping][r]

	def control(self, k, ping, x_meas):
		"""u = cu[k] @ x_meas + eu[k], eq 10a with x substituted.

		Unclamped — apply inclination_protection before sending.
		"""
		r = self.row(k, ping)
		return self.cu[ping][r] @ x_meas + self.eu[ping][r]

	def __repr__(self):
		mb = sum(a.nbytes for a in self.c + self.e + self.cu + self.eu) / 1e6
		return (f'Arrays(n={self.n}, m={self.m}, dt={self.dt}, '
		        f'stride={self.stride}, pings={self.n_pings}, '
		        f'interval={self.interval}s, tT={self.tT:.2f}s, {mb:.1f} MB)')


if __name__ == '__main__':
	a = Arrays()
	print(a)

	assert np.allclose(a.c[0][0], np.eye(a.n)), 'c(tau,0) should be identity'
	assert np.allclose(a.e[0][0], 0),           'e(0) should be zero'
	print('c(tau,0) = I  ok')
	print('e(0) = 0      ok')

	print(f'\nxd0 = ({a.xd0[0]:+.2f}, {a.xd0[1]:+.2f}, {a.xd0[2]:+.2f})')
	print('formation targets at t=0:')
	for i, off in enumerate(a.formation_offsets):
		t = a.xd0 + off
		print(f'  {i}: ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f})')

	for p in range(a.n_pings):
		k0, rows = a.k0[p], a.c[p].shape[0]
		k_end = k0 + (rows - 1) * a.stride
		print(f'  ping {p}: k0={k0:5d}  rows={rows:4d}  '
		      f't={k0*a.dt:5.2f} to {k_end*a.dt:5.2f}s')
