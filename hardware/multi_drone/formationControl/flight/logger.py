import csv
import json
import os
import shutil
import time
from datetime import datetime

import numpy as np


class Logger:
	"""Per-tick flight log.

	Writes two files into logs/:
	    flight_<stamp>.csv        sim-compatible columns, for plot.py
	                              and computeTATD
	    flight_<stamp>_full.csv   everything, for diagnosis

	Also copies the manifest alongside so any flight can be traced back to
	the exact arrays and parameters that produced it.
	"""

	def __init__(self, arrays, mapping, path='logs', arrays_path='arrays',
	             note='', slot_uris=None, volts=None):
		os.makedirs(path, exist_ok=True)
		stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

		self.stamp = stamp
		self.N     = arrays.N
		self.dt    = arrays.dt
		self.t0    = None
		self.last  = 0.0

		self.sim_path  = f'{path}/flight_{stamp}.csv'
		self.full_path = f'{path}/flight_{stamp}_full.csv'

		shutil.copy(f'{arrays_path}/manifest.json',
		            f'{path}/flight_{stamp}_manifest.json')

		with open(f'{path}/flight_{stamp}_meta.json', 'w') as f:
			json.dump({
				'stamp':    stamp,
				'note':     note,
				'N':        self.N,
				'dt':       self.dt,
				'stride':   arrays.stride,
				'tT':       arrays.tT,
				'interval': arrays.interval,
				'mapping':  [int(m) for m in mapping],
				# Which physical drone flew which slot. mapping is reset to
				# identity by the reorder, so without this the logs cannot
				# tell you whether a misbehaving slot is the same airframe
				# run to run.
				'slot_uris': list(slot_uris) if slot_uris else [],
				# Resting voltage per drone at arm time. Thrust per PWM
				# count falls as the cell sags, so this is the first thing
				# to check when vertical tracking looks worse than usual.
				'vbat': volts or {},
			}, f, indent=2)

		# sim-compatible: t, x0, y0, x1, y1, ..., xd, yd
		self.sim_file = open(self.sim_path, 'w', newline='')
		self.sim = csv.writer(self.sim_file)
		head = ['t']
		for i in range(self.N):
			head += [f'x{i}', f'y{i}']
		head += ['xd', 'yd']
		self.sim.writerow(head)

		# full: everything, measured and predicted side by side
		self.full_file = open(self.full_path, 'w', newline='')
		self.full = csv.writer(self.full_file)
		head = ['t', 'k', 'ping', 'wall', 'tick_ms']
		for tag in ('m', 'p'):
			for i in range(self.N):
				head += [f'{tag}x{i}', f'{tag}xd{i}', f'{tag}th{i}',
				         f'{tag}y{i}', f'{tag}yd{i}', f'{tag}ph{i}',
				         f'{tag}z{i}', f'{tag}zd{i}', f'{tag}ps{i}',
				         f'{tag}r{i}']
		self.full.writerow(head)

		self.n_rows  = 0
		self.n_meas  = 0
		self.tick_ms = []

	def write(self, k, ping, x_meas, x_pred, xd=None):
		"""One row. x_meas may be None on ticks where nothing was measured."""
		wall = time.time()
		if self.t0 is None:
			self.t0 = wall

		elapsed = wall - self.t0
		tick_ms = (elapsed - self.last) * 1000 if self.n_rows else 0.0
		self.last = elapsed
		if self.n_rows:
			self.tick_ms.append(tick_ms)

		t = k * self.dt

		row = [f'{t:.4f}']
		for i in range(self.N):
			row += [f'{x_pred[i*10 + 0]:.5f}', f'{x_pred[i*10 + 3]:.5f}']
		row += [f'{xd[0]:.5f}', f'{xd[1]:.5f}'] if xd is not None else ['', '']
		self.sim.writerow(row)

		row = [f'{t:.4f}', k, ping, f'{elapsed:.4f}', f'{tick_ms:.2f}']
		blank = [''] * (10 * self.N)
		row += [f'{v:.5f}' for v in x_meas] if x_meas is not None else blank
		row += [f'{v:.5f}' for v in x_pred]
		self.full.writerow(row)

		self.n_rows += 1
		if x_meas is not None:
			self.n_meas += 1

	def close(self):
		self.sim_file.close()
		self.full_file.close()

		if self.tick_ms:
			t = np.array(self.tick_ms)
			print(f'\n{self.n_rows} ticks, {self.n_meas} measured rows')
			print(f'  tick period  mean {t.mean():6.2f} ms   '
			      f'min {t.min():6.2f}   max {t.max():6.2f}')
			print(f'  achieved     {1000/t.mean():6.1f} Hz')
		print(f'  {self.sim_path}')
		print(f'  {self.full_path}')

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.close()
		return False


if __name__ == '__main__':
	from arrays import Arrays

	a = Arrays()
	mapping = np.arange(a.N)

	x0 = np.zeros(a.n)
	for i, off in enumerate(a.formation_offsets):
		t = a.xd0 + off
		x0[i*10 + 0], x0[i*10 + 3], x0[i*10 + 6] = t

	with Logger(a, mapping, note='dry run, no hardware') as log:
		for k in range(0, a.n_steps, a.stride):
			x_pred = a.predict(k, 0, x0)
			log.write(k, 0, x0 if k == 0 else None, x_pred)
			time.sleep(0.001)
