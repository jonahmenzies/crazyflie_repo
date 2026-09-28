import csv
import json
import os
import shutil
import time
from datetime import datetime

import numpy as np

# Column names for one drone's 10 states, in state vector order
STATE_NAMES = ['x', 'xd', 'th', 'y', 'yd', 'ph', 'z', 'zd', 'ps', 'r']


# Logs every tick of a flight. Each flight writes four files into logs/:
#     flight_<stamp>.csv            same columns as the sim, for plot.py and computeTATD
#     flight_<stamp>_full.csv       measured and predicted state, for diagnosis
#     flight_<stamp>_manifest.json  copy of the arrays manifest used
#     flight_<stamp>_meta.json      flight settings, drone slots and battery voltages
class Logger:

	def __init__(self, arrays, mapping, path='logs', arrays_path='arrays',
	             note='', slot_uris=None, volts=None):
		os.makedirs(path, exist_ok=True)
		stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

		self.stamp = stamp
		self.N = arrays.N
		self.dt = arrays.dt
		self.t0 = None      # wall clock of the first row
		self.last = 0.0     # elapsed time at the previous row

		self.sim_path = f'{path}/flight_{stamp}.csv'
		self.full_path = f'{path}/flight_{stamp}_full.csv'

		# Keep a copy of the manifest so the flight can be traced to its arrays
		shutil.copy(f'{arrays_path}/manifest.json',
		            f'{path}/flight_{stamp}_manifest.json')

		# Flight settings
		meta = {
			'stamp':     stamp,
			'note':      note,
			'N':         self.N,
			'dt':        self.dt,
			'stride':    arrays.stride,
			'tT':        arrays.tT,
			'interval':  arrays.interval,
			'mapping':   [int(m) for m in mapping],
			'slot_uris': list(slot_uris) if slot_uris else [],   # which drone flew which slot
			'vbat':      volts or {},                            # battery voltage at arm time
		}
		with open(f'{path}/flight_{stamp}_meta.json', 'w') as f:
			json.dump(meta, f, indent=2)

		# Sim-style file. Columns: t, x0, y0, x1, y1, ..., xd, yd
		self.sim_file = open(self.sim_path, 'w', newline='')
		self.sim = csv.writer(self.sim_file)

		header = ['t']
		for i in range(self.N):
			header += [f'x{i}', f'y{i}']
		header += ['xd', 'yd']
		self.sim.writerow(header)

		# Full file. Columns: timing, then measured (m) states, then predicted (p) states
		self.full_file = open(self.full_path, 'w', newline='')
		self.full = csv.writer(self.full_file)

		header = ['t', 'k', 'ping', 'wall', 'tick_ms']
		for tag in ('m', 'p'):
			for i in range(self.N):
				header += [f'{tag}{name}{i}' for name in STATE_NAMES]
		self.full.writerow(header)

		self.n_rows = 0      # rows written
		self.n_meas = 0      # rows that had a measured state
		self.tick_ms = []    # time between rows, for the timing summary

	# Write one row to each file. x_meas can be None if nothing was measured.
	def write(self, k, ping, x_meas, x_pred, xd=None):
		# Wall clock timing
		wall = time.time()
		if self.t0 is None:
			self.t0 = wall
		elapsed = wall - self.t0

		# Time since the previous row (0 for the first row)
		if self.n_rows > 0:
			tick_ms = (elapsed - self.last) * 1000
			self.tick_ms.append(tick_ms)
		else:
			tick_ms = 0.0
		self.last = elapsed

		t = k * self.dt

		# Sim-style row: predicted x, y of each drone, then the reference
		row = [f'{t:.4f}']
		for i in range(self.N):
			row += [f'{x_pred[i*10 + 0]:.5f}', f'{x_pred[i*10 + 3]:.5f}']
		if xd is not None:
			row += [f'{xd[0]:.5f}', f'{xd[1]:.5f}']
		else:
			row += ['', '']
		self.sim.writerow(row)

		# Full row: timing, measured state (blank if none), predicted state
		row = [f'{t:.4f}', k, ping, f'{elapsed:.4f}', f'{tick_ms:.2f}']
		if x_meas is not None:
			row += [f'{v:.5f}' for v in x_meas]
		else:
			row += [''] * (10 * self.N)
		row += [f'{v:.5f}' for v in x_pred]
		self.full.writerow(row)

		self.n_rows += 1
		if x_meas is not None:
			self.n_meas += 1

	# Close both files and print a timing summary
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

	# Lets Logger be used in a with block, closing the files at the end
	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.close()
		return False


# Self test: python3 flight/logger.py (run from formationControl/)
# Writes a log of the predicted trajectory from the ideal starting state
if __name__ == '__main__':
	from arrays import Arrays

	a = Arrays()
	mapping = np.arange(a.N)

	# Every drone starts exactly on its formation target
	x0 = np.zeros(a.n)
	for i, offset in enumerate(a.formation_offsets):
		x0[i*10 + 0], x0[i*10 + 3], x0[i*10 + 6] = a.xd0 + offset

	with Logger(a, mapping, note='dry run, no hardware') as log:
		for k in range(0, a.n_steps, a.stride):
			x_pred = a.predict(k, 0, x0)
			log.write(k, 0, x0 if k == 0 else None, x_pred)
			time.sleep(0.001)
