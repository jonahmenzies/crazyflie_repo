# TATD over any time window, for the hardware and for the sims made by
# sim_flight.py (if they exist). Same formula as tatd.py, but scored between
# two times instead of over the last fraction of the flight.
#
#     python3 analysis/tatd_window.py logs/flight_<stamp>_full.csv <t_start> <t_end>
#
# e.g. 5 10 scores the middle of the flight, after the formation has formed
# up and before the end-of-mission gains fade.
# Run from formationControl/

import json
import os
import re
import sys

import numpy as np

from sim_flight import sim_files

path = sys.argv[1]
T0, T1 = float(sys.argv[2]), float(sys.argv[3])
stamp_path = path.replace('_full.csv', '')

data = np.genfromtxt(path, delimiter=',', names=True)
N = sum(1 for c in data.dtype.names if re.fullmatch(r'rx\d+', c))
interval = float(json.load(open(f'{stamp_path}_meta.json'))['interval'])
win = (data['t'] >= T0 - 1e-9) & (data['t'] <= T1 + 1e-9)


# RMS distance of every drone from its reference over the window
def tatd(pos):
	err_sq = sum((pos(i, a)[win] - data[f'r{a}{i}'][win])**2 for i in range(N) for a in 'xyz')
	err_sq = err_sq[~np.isnan(err_sq)]
	return np.sqrt(err_sq.sum() / (N * len(err_sq)))


print(path.split('/')[-1])
print(f'  window t = {T0:g} to {T1:g} s ({win.sum()} rows)\n')

for name, f in sim_files(stamp_path, interval).items():
	if os.path.exists(f):
		s = np.genfromtxt(f, delimiter=',', names=True)[:len(data)]
		print(f'  TATD  {"sim " + name:<22s} {tatd(lambda i, a: s[f"{a}{i}"]):.4f} m')
print(f'  TATD  {f"hardware {interval:g}s":<22s} {tatd(lambda i, a: data[f"m{a}{i}"]):.4f} m')
