# Simulates a flight with the existing sims (simulation/src, via simflight/),
# starting from the state the real drones were in when the mission started
# (the hover state at the handover), then scores the sims and the hardware
# with the same TATD as tatd.py.
#
#     python3 analysis/sim_flight.py logs/flight_<stamp>_full.csv [frac]
#
# Sims run: closed loop, open loop, and semi-open loop at the flight's
# replan interval. Their paths are saved next to the log as
#     flight_<stamp>_sim_closed.csv, _sim_open.csv, _sim_semi_<interval>.csv
# for plot_vs_sim.py.
#
# params.cpp must still hold the params the flight was flown with.
# Run from formationControl/

import json
import re
import subprocess
import sys

import numpy as np

path = sys.argv[1]
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
stamp_path = path.replace('_full.csv', '')

STATE_NAMES = ['x', 'xd', 'th', 'y', 'yd', 'ph', 'z', 'zd', 'ps', 'r']


# Paths of the sim files for a flight, keyed by sim name
def sim_files(stamp_path, interval):
	return {
		'closed loop':              f'{stamp_path}_sim_closed.csv',
		f'semi-open {interval:g}s': f'{stamp_path}_sim_semi_{interval:g}.csv',
		'open loop':                f'{stamp_path}_sim_open.csv',
	}


# Time the flight stopped if it ended before the mission did (geofence breach
# or abort), otherwise None
def mission_end(stamp_path, data, meta):
	import os
	t_end = float(data['t'][-1])
	if os.path.exists(f'{stamp_path}_breach.json'):
		return float(json.load(open(f'{stamp_path}_breach.json'))['t'])
	if t_end < float(meta.get('tT', t_end)) - 0.06:
		return t_end
	return None


# TATD over the last frac of the rows (same window and formula as tatd.py).
# pos(i, axis) and ref(i, axis) return one column per drone and axis.
def tatd(pos, ref, N, n_rows, frac):
	start = int(np.floor((1.0 - frac) * n_rows + 0.5))
	err_sq = np.zeros(n_rows - start)
	for i in range(N):
		for axis in 'xyz':
			err_sq += (pos(i, axis)[start:] - ref(i, axis)[start:])**2
	err_sq = err_sq[~np.isnan(err_sq)]
	return np.sqrt(err_sq.sum() / (N * len(err_sq)))


if __name__ == '__main__':
	data = np.genfromtxt(path, delimiter=',', names=True)
	meta = json.load(open(f'{stamp_path}_meta.json'))
	interval = float(meta['interval'])
	N = sum(1 for c in data.dtype.names if re.fullmatch(r'rx\d+', c))
	n_rows = len(data)

	# Starting state: what the drones measured at the first tick of the mission
	x0 = np.array([data[f'm{name}{i}'][0] for i in range(N) for name in STATE_NAMES])
	if np.isnan(x0).any():
		sys.exit('first row of the log has no measured state')
	np.savetxt(f'{stamp_path}_x0.csv', x0[None, :], delimiter=',', fmt='%.6f')

	# Build simflight if needed, then run the sims
	subprocess.run(['make', '-s', '-C', 'simflight'], check=True, stdout=subprocess.DEVNULL)
	subprocess.run(['simflight/simflight', f'{stamp_path}_x0.csv', f'{stamp_path}_sim',
	                f'{interval:g}'], check=True, stdout=subprocess.DEVNULL)

	files = sim_files(stamp_path, interval)
	sims = {name: np.genfromtxt(f, delimiter=',', names=True)[:n_rows]
	        for name, f in files.items()}

	# The sims must have the reference the drones flew, i.e. same params
	ref = sims['closed loop']
	for i in range(N):
		for axis in 'xyz':
			if len(ref) != n_rows or np.abs(ref[f'r{axis}{i}'] - data[f'r{axis}{i}']).max() > 1e-4:
				sys.exit('the sims do not have the reference this flight flew. '
				         'Put params.cpp back to the params it was flown with.')

	# A flight that left the geofence stops early. TATD20 needs the end of
	# the mission, so score everything over the part that was flown instead.
	ended = mission_end(stamp_path, data, meta)
	if ended is not None:
		FRAC = 1.0
		print(f'{path.split("/")[-1]} stopped at t = {ended:.2f} s, '
		      f'scoring the whole flown part\n')

	log_ref = lambda i, axis: data[f'r{axis}{i}']
	print(path.split('/')[-1])
	print(f'  replan interval {interval:g}s, all runs start from the measured hover state')
	start = int(np.floor((1.0 - FRAC) * n_rows + 0.5))
	print(f'  window: last {FRAC:.0%} of {n_rows} rows, '
	      f't = {data["t"][start]:.2f} to {data["t"][-1]:.2f} s\n')

	label = f'TATD{round(FRAC*100)}' if FRAC < 1 else 'TATD'
	for name, s in sims.items():
		v = tatd(lambda i, axis: s[f'{axis}{i}'], log_ref, N, n_rows, FRAC)
		print(f'  {label}  {"sim " + name:<22s} {v:.4f} m')
	v = tatd(lambda i, axis: data[f'm{axis}{i}'], log_ref, N, n_rows, FRAC)
	print(f'  {label}  {f"hardware {interval:g}s":<22s} {v:.4f} m')
