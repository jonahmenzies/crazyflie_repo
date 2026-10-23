# One line per flight, so the logs can be told apart without opening them:
# when it flew, replan interval, path, formation, whether it used the live
# hover calibration, starting battery, biggest position jump and TATD20.
#
#     python3 analysis/flights.py              every flight in logs/
#     python3 analysis/flights.py 20261022     only flights whose stamp starts with this
#
# Also writes logs/flights_summary.csv for the report.
# Run from formationControl/

import csv
import glob
import json
import re
import sys

import numpy as np

PREFIX = sys.argv[1] if len(sys.argv) > 1 else ''

# Known formations, as formation_offsets in params.cpp
FORMATIONS = {
	'flat':  [[0.45, 0.32, 0], [-0.45, 0.32, 0], [0.45, -0.32, 0], [-0.45, -0.32, 0]],
	'tower': [[0.2, 0.2, -0.3], [-0.2, 0.2, -0.1], [-0.2, -0.2, 0.1], [0.2, -0.2, 0.3]],
	'wide':  [[0.65, 0.4, 0], [-0.65, 0.4, 0], [0.65, -0.4, 0], [-0.65, -0.4, 0]],
	'combo': [[0.55, 0.35, 0.25], [-0.55, 0.35, -0.25], [0.55, -0.35, -0.25], [-0.55, -0.35, 0.25]],
}

# Known paths, by where the formation centre starts
PATHS = {
	'oval':       [-0.42, -0.075, 0.70],
	'climb loop': [-0.35, -0.28, 0.50],
	'vert loop':  [-0.15, -0.075, 0.50],
}


def name_of(value, known, tol=0.01):
	for name, ref in known.items():
		if np.shape(ref) == np.shape(value) and np.allclose(value, ref, atol=tol):
			return name
	return 'other'


# TATD over the last 20% of the rows (same as tatd.py), or None if the log
# has no reference columns
def tatd20(data, N):
	if 'rx0' not in data.dtype.names:
		return None
	start = int(np.floor(0.8 * len(data) + 0.5))
	w = data[start:]
	err_sq = sum((w[f'm{a}{i}'] - w[f'r{a}{i}'])**2 for i in range(N) for a in 'xyz')
	err_sq = err_sq[~np.isnan(err_sq)]
	return float(np.sqrt(err_sq.sum() / (N * len(err_sq))))


# Biggest tick-to-tick jump in any drone's measured position (as jumps.py),
# or None if the log has no measured positions
def biggest_jump(data, N):
	jumps = []
	for i in range(N):
		if f'mx{i}' not in data.dtype.names or len(data) < 2:
			return None
		pos = np.array([data[f'm{a}{i}'] for a in 'xyz'])
		step = np.linalg.norm(np.diff(pos, axis=1), axis=0)
		if np.isnan(step).all():
			return None
		jumps.append(np.nanmax(step))
	return float(max(jumps))


rows = []
for meta_path in sorted(glob.glob(f'logs/flight_{PREFIX}*_meta.json')):
	stamp_path = meta_path.replace('_meta.json', '')
	stamp = stamp_path.split('flight_')[-1]
	try:
		meta = json.load(open(meta_path))
		manifest = json.load(open(f'{stamp_path}_manifest.json'))
		data = np.genfromtxt(f'{stamp_path}_full.csv', delimiter=',', names=True)
	except (OSError, ValueError):
		continue

	N = meta.get('N', 4)
	offsets = np.array(manifest['formation_offsets'])
	centre = np.array(manifest['xd0']) - offsets[0]
	volts = meta.get('vbat', {}).values()
	try:
		t = tatd20(data, N)
		jump = biggest_jump(data, N)
	except (ValueError, KeyError):
		t = jump = None

	# How the mission ended. TATD20 only counts for complete flights.
	end = 'done'
	try:
		breach = json.load(open(f'{stamp_path}_breach.json'))
		end = f'fence {breach["t"]:.1f}s ({breach["kind"][0]})'
		t = None
	except OSError:
		if 't' in data.dtype.names and len(data) and \
		   float(data['t'][-1]) < float(meta.get('tT', 0)) - 0.06:
			end = f'stop {float(data["t"][-1]):.1f}s'
			t = None

	rows.append({
		'stamp':     stamp,
		'interval':  f'{float(meta.get("interval", manifest["replan_interval"])):g}',
		'path':      name_of(centre, PATHS),
		'formation': name_of(offsets, FORMATIONS),
		'mode':      'pos' if 'position setpoint' in meta.get('note', '') else 'pwm',
		'hover':     'live' if 'live hover' in meta.get('note', '') else 'stored',
		'vbat_min':  f'{min(volts):.2f}' if volts else '',
		'jump_m':    f'{jump:.3f}' if jump is not None else '',
		'end':       end,
		'tatd20_m':  f'{t:.4f}' if t is not None else '',
	})

if not rows:
	sys.exit('no flights found')

cols = list(rows[0].keys())
widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
print('  '.join(c.ljust(widths[c]) for c in cols))
for r in rows:
	flag = '   <- glitch' if r['jump_m'] and float(r['jump_m']) > 0.1 else ''
	print('  '.join(r[c].ljust(widths[c]) for c in cols) + flag)

with open('logs/flights_summary.csv', 'w', newline='') as f:
	w = csv.DictWriter(f, fieldnames=cols)
	w.writeheader()
	w.writerows(rows)
print(f'\n{len(rows)} flights, saved logs/flights_summary.csv')
