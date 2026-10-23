# Side-by-side comparison of several flights.
#   Figure 1: top view of every flight, one row per replan interval, one
#             column per repeat, and one sim panel at the end of each row.
#   Figure 2: tracking error over time, one panel per replan interval.
#             Each flight its own colour: hardware solid, its sim dotted.
#
# Each flight's sim is semi-open loop at its own interval, started from its
# own hover state (sim_flight.py, run here if it hasn't been yet). The sims
# use whatever params.cpp and params.hpp hold now, so they must still be the
# ones the flights were flown with.
#
#     python3 analysis/compare.py [stamp prefix] [formation] [mode]
#     python3 analysis/compare.py 20261022 flat          PWM flights (the default)
#     python3 analysis/compare.py 2026102 flat pos       position-setpoint flights
#
# Only flights flown with the live hover calibration are included.
# Saves both figures to logs/plots/ and shows them.
# Run from formationControl/

import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

FORMATIONS = {
	'flat':  [[0.45, 0.32, 0], [-0.45, 0.32, 0], [0.45, -0.32, 0], [-0.45, -0.32, 0]],
	'tower': [[0.2, 0.2, -0.3], [-0.2, 0.2, -0.1], [-0.2, -0.2, 0.1], [0.2, -0.2, 0.3]],
	'wide':  [[0.65, 0.4, 0], [-0.65, 0.4, 0], [0.65, -0.4, 0], [-0.65, -0.4, 0]],
	'combo': [[0.55, 0.35, 0.25], [-0.55, 0.35, -0.25], [0.55, -0.35, -0.25], [-0.55, -0.35, 0.25]],
}


def name_of(value, known, tol=0.01):
	for name, ref in known.items():
		if np.shape(ref) == np.shape(value) and np.allclose(value, ref, atol=tol):
			return name
	return 'other'


# TATD over the last 20% of the rows (same as tatd.py). Position columns are
# mx0, my0, ... in a flight log and x0, y0, ... in a sim file.
def tatd20(data, N, tag='m'):
	start = int(np.floor(0.8 * len(data) + 0.5))
	w = data[start:]
	err_sq = sum((w[f'{tag}{a}{i}'] - w[f'r{a}{i}'])**2 for i in range(N) for a in 'xyz')
	err_sq = err_sq[~np.isnan(err_sq)]
	return float(np.sqrt(err_sq.sum() / (N * len(err_sq))))


# Biggest tick-to-tick jump in any drone's measured position (as jumps.py)
def biggest_jump(data, N):
	jumps = []
	for i in range(N):
		pos = np.array([data[f'm{a}{i}'] for a in 'xyz'])
		step = np.linalg.norm(np.diff(pos, axis=1), axis=0)
		if len(step) and not np.isnan(step).all():
			jumps.append(np.nanmax(step))
	return max(jumps) if jumps else 0.0


PREFIX = sys.argv[1] if len(sys.argv) > 1 else ''
FORMATION = sys.argv[2] if len(sys.argv) > 2 else None
MODE = sys.argv[3] if len(sys.argv) > 3 else 'pwm'      # 'pwm' or 'pos'
GLITCH = 0.1    # a jump bigger than this, m, marks a flight as having a glitch

# ---- Collect the flights ----

flights = []
for meta_path in sorted(glob.glob(f'logs/flight_{PREFIX}*_meta.json')):
	stamp_path = meta_path.replace('_meta.json', '')
	try:
		meta = json.load(open(meta_path))
		manifest = json.load(open(f'{stamp_path}_manifest.json'))
		data = np.genfromtxt(f'{stamp_path}_full.csv', delimiter=',', names=True)
	except (OSError, ValueError):
		continue
	if 'live hover' not in meta.get('note', '') or 'rx0' not in data.dtype.names:
		continue
	mode = 'pos' if 'position setpoint' in meta.get('note', '') else 'pwm'
	if mode != MODE:
		continue

	formation = name_of(np.array(manifest['formation_offsets']), FORMATIONS)
	if FORMATION and formation != FORMATION:
		continue

	N = meta.get('N', 4)
	ended = None
	if os.path.exists(f'{stamp_path}_breach.json'):
		ended = json.load(open(f'{stamp_path}_breach.json'))['t']
	elif data['t'][-1] < meta['tT'] - 0.06:
		ended = float(data['t'][-1])

	# The sim of this flight: semi-open loop at its interval, from its hover state
	stamp = stamp_path.split('flight_')[-1]
	sim_path = f'{stamp_path}_sim_semi_{float(meta["interval"]):g}.csv'
	if not os.path.exists(sim_path):
		r = subprocess.run([sys.executable, 'analysis/sim_flight.py', f'{stamp_path}_full.csv'],
		                   capture_output=True, text=True)
		if r.returncode != 0:
			why = (r.stderr.strip() or r.stdout.strip()).splitlines()
			print(f'  no sim for {stamp}: {why[-1] if why else "sim_flight.py failed"}')
	sim = None
	if os.path.exists(sim_path):
		sim = np.genfromtxt(sim_path, delimiter=',', names=True)    # whole mission, even if the flight stopped

	flights.append({
		'stamp':    stamp_path.split('flight_')[-1],
		'sim':      sim,
		'interval': float(meta['interval']),
		'data':     data,
		'N':        N,
		'ended':    ended,
		'tatd':     None if ended else tatd20(data, N),
		'glitch':   biggest_jump(data, N) > GLITCH,
	})

if not flights:
	sys.exit('no matching flights')

by_interval = defaultdict(list)
for f in flights:
	by_interval[f['interval']].append(f)
intervals = sorted(by_interval)
cols = max(len(v) for v in by_interval.values())
colors = plt.cm.tab10.colors
tag = f'{PREFIX or "all"}{"_" + FORMATION if FORMATION else ""}_{MODE}'
os.makedirs('logs/plots', exist_ok=True)


def label(f):
	time = f'{f["stamp"][9:11]}:{f["stamp"][11:13]}'
	result = f'stopped {f["ended"]:.1f}s' if f['ended'] else f'TATD20 {f["tatd"]:.3f}'
	return f'{time}  {result}' + ('  (glitch)' if f['glitch'] else '')


# ---- Figure 1: top views ----
# One row per replan interval: the real flights, then one sim panel at the
# end of the row. The sims of flights at the same interval barely differ, so
# one is shown: the sim of the first flight in the row.

fig1, axes = plt.subplots(len(intervals), cols + 1, figsize=(4.2 * (cols + 1), 3.4 * len(intervals)),
                          squeeze=False, sharex=True, sharey=True)
for r, iv in enumerate(intervals):
	row = by_interval[iv]
	for c in range(cols):
		ax = axes[r][c]
		if c >= len(row):
			ax.axis('off')
			continue
		f = row[c]
		d = f['data']
		for i in range(f['N']):
			col = colors[i % 10]
			ax.plot(d[f'rx{i}'], d[f'ry{i}'], '-', color='0.8', linewidth=1)
			ax.plot(d[f'mx{i}'], d[f'my{i}'], '-', color=col, linewidth=1.3)
			ax.plot(d[f'mx{i}'][0], d[f'my{i}'][0], 'o', color=col, markersize=3)
			if f['ended']:
				ax.plot(d[f'mx{i}'][-1], d[f'my{i}'][-1], 'x', color='r', markersize=6)
		ax.set_aspect('equal')
		ax.set_title(label(f), fontsize=9, color='r' if f['ended'] else 'k')
		if c == 0:
			ax.set_ylabel(f'{iv:g} s replan\ny (m)')

	# The sim panel, last column
	ax = axes[r][cols]
	f = next((f for f in row if f['sim'] is not None), None)
	if f is None:
		ax.axis('off')
		continue
	sm = f['sim']
	for i in range(f['N']):
		col = colors[i % 10]
		ax.plot(sm[f'rx{i}'], sm[f'ry{i}'], '-', color='0.8', linewidth=1)
		ax.plot(sm[f'x{i}'], sm[f'y{i}'], '-', color=col, linewidth=1.3)
		ax.plot(sm[f'x{i}'][0], sm[f'y{i}'][0], 'o', color=col, markersize=3)
	ax.set_aspect('equal')
	ax.set_title(f'SIM {iv:g} s  TATD20 {tatd20(sm, f["N"], tag=""):.3f}', fontsize=9,
	             color='tab:blue')
	for spine in ax.spines.values():
		spine.set_edgecolor('tab:blue')

for c in range(cols + 1):
	axes[-1][c].set_xlabel('x (m)')
fig1.suptitle(f'Top view ({tag}): hardware flights, sim in the last column (blue). '
              f'Reference grey, red x = where the flight stopped')
fig1.tight_layout(rect=[0, 0, 1, 0.95])
out1 = f'logs/plots/compare_{tag}_topview.png'
fig1.savefig(out1, dpi=130)

# ---- Figure 2: tracking error over time, one panel per replan interval ----
# Each flight gets its own colour: its hardware error solid, its sim dotted.

fig2, axes2 = plt.subplots(len(intervals), 1, figsize=(11, 2.6 * len(intervals) + 0.8),
                           squeeze=False, sharex=True, sharey=True)
t_end = max(f['data']['t'][-1] for f in flights)
for r, iv in enumerate(intervals):
	ax = axes2[r][0]
	for c, f in enumerate(by_interval[iv]):
		colour = colors[c % 10]
		d = f['data']
		err_sq = sum((d[f'm{a}{i}'] - d[f'r{a}{i}'])**2 for i in range(f['N']) for a in 'xyz')
		err = np.sqrt(err_sq / f['N'])
		ax.plot(d['t'], err, '-', color=colour, linewidth=1.5, label=label(f))
		if f['ended']:
			ax.plot(d['t'][-1], err[-1], 'x', color='r', markersize=8, zorder=5)
		if f['sim'] is not None:
			sm = f['sim']
			sim_sq = sum((sm[f'{a}{i}'] - sm[f'r{a}{i}'])**2 for i in range(f['N']) for a in 'xyz')
			ax.plot(sm['t'], np.sqrt(sim_sq / f['N']), ':', color=colour, linewidth=1.5)
	ax.axvspan(0.8 * t_end, t_end, color='0.93', zorder=0)
	ax.set_ylabel(f'{iv:g} s replan\nerror (m)')
	ax.legend(fontsize=8, loc='upper right')
	ax.grid(alpha=0.3)
axes2[-1][0].set_xlabel('t (s)')
fig2.suptitle(f'Tracking error ({tag}): hardware solid, sim of the same flight dotted, '
              f'red x = stopped, grey = TATD20 window')
fig2.tight_layout(rect=[0, 0, 1, 0.96])
out2 = f'logs/plots/compare_{tag}_error.png'
fig2.savefig(out2, dpi=130)

sims = [f for f in flights if f['sim'] is not None]

print(f'{len(flights)} flights over {len(intervals)} intervals, {len(sims)} with sims')
for out in (out1, out2):
	print(f'saved {out}')
plt.show()
