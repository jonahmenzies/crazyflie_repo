# Real flight against the sims started from the same hover state.
#   left    top view: each drone's path, hardware solid, sim dashed
#   middle  height of each drone over time
#   right   position error over time (RMS over the drones), hardware and
#           every sim, with the TATD20 window shaded
#
#     python3 analysis/plot_vs_sim.py logs/flight_<stamp>_full.csv
#
# Runs sim_flight.py first if the sim files aren't there yet.
# Saves logs/plots/flight_<stamp>_vs_sim.png and shows it.
# Run from formationControl/

import json
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

from sim_flight import sim_files

path = sys.argv[1]
stamp_path = path.replace('_full.csv', '')
stamp = stamp_path.split('/')[-1]

data = np.genfromtxt(path, delimiter=',', names=True)
interval = float(json.load(open(f'{stamp_path}_meta.json'))['interval'])
N = sum(1 for c in data.dtype.names if re.fullmatch(r'rx\d+', c))
semi = f'semi-open {interval:g}s'

files = sim_files(stamp_path, interval)
if not all(os.path.exists(f) for f in files.values()):
	subprocess.run([sys.executable, 'analysis/sim_flight.py', path], check=True)
sims = {name: np.genfromtxt(f, delimiter=',', names=True)[:len(data)]
        for name, f in files.items()}

# Where the flight left the geofence, if it did
breach = None
if os.path.exists(f'{stamp_path}_breach.json'):
	breach = json.load(open(f'{stamp_path}_breach.json'))

t = data['t']
colors = plt.cm.tab10.colors
fig, (ax_xy, ax_z, ax_e) = plt.subplots(1, 3, figsize=(17, 5.5))

# ---- Top view and height, hardware against the sim at the same interval ----
s = sims[semi]
for i in range(N):
	c = colors[i % 10]
	ax_xy.plot(data[f'rx{i}'], data[f'ry{i}'], '-', color='0.75', linewidth=1)
	ax_xy.plot(s[f'x{i}'], s[f'y{i}'], '--', color=c, linewidth=1.3)
	ax_xy.plot(data[f'mx{i}'], data[f'my{i}'], '-', color=c, linewidth=1.8,
	           label=f'slot {i}')
	ax_xy.plot(data[f'mx{i}'][0], data[f'my{i}'][0], 'o', color=c)

	ax_z.plot(t, data[f'rz{i}'], '-', color='0.75', linewidth=1)
	ax_z.plot(t, s[f'z{i}'], '--', color=c, linewidth=1.3)
	ax_z.plot(t, data[f'mz{i}'], '-', color=c, linewidth=1.8)

ax_xy.set_aspect('equal')
ax_xy.set_xlabel('x (m)')
ax_xy.set_ylabel('y (m)')
ax_xy.set_title(f'Top view: hardware (solid), sim {semi} (dashed), reference (grey)',
                fontsize=10)
ax_xy.legend(fontsize=8)

ax_z.set_xlabel('t (s)')
ax_z.set_ylabel('z (m)')
ax_z.set_title('Height', fontsize=10)

# ---- Position error over time ----
def rms_error(pos):
	err_sq = sum((pos(i, a) - data[f'r{a}{i}'])**2 for i in range(N) for a in 'xyz')
	return np.sqrt(err_sq / N)

ax_e.plot(t, rms_error(lambda i, a: data[f'm{a}{i}']), '-', color='k',
          linewidth=2, label=f'hardware {interval:g}s')
for (name, s), style in zip(sims.items(), ('-', '--', ':')):
	ax_e.plot(t, rms_error(lambda i, a: s[f'{a}{i}']), style, linewidth=1.5,
	          label=f'sim {name}')

if breach:
	for ax in (ax_z, ax_e):
		ax.axvline(breach['t'], color='r', linestyle='--', linewidth=1)
	ax_e.plot([], [], 'r--', label=f'geofence ({breach["kind"]}) {breach["t"]:.1f}s')
else:
	start = int(np.floor(0.8 * len(t) + 0.5))
	ax_e.axvspan(t[start], t[-1], color='0.9', zorder=0, label='TATD20 window')
ax_e.set_xlabel('t (s)')
ax_e.set_ylabel('position error, RMS over drones (m)')
ax_e.set_title('Tracking error', fontsize=10)
ax_e.legend(fontsize=8)

fig.suptitle(stamp)
fig.tight_layout()

os.makedirs('logs/plots', exist_ok=True)
out = f'logs/plots/{stamp}_vs_sim.png'
fig.savefig(out, dpi=150)
print(f'saved {out}')
plt.show()
