# 3D plot of one drone: predicted path and measured points.
#
#     python3 analysis/plot_0.py logs/flight_<stamp>_full.csv [slot]     slot defaults to 0
#
# Run from formationControl/

import sys

import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]
DRONE = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Read the CSV by column name. Empty cells become NaN.
data = np.genfromtxt(path, delimiter=',', names=True)
cols = data.dtype.names

# Column names for this drone: p = predicted, m = measured
px, py, pz = f'px{DRONE}', f'py{DRONE}', f'pz{DRONE}'
mx, my, mz = f'mx{DRONE}', f'my{DRONE}', f'mz{DRONE}'

if px not in cols:
	print(f'No column {px} — this looks like the sim-compatible CSV.')
	print('Use the _full.csv instead.')
	sys.exit(1)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Predicted path with start and end marked
ax.plot(data[px], data[py], data[pz], linewidth=1.8, label='predicted')
ax.scatter(data[px][0], data[py][0], data[pz][0],
           marker='o', s=60, label='start')
ax.scatter(data[px][-1], data[py][-1], data[pz][-1],
           marker='x', s=60, label='end')

# Measured positions, from rows that have them
if mx in cols:
	has = ~(np.isnan(data[mx]) | np.isnan(data[my]) | np.isnan(data[mz]))
	if has.any():
		ax.plot(data[mx][has], data[my][has], data[mz][has], 'o--', markersize=5,
		        alpha=0.8, label='measured (pings)')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title(f'Drone {DRONE} — {path.split("/")[-1]}')
ax.legend()

# Same scale on all three axes, centred on the path (z never below 0)
pts = np.array([data[px], data[py], data[pz]])
mid = pts.mean(axis=1)
half = max(pts.max(axis=1) - pts.min(axis=1)) / 2 or 0.1
ax.set_xlim(mid[0] - half, mid[0] + half)
ax.set_ylim(mid[1] - half, mid[1] + half)
ax.set_zlim(max(0, mid[2] - half), mid[2] + half)

plt.tight_layout()
plt.show()
