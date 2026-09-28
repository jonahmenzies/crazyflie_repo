# Top-down (x, y) plot of a flight or sim run.
#
#     python3 analysis/plot.py logs/flight_<stamp>_full.csv    predicted paths and measured points
#     python3 analysis/plot.py logs/flight_<stamp>.csv         sim-style file, with the reference
#
# Run from formationControl/

import csv
import sys

import matplotlib.pyplot as plt

fname = sys.argv[1] if len(sys.argv) > 1 else 'closedLoop.csv'
rows = list(csv.DictReader(open(fname)))
cols = rows[0].keys()

# The logger's _full.csv has a tick_ms column, the sim-style file doesn't
full = 'tick_ms' in cols


# Value from a row as a float, or None if the cell is empty
def get(r, key):
	v = r[key]
	return float(v) if v not in ('', None) else None


if full:
	# Number of drones, from the predicted x columns (px0, px1, ...)
	N = sum(1 for k in cols if k.startswith('px') and not k.startswith('pxd'))

	fig, ax = plt.subplots(figsize=(7, 7))
	colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

	for i in range(N):
		c = colours[i % len(colours)]

		# Predicted path
		xs = [get(r, f'px{i}') for r in rows]
		ys = [get(r, f'py{i}') for r in rows]
		ax.plot(xs, ys, linewidth=1.0, color=c, label=f'drone {i} predicted')

		# Measured positions, from rows that have them
		measured = [(get(r, f'mx{i}'), get(r, f'my{i}')) for r in rows]
		measured = [p for p in measured if p[0] is not None]
		if measured:
			ax.scatter([p[0] for p in measured], [p[1] for p in measured],
			           s=45, color=c, marker='x', zorder=3)

		# Start (circle) and end (diamond)
		ax.scatter(xs[0], ys[0], s=40, marker='o', color=c)
		ax.scatter(xs[-1], ys[-1], s=50, marker='D', color=c)

	ax.set_title(f'{fname}   (x = measured at pings)')

else:
	# Number of drones, from the x columns (x0, x1, ...)
	N = sum(1 for k in cols if k.startswith('x') and k != 'xd')

	fig, ax = plt.subplots(figsize=(7, 7))

	for i in range(N):
		xs = [float(r[f'x{i}']) for r in rows]
		ys = [float(r[f'y{i}']) for r in rows]
		ax.plot(xs, ys, linewidth=1.0, label=f'drone {i}')
		ax.scatter(xs[0], ys[0], s=40, marker='o')
		ax.scatter(xs[-1], ys[-1], s=50, marker='D')

	# Reference path, if the file has one
	xd = [get(r, 'xd') for r in rows]
	yd = [get(r, 'yd') for r in rows]
	if any(v is not None for v in xd):
		ax.plot(xd, yd, 'k--', linewidth=2, label='desired')

	ax.set_title(fname)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_aspect('equal')
ax.grid(True)
ax.legend(fontsize=8)
plt.show()
