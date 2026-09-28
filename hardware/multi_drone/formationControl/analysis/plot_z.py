# Measured vs predicted altitude for every drone. For diagnosing sink.
#
#     python3 analysis/plot_z.py logs/flight_<stamp>_full.csv
#
# Run from formationControl/

import re
import sys

import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]

# Read the CSV by column name. Empty cells become NaN.
data = np.genfromtxt(path, delimiter=',', names=True)

# Number of drones, from the measured z columns (mz0, mz1, ...)
N = sum(1 for c in data.dtype.names if re.fullmatch(r'mz\d+', c))

fig, ax = plt.subplots(figsize=(11, 6))
colors = plt.cm.tab10.colors

for i in range(N):
	c = colors[i % 10]

	# Predicted, dashed
	ax.plot(data['t'], data[f'pz{i}'], '--', color=c, alpha=0.5,
	        label=f'slot {i} predicted')

	# Measured, solid, from rows that have it
	has = ~np.isnan(data[f'mz{i}'])
	ax.plot(data['t'][has], data[f'mz{i}'][has], '-', color=c, linewidth=1.8,
	        label=f'slot {i} measured')

ax.axhline(0.0, color='k', linewidth=1.2)
ax.set_xlabel('t (s)')
ax.set_ylabel('z (m)')
ax.set_title(path.split('/')[-1])
ax.legend(ncol=2, fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Start, end and lowest measured height per drone
print('measured z:')
for i in range(N):
	m = data[f'mz{i}'][~np.isnan(data[f'mz{i}'])]
	if len(m):
		print(f'  slot {i}: start {m[0]:+.3f}  end {m[-1]:+.3f}  min {m.min():+.3f}')
