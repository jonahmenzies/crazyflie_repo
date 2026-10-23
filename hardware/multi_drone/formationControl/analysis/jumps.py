# Finds sudden jumps in each drone's measured position, the sign of a
# positioning glitch: at 0.2 m/s a real drone moves about 1 cm per tick.
#
#     python3 analysis/jumps.py logs/flight_<stamp>_full.csv [threshold_m]
#
# Prints each drone's biggest tick-to-tick jump, and every jump over the
# threshold (default 0.05 m).
# Run from formationControl/

import re
import sys

import numpy as np

path = sys.argv[1]
THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

data = np.genfromtxt(path, delimiter=',', names=True)
N = sum(1 for c in data.dtype.names if re.fullmatch(r'rx\d+', c))
t = data['t']

print(path.split('/')[-1])
for i in range(N):
	pos = np.array([data[f'm{a}{i}'] for a in 'xyz'])
	step = np.linalg.norm(np.diff(pos, axis=1), axis=0)    # distance moved each tick
	k = int(np.nanargmax(step))
	print(f'\n  slot {i}: biggest jump {step[k]:.3f} m at t = {t[k+1]:.2f} s')

	for k in np.where(step > THRESHOLD)[0]:
		d = pos[:, k+1] - pos[:, k]
		print(f'    t = {t[k+1]:5.2f} s  {step[k]:.3f} m  '
		      f'(dx {d[0]:+.3f}, dy {d[1]:+.3f}, dz {d[2]:+.3f})')
