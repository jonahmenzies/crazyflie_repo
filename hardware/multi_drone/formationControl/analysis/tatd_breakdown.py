# Where the TATD error comes from: per drone and per axis, the mean error
# (a steady offset) and the RMS error over the same window as tatd.py.
# Then the formation shape (pair distances), and whether the formation
# centre is trailing the reference or sitting off to the side of it.
#
#     python3 analysis/tatd_breakdown.py logs/flight_<stamp>_full.csv [frac]
#
# Run from formationControl/

import re
import sys

import numpy as np

path = sys.argv[1]
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2

data = np.genfromtxt(path, delimiter=',', names=True)
N = sum(1 for c in data.dtype.names if re.fullmatch(r'rx\d+', c))

n_rows = len(data)
start = int(np.floor((1.0 - FRAC) * n_rows + 0.5))
window = data[start:]

print(path.split('/')[-1])
print(f'  window t = {window["t"][0]:.2f} to {window["t"][-1]:.2f} s\n')
print('  slot  axis    mean err    rms err    (measured - reference, m)')

for i in range(N):
	for axis in 'xyz':
		err = window[f'm{axis}{i}'] - window[f'r{axis}{i}']
		err = err[~np.isnan(err)]
		print(f'  {i:4d}  {axis:>4s}    {err.mean():+8.3f}   {np.sqrt((err**2).mean()):8.3f}')
	print()

# Formation shape: distance between each pair of drones compared with the
# distance the formation offsets ask for.
print('  pair   wanted     flown (mean)   error')
for i in range(N):
	for j in range(i + 1, N):
		want = np.array([window[f'r{a}{i}'] - window[f'r{a}{j}'] for a in 'xyz'])
		have = np.array([window[f'm{a}{i}'] - window[f'm{a}{j}'] for a in 'xyz'])
		ok = ~np.isnan(have).any(axis=0)
		dw = np.linalg.norm(want[:, ok], axis=0).mean()
		dh = np.linalg.norm(have[:, ok], axis=0).mean()
		print(f'  {i}-{j}   {dw:6.3f}     {dh:6.3f}        {dh - dw:+.3f}')

# Formation centre error split along the direction the reference is moving
# (negative along = behind the reference) and across it (positive = left).
t = window['t']
ref = np.array([np.mean([window[f'r{a}{i}'] for i in range(N)], axis=0) for a in 'xy'])
meas = np.array([np.mean([window[f'm{a}{i}'] for i in range(N)], axis=0) for a in 'xy'])
vel = np.gradient(ref, t, axis=1)
speed = np.linalg.norm(vel, axis=0)
ok = ~np.isnan(meas).any(axis=0) & (speed > 1e-3)

along_dir = vel[:, ok] / speed[ok]
left_dir = np.array([-along_dir[1], along_dir[0]])
err = meas[:, ok] - ref[:, ok]
along = (err * along_dir).sum(axis=0)
across = (err * left_dir).sum(axis=0)

print('\n  formation centre (xy), measured - reference')
print(f'    along path   mean {along.mean():+.3f} m   (negative = behind)')
print(f'    across path  mean {across.mean():+.3f} m   (positive = left of path)')
print(f'    reference speed  {speed[ok].mean():.3f} m/s, '
      f'so the along error is {-along.mean() / speed[ok].mean():+.2f} s of lag')
