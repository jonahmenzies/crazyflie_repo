# How much control authority the current params give, compared with the
# drift a thrust calibration error would cause. Run after changing params
# and regenerating the arrays.
#
#     python3 analysis/authority.py
#
# Run from formationControl/

import sys
sys.path.insert(0, 'flight')

import numpy as np
from arrays import Arrays

a = Arrays()

# Every drone starts exactly on its formation target
x0 = np.zeros(a.n)
for i, offset in enumerate(a.formation_offsets):
	x0[i*10 + 0], x0[i*10 + 3], x0[i*10 + 6] = a.xd0 + offset

# Timesteps covered by the first ping
last_k = a.k0[0] + a.c[0].shape[0] * a.stride
rows = [k for k in range(0, a.n_steps, a.stride) if k < last_k]

# Control input at each of those timesteps, shape (T, 4N)
u = np.array([a.control(k, 0, x0) for k in rows])

hover_N = a.mass * a.g
print(f'hover force        {hover_N:.4f} N\n')

# Largest command on each channel, per drone
print(f'{"slot":>4}  {"|u_z| max":>10}  {"% of hover":>10}  '
      f'{"pitch max":>10}  {"roll max":>9}  {"yawrate max":>11}')
for i in range(a.N):
	b = i * 4
	uz = np.abs(u[:, b + 2]).max()
	pitch = np.degrees(np.abs(u[:, b + 0]).max())
	roll = np.degrees(np.abs(u[:, b + 1]).max())
	yaw = np.degrees(np.abs(u[:, b + 3]).max())
	print(f'{i:4d}  {uz:10.5f}  {100*uz/hover_N:9.2f}%  '
	      f'{pitch:9.2f}d  {roll:8.2f}d  {yaw:10.2f}d/s')

# Largest vertical command across all drones
uz_max = np.abs(u[:, 2::4]).max()
print(f'\nvertical authority   {100*uz_max/hover_N:.2f}% of hover')

# z is a double integrator (z'' = -u_z/m), so a constant force error e
# moves the drone 0.5*(e/m)*t^2 in time t
print(f'\ndrift from a thrust calibration error, over 2 s:')
for pct in (1, 2, 5, 10):
	e = hover_N * pct / 100
	d = 0.5 * (e / a.mass) * 2.0**2
	print(f'  {pct:2d}% error  ->  {e:.5f} N  ->  {d:5.2f} m')

verdict = 'BELOW' if uz_max < hover_N * 0.05 else 'above'
print(f'\nauthority is {verdict} a 5% calibration error')
