"""Quantify how much control authority the current params actually give.

Run after any params change + regenerate. Compares the controller's
commanded force range against the disturbance a thrust calibration error
causes, since that is the error the controller has to overcome.
"""

import sys
sys.path.insert(0, 'flight')

import numpy as np
from arrays import Arrays

a = Arrays()

x0 = np.zeros(a.n)
for i, off in enumerate(a.formation_offsets):
	t = a.xd0 + off
	x0[i*10 + 0], x0[i*10 + 3], x0[i*10 + 6] = t

rows = [k for k in range(0, a.n_steps, a.stride)
        if k < a.k0[0] + a.c[0].shape[0] * a.stride]

u = np.array([a.control(k, 0, x0) for k in rows])     # (T, 4N)

hover_N = a.mass * a.g
print(f'hover force        {hover_N:.4f} N\n')

print(f'{"slot":>4}  {"|u_z| max":>10}  {"% of hover":>10}  '
      f'{"pitch max":>10}  {"roll max":>9}  {"yawrate max":>11}')
for i in range(a.N):
	uz    = np.abs(u[:, i*4 + 2]).max()
	pitch = np.degrees(np.abs(u[:, i*4 + 0]).max())
	roll  = np.degrees(np.abs(u[:, i*4 + 1]).max())
	yaw   = np.degrees(np.abs(u[:, i*4 + 3]).max())
	print(f'{i:4d}  {uz:10.5f}  {100*uz/hover_N:9.2f}%  '
	      f'{pitch:9.2f}d  {roll:8.2f}d  {yaw:10.2f}d/s')

uz_max = np.abs(u[:, 2::4]).max()
print(f'\nvertical authority   {100*uz_max/hover_N:.2f}% of hover')

# z is a double integrator: zdd = -u_z/m, so a constant force error e
# displaces the drone by 0.5*(e/m)*t^2.
print(f'\ndrift from a thrust calibration error, over 2 s:')
for pct in (1, 2, 5, 10):
	e = hover_N * pct / 100
	d = 0.5 * (e / a.mass) * 2.0**2
	print(f'  {pct:2d}% error  ->  {e:.5f} N  ->  {d:5.2f} m')

verdict = 'BELOW' if uz_max < hover_N * 0.05 else 'above'
print(f'\nauthority is {verdict} a 5% calibration error')
