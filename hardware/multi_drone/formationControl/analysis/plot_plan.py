# Plot what the controller plans to do from a given starting state.
# No hardware. Shows whether the plan itself is sensible before flying it.
#
#     python3 analysis/plot_plan.py slots      every drone starts on its formation slot
#     python3 analysis/plot_plan.py takeoff    every drone starts at takeoff height
#
# Run from formationControl/

import sys
sys.path.insert(0, 'flight')

import numpy as np
import matplotlib.pyplot as plt
from arrays import Arrays
from flight_loop import TAKEOFF_Z

a = Arrays()
MODE = sys.argv[1] if len(sys.argv) > 1 else 'slots'

# Starting state: formation slot xy, with z either the slot height or takeoff height
x0 = np.zeros(a.n)
for i, offset in enumerate(a.formation_offsets):
	x, y, z = a.xd0 + offset
	x0[i*10 + 0] = x
	x0[i*10 + 3] = y
	x0[i*10 + 6] = z if MODE == 'slots' else TAKEOFF_Z

# Timesteps covered by the first ping
last_k = a.k0[0] + a.c[0].shape[0] * a.stride
rows = [k for k in range(0, a.n_steps, a.stride) if k < last_k]

# Planned position and control input for each drone at each timestep
t = np.array([k * a.dt for k in rows])
pos = np.zeros((len(rows), a.N, 3))
u = np.zeros((len(rows), a.N, 4))

for r, k in enumerate(rows):
	x_pred = a.predict(k, 0, x0)
	u_k = a.control(k, 0, x0)
	for i in range(a.N):
		pos[r, i] = (x_pred[i*10 + 0], x_pred[i*10 + 3], x_pred[i*10 + 6])
		u[r, i] = u_k[i*4 : i*4 + 4]

fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

# Top: altitude
for i in range(a.N):
	ax[0].plot(t, pos[:, i, 2], label=f'slot {i}')
ax[0].axhline(0.0, color='k', linewidth=1.2)
ax[0].set_ylabel('z (m)')
ax[0].set_title(f'Planned altitude — x0 = {MODE}')
ax[0].legend(ncol=5, fontsize=8)
ax[0].grid(alpha=0.3)

# Middle: x position
for i in range(a.N):
	ax[1].plot(t, pos[:, i, 0], label=f'slot {i}')
ax[1].set_ylabel('x (m)')
ax[1].grid(alpha=0.3)

# Bottom: vertical control input
for i in range(a.N):
	ax[2].plot(t, u[:, i, 2], label=f'slot {i}')
ax[2].axhline(0.0, color='k', linewidth=0.8)
ax[2].set_ylabel('u_z (N)')
ax[2].set_xlabel('t (s)')
ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Check whether the plan ever goes below the floor
below = pos[:, :, 2] < 0.0
if below.any():
	first = np.argmax(below.any(axis=1))
	print(f'plan goes below z=0 at t={t[first]:.2f}s')
	print(f'  min z per slot: '
	      + ', '.join(f'{i}:{pos[:, i, 2].min():+.2f}' for i in range(a.N)))
else:
	print(f'plan stays above ground. z range '
	      f'{pos[:,:,2].min():+.2f} to {pos[:,:,2].max():+.2f} m')
