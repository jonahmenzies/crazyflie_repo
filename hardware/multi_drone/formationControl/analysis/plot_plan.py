"""Plot the planned trajectory from the arrays, given a starting state.

No hardware, no flight. Shows what the controller intends to do from a
given x0, so you can see whether the plan itself dives.
"""

import sys
sys.path.insert(0, 'flight')

import numpy as np
import matplotlib.pyplot as plt
from arrays import Arrays

a = Arrays()

# Two starting states worth comparing:
#   'slots'   — every drone already on its formation slot (the ideal)
#   'takeoff' — every drone at the hover height the flight loop uses,
#               still sitting at its ground xy. This is what the
#               controller actually sees at k=0 in a real flight.
MODE       = sys.argv[1] if len(sys.argv) > 1 else 'slots'
TAKEOFF_Z  = 0.35

x0 = np.zeros(a.n)
for i, off in enumerate(a.formation_offsets):
	t = a.xd0 + off
	x0[i*10 + 0] = t[0]
	x0[i*10 + 3] = t[1]
	x0[i*10 + 6] = t[2] if MODE == 'slots' else TAKEOFF_Z

ticks = list(range(0, a.n_steps, a.stride))
rows  = [k for k in ticks if k < a.k0[0] + a.c[0].shape[0] * a.stride]

t   = np.array([k * a.dt for k in rows])
pos = np.zeros((len(rows), a.N, 3))
u   = np.zeros((len(rows), a.N, 4))

for r, k in enumerate(rows):
	xp = a.predict(k, 0, x0)
	uu = a.control(k, 0, x0)
	for i in range(a.N):
		pos[r, i] = (xp[i*10 + 0], xp[i*10 + 3], xp[i*10 + 6])
		u[r, i]   = uu[i*4 : i*4 + 4]

fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

for i in range(a.N):
	ax[0].plot(t, pos[:, i, 2], label=f'slot {i}')
ax[0].axhline(0.0, color='k', linewidth=1.2)
ax[0].set_ylabel('z (m)')
ax[0].set_title(f'Planned altitude — x0 = {MODE}')
ax[0].legend(ncol=5, fontsize=8)
ax[0].grid(alpha=0.3)

for i in range(a.N):
	ax[1].plot(t, pos[:, i, 0], label=f'slot {i}')
ax[1].set_ylabel('x (m)')
ax[1].grid(alpha=0.3)

for i in range(a.N):
	ax[2].plot(t, u[:, i, 2], label=f'slot {i}')
ax[2].axhline(0.0, color='k', linewidth=0.8)
ax[2].set_ylabel('u_z (N)')
ax[2].set_xlabel('t (s)')
ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

below = pos[:, :, 2] < 0.0
if below.any():
	first = np.argmax(below.any(axis=1))
	print(f'plan goes below z=0 at t={t[first]:.2f}s')
	print(f'  min z per slot: '
	      + ', '.join(f'{i}:{pos[:, i, 2].min():+.2f}' for i in range(a.N)))
else:
	print(f'plan stays above ground. z range '
	      f'{pos[:,:,2].min():+.2f} to {pos[:,:,2].max():+.2f} m')
