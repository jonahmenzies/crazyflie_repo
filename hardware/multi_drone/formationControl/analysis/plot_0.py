import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DRONE = 0

path = sys.argv[1]
df = pd.read_csv(path)

px, py, pz = f'px{DRONE}', f'py{DRONE}', f'pz{DRONE}'
mx, my, mz = f'mx{DRONE}', f'my{DRONE}', f'mz{DRONE}'

if px not in df.columns:
	print(f'No column {px} — this looks like the sim-compatible CSV.')
	print('Use the _full.csv instead.')
	sys.exit(1)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(df[px], df[py], df[pz], linewidth=1.8, label='predicted')
ax.scatter(df[px].iloc[0], df[py].iloc[0], df[pz].iloc[0],
           marker='o', s=60, label='start')
ax.scatter(df[px].iloc[-1], df[py].iloc[-1], df[pz].iloc[-1],
           marker='x', s=60, label='end')

if mx in df.columns:
	m = df[[mx, my, mz]].dropna()
	if len(m):
		ax.plot(m[mx], m[my], m[mz], 'o--', markersize=5,
		        alpha=0.8, label='measured (pings)')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title(f'Drone {DRONE} — {path.split("/")[-1]}')
ax.legend()

pts = np.array([df[px], df[py], df[pz]])
mid = pts.mean(axis=1)
rng = max(pts.max(axis=1) - pts.min(axis=1)) / 2 or 0.1
ax.set_xlim(mid[0] - rng, mid[0] + rng)
ax.set_ylim(mid[1] - rng, mid[1] + rng)
ax.set_zlim(max(0, mid[2] - rng), mid[2] + rng)

plt.tight_layout()
plt.show()
