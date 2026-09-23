"""Measured vs predicted altitude for all drones. For diagnosing sink."""

import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1]
df = pd.read_csv(path)

N = sum(1 for c in df.columns if re.fullmatch(r'mz\d+', c))

fig, ax = plt.subplots(figsize=(11, 6))
colors = plt.cm.tab10.colors

for i in range(N):
	c = colors[i % 10]
	ax.plot(df['t'], df[f'pz{i}'], '--', color=c, alpha=0.5,
	        label=f'slot {i} predicted')
	m = df[['t', f'mz{i}']].dropna()
	ax.plot(m['t'], m[f'mz{i}'], '-', color=c, linewidth=1.8,
	        label=f'slot {i} measured')

ax.axhline(0.0, color='k', linewidth=1.2)
ax.set_xlabel('t (s)')
ax.set_ylabel('z (m)')
ax.set_title(path.split('/')[-1])
ax.legend(ncol=2, fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print('measured z:')
for i in range(N):
	m = df[f'mz{i}'].dropna()
	if len(m):
		print(f'  slot {i}: start {m.iloc[0]:+.3f}  end {m.iloc[-1]:+.3f}  '
		      f'min {m.min():+.3f}')
