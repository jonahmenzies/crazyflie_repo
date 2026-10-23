# Each drone's pitch and roll while it hovers still at the start of each
# mission (first logged row, straight after the position-hold settle).
# A drone that needs a steady tilt to hover in place has an attitude bias:
# commanding 0 deg makes it accelerate. If the same drone shows the same
# tilt flight after flight, that's a trim offset worth correcting.
#
#     python3 analysis/trim.py [stamp prefix]      e.g. 20261022
#
# Run from formationControl/

import glob
import json
import re
import sys

import numpy as np

PREFIX = sys.argv[1] if len(sys.argv) > 1 else ''
G = 9.81

per_drone = {}    # drone label -> list of (pitch, roll) in degrees
for meta_path in sorted(glob.glob(f'logs/flight_{PREFIX}*_meta.json')):
	meta = json.load(open(meta_path))
	uris = meta.get('slot_uris')
	if not uris:
		continue
	try:
		data = np.genfromtxt(meta_path.replace('_meta.json', '_full.csv'),
		                     delimiter=',', names=True, max_rows=2)
	except (OSError, ValueError):
		continue

	for i, uri in enumerate(uris):
		if f'mth{i}' not in data.dtype.names:
			continue
		pitch = np.degrees(data[f'mth{i}'][0])
		roll = np.degrees(data[f'mph{i}'][0])
		if np.isnan(pitch) or np.isnan(roll):
			continue
		per_drone.setdefault(int(uri[-2:]), []).append((pitch, roll))

if not per_drone:
	sys.exit('no flights with slot_uris and measured attitude found')

print('Hover attitude at the start of each mission (deg)\n')
print('drone  flights   pitch mean (sd)    roll mean (sd)    drift in 2 s from that tilt')
for drone in sorted(per_drone):
	a = np.array(per_drone[drone])
	tilt = np.radians(np.hypot(a[:, 0].mean(), a[:, 1].mean()))
	drift = 0.5 * G * np.sin(tilt) * 2.0**2
	print(f'  {drone:2d}   {len(a):5d}    {a[:, 0].mean():+6.2f} ({a[:, 0].std():.2f})'
	      f'     {a[:, 1].mean():+6.2f} ({a[:, 1].std():.2f})     {drift:.2f} m')
