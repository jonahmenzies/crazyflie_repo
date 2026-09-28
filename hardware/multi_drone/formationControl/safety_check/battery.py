# Battery level of each drone, checked one at a time. Works with some drones off.
#
#     python3 safety_check/battery.py                 every drone in config/uris.yaml
#     python3 safety_check/battery.py radio://...     just the drones given
#
# Run from formationControl/

import sys

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from common import load_uris, read_once

# Resting voltage -> charge % for a 1S LiPo. Only valid at rest,
# the cell sags 0.2-0.3 V under motor load.
SOC_CURVE = [
	(3.00,   0),
	(3.50,   5),
	(3.70,  15),
	(3.80,  30),
	(3.85,  40),
	(3.90,  50),
	(3.95,  60),
	(4.00,  70),
	(4.05,  80),
	(4.10,  90),
	(4.20, 100),
]


# Charge % for a resting voltage, interpolated from SOC_CURVE
def soc_percent(v):
	if v <= SOC_CURVE[0][0]:
		return 0
	if v >= SOC_CURVE[-1][0]:
		return 100

	for (v0, p0), (v1, p1) in zip(SOC_CURVE, SOC_CURVE[1:]):
		if v0 <= v <= v1:
			return int(round(p0 + (v - v0) * (p1 - p0) / (v1 - v0)))
	return 0


# One-word status for a resting voltage
def status_for(v):
	if v >= 4.10:
		return 'FULL'
	if v >= 3.95:
		return 'GOOD'
	if v >= 3.85:
		return 'MARGINAL'
	return 'LOW - CHARGE'


cflib.crtp.init_drivers()
uris = sys.argv[1:] or load_uris()

print()
for uri in uris:
	# Connect, read the voltage once, disconnect
	try:
		with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
			v = read_once(scf, 'pm.vbat', 'float', timeout=3.0)
	except Exception as ex:
		print(f'{uri}   offline ({type(ex).__name__})')
		continue

	if v is None:
		print(f'{uri}   no data')
	else:
		print(f'{uri}   {v:.2f} V   {soc_percent(v):3d}%   {status_for(v)}')
print()
