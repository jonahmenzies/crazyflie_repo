# Reboots drones over the radio. Clears a supervisor lock without
# power cycling each drone by hand.
#
#     python3 safety_check/reboot.py                 every drone in config/uris.yaml
#     python3 safety_check/reboot.py radio://...     just the drones given
#
# Run from formationControl/

import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.power_switch import PowerSwitch

from common import load_uris

uris = sys.argv[1:] or load_uris()

cflib.crtp.init_drivers()

# Send the reboot to each drone
for uri in uris:
	try:
		PowerSwitch(uri).stm_power_cycle()
		print(f'{uri}  rebooting')
	except Exception as ex:
		print(f'{uri}  failed ({type(ex).__name__})')
	time.sleep(0.5)

print('\nWaiting for boot...')
time.sleep(5.0)

# Check each drone came back
for uri in uris:
	try:
		with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')):
			print(f'{uri}  back up')
	except Exception as ex:
		print(f'{uri}  not responding ({type(ex).__name__})')
