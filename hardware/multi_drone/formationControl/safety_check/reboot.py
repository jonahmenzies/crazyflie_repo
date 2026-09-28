"""Reboot drones over the radio. Clears a latched supervisor lock without
walking over to each one and cycling the power switch.
"""

import sys
import time
import yaml
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.console import Console
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.power_switch import PowerSwitch


def _tolerant_incoming(self, packet):
	try:
		text = packet.data.decode('UTF-8', errors='replace')
	except Exception:
		return
	self.receivedChar.call(text)


Console._incoming = _tolerant_incoming


def load_uris(config='config/uris.yaml'):
	with open(config) as f:
		return yaml.safe_load(f)['uris']


if len(sys.argv) > 1:
	uris = sys.argv[1:]
else:
	uris = load_uris()

cflib.crtp.init_drivers()

for uri in uris:
	try:
		PowerSwitch(uri).stm_power_cycle()
		print(f'{uri}  rebooting')
	except Exception as ex:
		print(f'{uri}  failed ({type(ex).__name__})')
	time.sleep(0.5)

print('\nWaiting for boot...')
time.sleep(5.0)

for uri in uris:
	try:
		with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')):
			print(f'{uri}  back up')
	except Exception as ex:
		print(f'{uri}  not responding ({type(ex).__name__})')
