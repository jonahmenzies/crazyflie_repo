# Prints the Lighthouse base station geometry stored on a drone.
#
#     python3 safety_check/read_geo.py [uri]      default is drone 1
#
# Run from formationControl/

import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.mem import LighthouseMemHelper
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from common import uri_from_args

URI = uri_from_args()
result = {'geos': {}, 'done': False}


# Called by cflib once the geometry has been read
def on_geos(geo_data):
	result['geos'] = geo_data
	result['done'] = True


cflib.crtp.init_drivers()
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
	helper = LighthouseMemHelper(scf.cf)
	helper.read_all_geos(on_geos)
	while not result['done']:
		time.sleep(0.1)

for bs_id, geo in result['geos'].items():
	print(f'Base station {bs_id}:')
	print(f'  origin: {geo.origin}')
	print(f'  rotation matrix: {geo.rotation_matrix}')
	print(f'  valid: {geo.valid}')
