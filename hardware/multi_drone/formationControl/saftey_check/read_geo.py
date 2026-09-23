import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.localization.lighthouse_config_manager import LighthouseConfigWriter
from cflib.crazyflie.mem import LighthouseMemHelper

URI = 'radio://0/10/2M/E7E7E7E701'

geos = {}
done = False

def got_geo(geo_data):
    global geos, done
    geos = geo_data
    done = True

cflib.crtp.init_drivers()
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    helper = LighthouseMemHelper(scf.cf)
    helper.read_all_geos(got_geo)
    while not done:
        time.sleep(0.1)

for bs_id, geo in geos.items():
    print(f'Base station {bs_id}:')
    print(f'  origin: {geo.origin}')
    print(f'  rotation matrix: {geo.rotation_matrix}')
    print(f'  valid: {geo.valid}')
