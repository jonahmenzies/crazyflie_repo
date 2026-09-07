import time
import yaml
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


class Swarm:
	def __init__(self, config='config/uris.yaml', cache='./cache'):
		with open(config) as f:
			cfg = yaml.safe_load(f)
		self.uris     = cfg['uris']
		self.required = cfg.get('swarm', {}).get('required', 5)
		self.cache    = cache
		self.scf  = []
		self.live = []

	def __enter__(self):
		cflib.crtp.init_drivers()
		for uri in self.uris:
			scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache=self.cache))
			try:
				scf.open_link()
			except Exception as ex:
				print(f'  {uri}  no response ({type(ex).__name__})')
				continue
			self.scf.append(scf)
			self.live.append(uri)
			print(f'  {uri}  drone {len(self.live) - 1}')

		if len(self.scf) != self.required:
			self.close()
			raise RuntimeError(f'need {self.required} drones, got {len(self.scf)}')

		self.N = len(self.scf)
		return self

	def __exit__(self, exc_type, exc, tb):
		self.stop_all()
		self.close()
		return False

	def close(self):
		for scf in self.scf:
			try:
				scf.close_link()
			except Exception:
				pass
		self.scf = []

	def reset_estimators(self):
		for scf in self.scf:
			scf.cf.param.set_value('kalman.resetEstimation', '1')
		time.sleep(0.2)
		for scf in self.scf:
			scf.cf.param.set_value('kalman.resetEstimation', '0')
		time.sleep(2.0)

	def stop_all(self):
		for scf in self.scf:
			try:
				scf.cf.commander.send_stop_setpoint()
			except Exception:
				pass
