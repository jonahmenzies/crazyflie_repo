import subprocess
import sys
import time

import numpy as np
import yaml

import arrays as arrays_mod
import assign
import setpoint_map
from logger import Logger
from state_adapter import StateAdapter
from swarm import Swarm, label

TAKEOFF_Z    = 0.35     # hover height before the mission starts, metres
TAKEOFF_RATE = 0.30     # climb speed, m/s
LAND_RATE    = 0.25     # descent speed, m/s
SETTLE       = 2.0      # seconds to stabilise at hover before reading x0


def generate(interval):
	print(f'Generating arrays for {interval}s replan interval...')
	subprocess.run(['./generate', str(interval)], cwd='generator', check=True)


def geofence(config='config/uris.yaml'):
	with open(config) as f:
		g = yaml.safe_load(f).get('safety', {}).get('geofence')
	return g


def inside(x, N, g, labels=None):
	"""True if every drone's position is inside the fence."""
	if g is None:
		return True, None
	for i in range(N):
		who = f'drone {labels[i]}' if labels else f'slot {i}'
		p = (x[i*10 + 0], x[i*10 + 3], x[i*10 + 6])
		for v, (lo, hi), ax in zip(p, (g['x'], g['y'], g['z']), 'xyz'):
			if not (lo <= v <= hi):
				return False, f'{who} {ax}={v:+.2f} outside [{lo}, {hi}]'
	return True, None


def ramp(sw, start, target_z, rate, dt=0.05):
	"""Move every drone vertically from its current position to target_z."""
	z0 = np.array([p[2] for p in start])
	steps = max(1, int(abs(target_z - z0.mean()) / rate / dt))

	for s in range(steps + 1):
		f = (s + 1) / (steps + 1)
		for i, scf in enumerate(sw.scf):
			z = z0[i] + f * (target_z - z0[i])
			scf.cf.commander.send_position_setpoint(
				start[i][0], start[i][1], z, 0.0)
		time.sleep(dt)


def fly(interval, note='', mapper=setpoint_map):
	"""Fly the mission.

	mapper is the module supplying send() — which commander is used, and
	therefore how much of u actually reaches the drone. See the variant
	runners flight_loop_PWM.py and flight_loop_full.py.
	"""
	generate(interval)

	a = arrays_mod.Arrays()
	print(a)
	print(f'commander: {mapper.__name__}')

	g = geofence()
	ticks = range(0, a.n_steps, a.stride)

	with Swarm() as sw:
		sw.preflight_check()
		volts = sw.battery_check()
		sw.reset_estimators()

		adapter = StateAdapter(sw)
		adapter.start()

		try:
			# ---- assignment, on the ground ----
			ground = adapter.positions()
			mapping = assign.assign(ground, a, xy_only=True,
			                        labels=sw.labels())

			# Reorder the swarm so list position i IS formation slot i.
			# read() fills the state vector by list position, while the
			# arrays expect slot order — without this every drone's state
			# lands in the wrong block and the coupled cost acts on the
			# wrong pairs.
			order   = np.argsort(mapping)
			sw.scf  = [sw.scf[i]  for i in order]
			sw.live = [sw.live[i] for i in order]
			adapter.reorder(order)
			ground  = ground[order]
			mapping = np.arange(a.N)

			slot_uris   = list(sw.live)
			slot_labels = sw.labels()
			print('\nSlot assignment:')
			for s, uri in enumerate(slot_uris):
				print(f'  slot {s}: drone {label(uri)}')

			input('\nProps will spin. Enter to arm, Ctrl-C to abort: ')

			# unlock the low-level commander
			for scf in sw.scf:
				scf.cf.commander.send_setpoint(0, 0, 0, 0)

			# ---- takeoff, on position setpoints ----
			print(f'Climbing to {TAKEOFF_Z:.2f} m...')
			ramp(sw, ground, TAKEOFF_Z, TAKEOFF_RATE)

			hold = adapter.positions()
			t0 = time.time()
			while time.time() - t0 < SETTLE:
				for i, scf in enumerate(sw.scf):
					scf.cf.commander.send_position_setpoint(
						hold[i][0], hold[i][1], TAKEOFF_Z, 0.0)
				time.sleep(0.05)

			# ---- mission ----
			x_meas = adapter.read()
			ping   = 0
			print(f'\nMission: {len(ticks)} ticks, {a.n_pings} pings, '
			      f'{a.tT:.2f}s\n')

			with Logger(a, mapping, note=note, slot_uris=slot_uris,
			            volts=volts) as log:
				t_start = time.time()

				for k in ticks:
					target = t_start + k * a.dt
					now = time.time()
					if target > now:
						time.sleep(target - now)

					x_now = adapter.read()

					if ping + 1 < a.n_pings and k >= a.k0[ping + 1]:
						ping += 1
						x_meas = x_now
						print(f'  ping {ping}  t={k*a.dt:5.2f}s')

					x_pred = a.predict(k, ping, x_meas)
					u      = a.control(k, ping, x_meas)

					ok, why = inside(x_pred, a.N, g, slot_labels)
					if not ok:
						raise RuntimeError(f'geofence (predicted): {why}')

					ok, why = inside(x_now, a.N, g, slot_labels)
					if not ok:
						raise RuntimeError(f'geofence (measured): {why}')

					mapper.send(sw, x_pred, u, mapping, a.N, a.mass, a.g)
					log.write(k, ping, x_now, x_pred,
					          xd=(a.xd[k, 0], a.xd[k, 3]))

			# ---- land where they finished ----
			print('\nLanding...')
			ramp(sw, adapter.positions(), 0.05, LAND_RATE)
			sw.stop_all()

		except KeyboardInterrupt:
			print('\nAborted, cutting thrust')
			sw.stop_all()
		except RuntimeError as ex:
			print(f'\n{ex}, cutting thrust')
			sw.stop_all()
			raise
		finally:
			adapter.stop()


def dry(interval):
	"""Walk the trajectory with no hardware. Checks pacing and geofence."""
	generate(interval)
	a = arrays_mod.Arrays()
	print(a)

	g = geofence()
	mapping = np.arange(a.N)

	x_meas = np.zeros(a.n)
	for i, off in enumerate(a.formation_offsets):
		t = a.xd0 + off
		x_meas[i*10 + 0], x_meas[i*10 + 3], x_meas[i*10 + 6] = t

	ping = 0
	with Logger(a, mapping, note=f'dry run, {interval}s') as log:
		t_start = time.time()
		for k in range(0, a.n_steps, a.stride):
			target = t_start + k * a.dt
			now = time.time()
			if target > now:
				time.sleep(target - now)

			if ping + 1 < a.n_pings and k >= a.k0[ping + 1]:
				ping += 1
				print(f'  ping {ping}  t={k*a.dt:5.2f}s')

			x_pred = a.predict(k, ping, x_meas)
			u      = a.control(k, ping, x_meas)

			ok, why = inside(x_pred, a.N, g)
			if not ok:
				print(f'  GEOFENCE t={k*a.dt:5.2f}s  {why}')

			log.write(k, ping, x_meas, x_pred,
			          xd=(a.xd[k, 0], a.xd[k, 3]))


if __name__ == '__main__':
	args = [a for a in sys.argv[1:] if not a.startswith('-')]
	interval = float(args[0]) if args else 5.0

	if '--dry' in sys.argv:
		dry(interval)
	else:
		fly(interval, note=f'{interval}s replan')
