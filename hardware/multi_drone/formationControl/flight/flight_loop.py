# Flies the formation mission.
# Takeoff, hover and landing use position setpoints.
# The mission itself sends the paper's u with send_setpoint (see setpoint_map_pwm.py).
#
#     python3 flight/flight_loop.py 5          fly with a 5 s replan interval
#     python3 flight/flight_loop.py 5 --dry    no hardware, checks timing and geofence
#
# Run from formationControl/

import subprocess
import sys
import time

import numpy as np
import yaml

import arrays as arrays_mod
import assign
import setpoint_map_pwm as pwm
from logger import Logger
from state_adapter import StateAdapter
from swarm import Swarm, label

# ---- Settings ----

TAKEOFF_Z    = 0.35     # hover height before the mission starts, m
TAKEOFF_RATE = 0.30     # climb speed, m/s
LAND_Z       = 0.05     # height the landing ramp goes down to, m
LAND_RATE    = 0.25     # descent speed, m/s
SETTLE       = 2.0      # seconds to hover before reading the starting state


# ---- Helpers ----

# Run the C++ generator to build the arrays for this replan interval
def generate(interval):
	print(f'Generating arrays for {interval}s replan interval...')
	subprocess.run(['./generate', str(interval)], cwd='generator', check=True)


# Geofence limits from the config file, or None if there aren't any
def geofence(config='config/uris.yaml'):
	with open(config) as f:
		cfg = yaml.safe_load(f)
	return cfg.get('safety', {}).get('geofence')


# Check every drone in state vector x is inside the geofence.
# Returns (True, None) if so, otherwise (False, reason).
def inside(x, N, g, labels=None):
	if g is None:
		return True, None

	for i in range(N):
		who = f'drone {labels[i]}' if labels else f'slot {i}'
		b = i * 10
		position = (x[b + 0], x[b + 3], x[b + 6])

		for axis, v in zip('xyz', position):
			lo, hi = g[axis]
			if not (lo <= v <= hi):
				return False, f'{who} {axis}={v:+.2f} outside [{lo}, {hi}]'

	return True, None


# Move every drone straight up or down from start to target_z
def ramp(sw, start, target_z, rate, dt=0.05):
	z0 = np.array([p[2] for p in start])
	steps = max(1, int(abs(target_z - z0.mean()) / rate / dt))

	for s in range(steps + 1):
		fraction = (s + 1) / (steps + 1)    # how far along the ramp we are
		for i, scf in enumerate(sw.scf):
			z = z0[i] + fraction * (target_z - z0[i])
			scf.cf.commander.send_position_setpoint(start[i][0], start[i][1], z, 0.0)
		time.sleep(dt)


# Sleep until wall clock time target (returns straight away if already past it)
def wait_until(target):
	now = time.time()
	if target > now:
		time.sleep(target - now)


# True if the next ping starts at timestep k
def new_ping(a, k, ping):
	return ping + 1 < a.n_pings and k >= a.k0[ping + 1]


# State vector with every drone exactly on its formation target, everything else zero
def formation_start(a):
	x = np.zeros(a.n)
	for i, offset in enumerate(a.formation_offsets):
		x[i*10 + 0], x[i*10 + 3], x[i*10 + 6] = a.xd0 + offset
	return x


# ---- Flight steps ----

# Print each slot's thrust calibration. Refuse to fly if a drone has none.
def check_calibration(slot_uris):
	print('\nThrust calibration:')
	missing = []

	for s, uri in enumerate(slot_uris):
		if uri in pwm.THRUST:
			hover, slope = pwm.THRUST[uri]
			print(f'  slot {s}: drone {label(uri):2d}  hover {hover}  '
			      f'slope {slope:.3e} N/count')
		else:
			missing.append(label(uri))
			print(f'  slot {s}: drone {label(uri):2d}  NOT CALIBRATED')

	if missing:
		raise RuntimeError(
			f'no thrust calibration for drone(s) {missing}, '
			f'run thrust_id.py and add them to setpoint_map_pwm.THRUST')


# Match drones on the ground to formation slots, then reorder the swarm so
# list position i is slot i. The arrays expect slot order, so without the
# reorder each drone's state would land in the wrong part of the state vector.
# Returns the ground positions in the new order.
def assign_slots(sw, adapter, a):
	ground = adapter.positions()
	mapping = assign.assign(ground, a, xy_only=True, labels=sw.labels())

	order = np.argsort(mapping)
	sw.scf = [sw.scf[i] for i in order]
	sw.live = [sw.live[i] for i in order]
	adapter.reorder(order)

	print('\nSlot assignment:')
	for s, uri in enumerate(sw.live):
		print(f'  slot {s}: drone {label(uri)}')

	return ground[order]


# Unlock the motors (the firmware needs one zero setpoint first)
def arm(sw):
	for scf in sw.scf:
		scf.cf.commander.send_setpoint(0, 0, 0, 0)


# Climb to TAKEOFF_Z, then hover in place for SETTLE seconds
def takeoff(sw, adapter, ground):
	print(f'Climbing to {TAKEOFF_Z:.2f} m...')
	ramp(sw, ground, TAKEOFF_Z, TAKEOFF_RATE)

	hold = adapter.positions()
	t0 = time.time()
	while time.time() - t0 < SETTLE:
		for i, scf in enumerate(sw.scf):
			scf.cf.commander.send_position_setpoint(hold[i][0], hold[i][1], TAKEOFF_Z, 0.0)
		time.sleep(0.05)


# The formation mission: send the paper's u every tick until the end
def mission(sw, adapter, a, fence, note, volts):
	mapping = np.arange(a.N)    # after the reorder, slot i is list position i
	slot_uris = list(sw.live)
	slot_labels = sw.labels()
	ticks = range(0, a.n_steps, a.stride)

	x_meas = adapter.read()     # starting state for ping 0
	ping = 0
	print(f'\nMission: {len(ticks)} ticks, {a.n_pings} pings, '
	      f'{a.tT:.2f}s, send_setpoint (PWM)\n')

	with Logger(a, mapping, note=note, slot_uris=slot_uris, volts=volts) as log:
		t_start = time.time()

		for k in ticks:
			wait_until(t_start + k * a.dt)
			x_now = adapter.read()

			# At each ping, the current state becomes the new starting state
			if new_ping(a, k, ping):
				ping += 1
				x_meas = x_now
				print(f'  ping {ping}  t={k*a.dt:5.2f}s')

			x_pred = a.predict(k, ping, x_meas)
			u = a.control(k, ping, x_meas)

			# Stop if the predicted or measured position leaves the geofence
			ok, why = inside(x_pred, a.N, fence, slot_labels)
			if not ok:
				raise RuntimeError(f'geofence (predicted): {why}')

			ok, why = inside(x_now, a.N, fence, slot_labels)
			if not ok:
				raise RuntimeError(f'geofence (measured): {why}')

			pwm.send(sw, x_pred, u, mapping, a.N, a.mass, a.g)
			log.write(k, ping, x_now, x_pred, xd=(a.xd[k, 0], a.xd[k, 3]))


# ---- Full flight ----

def fly(interval, note=''):
	generate(interval)
	a = arrays_mod.Arrays()
	print(a)
	fence = geofence()

	with Swarm() as sw:
		# Checks before anything spins
		sw.preflight_check()
		volts = sw.battery_check()
		sw.reset_estimators()

		# Start receiving state from every drone
		adapter = StateAdapter(sw)
		adapter.start()

		try:
			ground = assign_slots(sw, adapter, a)
			check_calibration(list(sw.live))

			input('\nProps will spin. Enter to arm, Ctrl-C to abort: ')
			arm(sw)
			takeoff(sw, adapter, ground)
			mission(sw, adapter, a, fence, note, volts)

			# Land where they finished
			print('\nLanding...')
			ramp(sw, adapter.positions(), LAND_Z, LAND_RATE)
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


# ---- Dry run ----

# Walks the mission with no hardware. The drones are assumed to stay on
# their starting targets, so the commands at pings 1+ are larger than in flight.
def dry(interval):
	generate(interval)
	a = arrays_mod.Arrays()
	print(a)

	fence = geofence()
	mapping = np.arange(a.N)
	x_meas = formation_start(a)
	ping = 0

	with Logger(a, mapping, path='logs/dry', note=f'dry run, {interval}s, PWM') as log:
		t_start = time.time()

		for k in range(0, a.n_steps, a.stride):
			wait_until(t_start + k * a.dt)

			# Track when a ping starts (ping 0 starts at k = 0)
			starting = (k == 0)
			if new_ping(a, k, ping):
				ping += 1
				starting = True

			x_pred = a.predict(k, ping, x_meas)
			u = a.control(k, ping, x_meas)

			# Print the PWM commands at the start of each ping
			if starting:
				print(f'\n  ping {ping}  t={k*a.dt:5.2f}s  (default thrust cal)')
				pwm.describe(u, mapping, a.N, a.mass, a.g)

			ok, why = inside(x_pred, a.N, fence)
			if not ok:
				print(f'  GEOFENCE t={k*a.dt:5.2f}s  {why}')

			log.write(k, ping, x_meas, x_pred, xd=(a.xd[k, 0], a.xd[k, 3]))


if __name__ == '__main__':
	args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
	interval = float(args[0]) if args else 5.0

	if '--dry' in sys.argv:
		dry(interval)
	else:
		fly(interval, note=f'{interval}s replan, send_setpoint (PWM)')
