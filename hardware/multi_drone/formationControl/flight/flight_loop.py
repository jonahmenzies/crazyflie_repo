# Flies the formation mission.
# Takeoff, hover and landing use position setpoints.
# The mission itself sends the paper's u with send_setpoint (see setpoint_map_pwm.py).
#
#     python3 flight/flight_loop.py 5               fly with a 5 s replan interval
#     python3 flight/flight_loop.py 5 --pos         send the predicted position instead of u
#     python3 flight/flight_loop.py 5 --no-reboot   same, without rebooting the drones first
#     python3 flight/flight_loop.py 5 --dry         no hardware, checks timing and geofence
#
# Two ways of driving the drones during the mission:
#   default (PWM)  the paper's u, as roll, pitch, yaw rate and thrust
#   --pos          the predicted state x_pred = c x_meas + e, sent as a
#                  position setpoint, so the Crazyflie's onboard position
#                  controller flies the drone along the predicted path
#
# Every flight reboots the drones first, then after the mission flies each
# drone back over the spot it took off from and lands it there.
#
# Run from formationControl/

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml
import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.utils.power_switch import PowerSwitch

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
SETTLE       = 2.0      # seconds to hover before measuring the hover thrust
HOVER_MEAS   = 3.0      # seconds the hover thrust is averaged over
HOVER_LIMIT  = 0.10     # abort if a live hover differs from THRUST by more than this
REBOOT_WAIT  = 5.0      # seconds for rebooted drones to boot, on top of Swarm's own wait
RETURN_SPEED = 0.25     # speed flying back to the start positions, m/s
RETURN_SEP   = 0.30     # land in place instead if two drones would get closer than this, m


# ---- Helpers ----

# Raised when a drone leaves the geofence during the mission. The flight then
# lands in place instead of cutting thrust, and the breach is saved next to
# the log as flight_<stamp>_breach.json.
class GeofenceBreach(RuntimeError):
	def __init__(self, kind, why, t):
		super().__init__(f'geofence ({kind}) at t={t:.2f}s: {why}')
		self.kind = kind      # 'predicted' or 'measured'
		self.why = why        # which drone and axis
		self.t = t            # mission time, s
		self.stamp = None     # log stamp, filled in by mission()

# Make sure arrays/ holds arrays for this replan interval and the current params.
# Rebuilds the generator first if params.cpp (or any source) changed.
# Reuses the arrays already there if they are newer than the generator and
# their interval divides this one, so fly 0.1 s first for each param set.
# Otherwise clears arrays/ and runs the generator.
def generate(interval):
	subprocess.run(['make', '-s', 'generate'], cwd='generator', check=True,
	               stdout=subprocess.DEVNULL)

	if arrays_mod.reusable('arrays', interval, 'generator/generate'):
		print(f'Reusing arrays for {interval}s replan interval')
		return

	for f in glob.glob('arrays/*.bin') + glob.glob('arrays/manifest.json'):
		os.remove(f)

	print(f'Generating arrays for {interval}s replan interval...')
	subprocess.run(['./generate', str(interval)], cwd='generator', check=True)


# Power cycle every drone in the config, so each flight starts from a fresh
# boot: estimator reset, no supervisor lock, decks re-initialised.
def reboot_all(config='config/uris.yaml'):
	with open(config) as f:
		uris = yaml.safe_load(f)['uris']

	cflib.crtp.init_drivers()
	print('Rebooting drones...')
	for uri in uris:
		ps = PowerSwitch(uri)
		try:
			ps.stm_power_cycle()
			print(f'  drone {label(uri):2d}  rebooting')
		except Exception as ex:
			print(f'  drone {label(uri):2d}  reboot failed ({type(ex).__name__})')
		finally:
			ps.close()

	print(f'Waiting {REBOOT_WAIT:.0f} s for them to boot...')
	time.sleep(REBOOT_WAIT)


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


# Hold every drone at its hover point for the given time
def hold_position(sw, hold, seconds):
	t0 = time.time()
	while time.time() - t0 < seconds:
		for i, scf in enumerate(sw.scf):
			scf.cf.commander.send_position_setpoint(hold[i][0], hold[i][1], TAKEOFF_Z, 0.0)
		time.sleep(0.05)


# Average each drone's own thrust command while it holds position. The
# onboard integral term settles on exactly the thrust this drone needs to
# hover on this battery right now, so the mission uses that instead of the
# stored hover PWM. Sets pwm.HOVER_LIVE and returns {uri: hover}.
def measure_hover(sw, hold):
	readings = [[] for _ in sw.scf]
	logs = []
	for i, scf in enumerate(sw.scf):
		log = LogConfig(name=f'Hover{i}', period_in_ms=100)
		log.add_variable('stabilizer.thrust', 'float')
		scf.cf.log.add_config(log)
		log.data_received_cb.add_callback(
			lambda ts, data, conf, i=i: readings[i].append(data['stabilizer.thrust']))
		logs.append(log)

	for log in logs:
		log.start()
	hold_position(sw, hold, HOVER_MEAS)

	# Remove the log blocks. Cleanup must never stop a flight, so errors are ignored.
	for log in logs:
		try:
			log.stop()
		except Exception:
			pass
	time.sleep(0.2)
	for log in logs:
		try:
			log.delete()
		except Exception:
			pass

	print('\nLive hover thrust:')
	hovers = {}
	for uri, r in zip(sw.live, readings):
		stored, _ = pwm.THRUST[uri]
		if len(r) < 10:
			print(f'  drone {label(uri):2d}  only {len(r)} readings, using stored {stored}')
			continue
		live = float(np.mean(r))
		change = (live - stored) / stored
		print(f'  drone {label(uri):2d}  live {live:6.0f}  stored {stored}  ({change:+.1%})')
		if abs(change) > HOVER_LIMIT:
			raise RuntimeError(f'drone {label(uri)} live hover {live:.0f} is '
			                   f'{change:+.0%} from stored {stored}')
		hovers[uri] = int(round(live))

	pwm.HOVER_LIVE.clear()
	pwm.HOVER_LIVE.update(hovers)
	return hovers


# Hold every drone at its own (x, y, z) for the given time
def hold_at(sw, points, seconds):
	t0 = time.time()
	while time.time() - t0 < seconds:
		for i, scf in enumerate(sw.scf):
			scf.cf.commander.send_position_setpoint(points[i][0], points[i][1], points[i][2], 0.0)
		time.sleep(0.05)


# Closest any two drones get if they all fly straight from start to end,
# leaving and arriving together
def closest_approach(start, end, samples=100):
	closest = np.inf
	for s in np.linspace(0.0, 1.0, samples + 1):
		p = start + s * (end - start)
		for i in range(len(p)):
			for j in range(i + 1, len(p)):
				closest = min(closest, np.linalg.norm(p[i] - p[j]))
	return closest


# Fly every drone back over the spot it took off from, keeping its current
# height, all arriving together. Returns where to land. If the straight
# paths would bring two drones within RETURN_SEP, lands them where they are.
def return_home(sw, adapter, ground):
	start = adapter.positions()
	end = np.column_stack([ground[:, 0], ground[:, 1], start[:, 2]])

	gap = closest_approach(start, end)
	if gap < RETURN_SEP:
		print(f'\nPaths back would come within {gap:.2f} m, landing in place')
		return start

	dist = np.linalg.norm(end - start, axis=1).max()
	steps = max(1, int(dist / RETURN_SPEED / 0.05))
	print(f'\nReturning to start positions: {dist:.2f} m, {steps * 0.05:.1f} s, '
	      f'closest pass {gap:.2f} m')

	for s in range(1, steps + 1):
		p = start + (s / steps) * (end - start)
		for i, scf in enumerate(sw.scf):
			scf.cf.commander.send_position_setpoint(p[i][0], p[i][1], p[i][2], 0.0)
		time.sleep(0.05)

	hold_at(sw, end, 1.0)
	return end


# Climb to TAKEOFF_Z, settle, then measure each drone's hover thrust
def takeoff(sw, adapter, ground):
	print(f'Climbing to {TAKEOFF_Z:.2f} m...')
	ramp(sw, ground, TAKEOFF_Z, TAKEOFF_RATE)

	hold = adapter.positions()
	hold_position(sw, hold, SETTLE)
	return measure_hover(sw, hold)


# Send every drone its predicted position (x, y, z and yaw from x_pred).
# The Crazyflie's PID position controller is proportional only in x and y
# (posCtlPid.xKp = 2, no integral, no velocity feedforward), so chasing a
# moving setpoint it settles v / Kp behind it: 0.1 m at 0.2 m/s. Leading the
# x, y setpoint by the predicted velocity / Kp cancels that lag. z has an
# integral term onboard and barely moves, so it gets no lead.
POS_KP = 2.0      # posCtlPid.xKp and yKp, firmware default

def send_positions(sw, x_pred, N):
	for i, scf in enumerate(sw.scf):
		b = i * 10
		x = x_pred[b + 0] + x_pred[b + 1] / POS_KP
		y = x_pred[b + 3] + x_pred[b + 4] / POS_KP
		scf.cf.commander.send_position_setpoint(
			float(x), float(y), float(x_pred[b + 6]), float(np.degrees(x_pred[b + 8])))


# The formation mission: send the paper's u (or, with position=True, the
# predicted position) every tick until the end
def mission(sw, adapter, a, fence, note, volts, position=False):
	mapping = np.arange(a.N)    # after the reorder, slot i is list position i
	slot_uris = list(sw.live)
	slot_labels = sw.labels()
	ticks = range(0, a.n_steps, a.stride)

	x_meas = adapter.read()     # starting state for ping 0
	ping = 0
	how = 'position setpoints (x_pred)' if position else 'send_setpoint (PWM)'
	print(f'\nMission: {len(ticks)} ticks, {a.n_pings} pings, {a.tT:.2f}s, {how}\n')

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
				breach = GeofenceBreach('predicted', why, k * a.dt)
				breach.stamp = log.stamp
				raise breach

			ok, why = inside(x_now, a.N, fence, slot_labels)
			if not ok:
				breach = GeofenceBreach('measured', why, k * a.dt)
				breach.stamp = log.stamp
				raise breach

			if position:
				send_positions(sw, x_pred, a.N)
			else:
				pwm.send(sw, x_pred, u, mapping, a.N, a.mass, a.g)
			log.write(k, ping, x_now, x_pred, xd=a.xd[k])


# ---- Full flight ----

def fly(interval, note='', reboot=True, position=False):
	generate(interval)
	a = arrays_mod.Arrays(interval=interval)
	print(a)
	fence = geofence()

	if reboot:
		reboot_all()

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
			hovers = takeoff(sw, adapter, ground)
			hover_note = ', '.join(f'{label(u)}:{h:.0f}' for u, h in hovers.items())
			mission(sw, adapter, a, fence, f'{note}, live hover {hover_note}', volts,
			        position=position)

			# Fly back over the take-off spots and land there
			home = return_home(sw, adapter, ground)
			print('Landing...')
			ramp(sw, home, LAND_Z, LAND_RATE)
			sw.stop_all()

		except KeyboardInterrupt:
			print('\nAborted, cutting thrust')
			sw.stop_all()
		except GeofenceBreach as ex:
			# Save what happened, then come straight down where they are
			print(f'\n{ex}')
			with open(f'logs/flight_{ex.stamp}_breach.json', 'w') as f:
				json.dump({'kind': ex.kind, 'why': ex.why, 't': ex.t}, f, indent=2)
			print('Landing in place...')
			here = adapter.positions()
			hold_at(sw, here, 0.5)
			ramp(sw, here, LAND_Z, LAND_RATE)
			sw.stop_all()
		except RuntimeError as ex:
			print(f'\n{ex}, cutting thrust')
			sw.stop_all()
			raise
		except Exception:
			print('\nUnexpected error, cutting thrust')
			sw.stop_all()
			raise
		finally:
			adapter.stop()


# ---- Dry run ----

# Walks the mission with no hardware. The drones are assumed to stay on
# their starting targets, so the commands at pings 1+ are larger than in flight.
def dry(interval):
	generate(interval)
	a = arrays_mod.Arrays(interval=interval)
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

			log.write(k, ping, x_meas, x_pred, xd=a.xd[k])


if __name__ == '__main__':
	args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
	interval = float(args[0]) if args else 5.0

	if '--dry' in sys.argv:
		dry(interval)
	else:
		position = '--pos' in sys.argv
		how = 'position setpoint' if position else 'send_setpoint (PWM)'
		fly(interval, note=f'{interval}s replan, {how}',
		    reboot='--no-reboot' not in sys.argv, position=position)
