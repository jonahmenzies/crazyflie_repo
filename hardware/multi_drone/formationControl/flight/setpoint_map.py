import numpy as np

# Full-state setpoints. The firmware closes the position loop and does its
# own thrust conversion, so no PWM calibration is needed here. The control
# input u enters as acceleration feedforward on the z channel.
#
# Note this is the least faithful of the three commander options: the
# firmware regenerates its own attitude from position error, so u[0]
# (pitch), u[1] (roll) and u[3] (yaw rate) are computed and discarded.
# send_zdistance_setpoint applies three of the four channels directly.
#
# From Ai in params.cpp:
#     xdd =  g * theta        (row 1)
#     ydd = -g * phi          (row 4)
#     zdd = -u_z / mass       (row 7)
#
# zdd has no gravity term, so u_z is the deviation from hover, not total
# thrust. The firmware handles gravity itself.

# Matches inclinationProtection in computeControl.cpp
TILT_MAX    = np.pi / 4.0
YAWRATE_MAX = 2.0 * np.pi

# cflib packs every full-state field as int16 after scaling by 1000, so the
# magnitude ceiling is 32.767 in whatever unit the field uses — metres,
# m/s, m/s^2, deg/s. Clamp rather than let struct.error kill a flight
# mid-air. Hitting this means the prediction has already diverged, so the
# warning matters more than the clamp.
INT16_LIMIT = 32.7


def inclination_protection(u, N, mass, g):
	"""Clamp u. Same limits as the C++ sim."""
	u = u.copy()
	for i in range(N):
		b = i * 4
		u[b + 0] = np.clip(u[b + 0], -TILT_MAX, TILT_MAX)          # pitch
		u[b + 1] = np.clip(u[b + 1], -TILT_MAX, TILT_MAX)          # roll
		u[b + 2] = np.clip(u[b + 2], -mass * g, 2.0 * mass * g)    # thrust
		u[b + 3] = np.clip(u[b + 3], -YAWRATE_MAX, YAWRATE_MAX)    # yaw rate
	return u


def euler_to_quat(roll, pitch, yaw):
	"""Radians to quaternion (x, y, z, w). ZYX intrinsic, matching firmware."""
	cr, sr = np.cos(roll  * 0.5), np.sin(roll  * 0.5)
	cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
	cy, sy = np.cos(yaw   * 0.5), np.sin(yaw   * 0.5)

	return (sr*cp*cy - cr*sp*sy,    # x
	        cr*sp*cy + sr*cp*sy,    # y
	        cr*cp*sy - sr*sp*cy,    # z
	        cr*cp*cy + sr*sp*sy)    # w


def unpack(x_pred, u, i, mass, g):
	"""Slice drone i's predicted state and control into firmware units.

	State order per agent is [x, xd, theta, y, yd, phi, z, zd, psi, r].
	Control order is [pitch, roll, u_z, yawrate].
	"""
	b = i * 10
	c = i * 4

	pos = (x_pred[b + 0], x_pred[b + 3], x_pred[b + 6])
	vel = (x_pred[b + 1], x_pred[b + 4], x_pred[b + 7])
	att = (x_pred[b + 5], x_pred[b + 2], x_pred[b + 8])   # roll, pitch, yaw

	acc = ( g * x_pred[b + 2],       # xdd =  g * theta
	       -g * x_pred[b + 5],       # ydd = -g * phi
	       -u[c + 2] / mass)         # zdd = -u_z / mass

	yawrate = x_pred[b + 9]
	return pos, vel, acc, att, yawrate


def _clamp(v, name, warned):
	if abs(v) > INT16_LIMIT:
		if name not in warned:
			print(f'  clamped {name}: {v:+.2f} exceeds int16 range')
			warned.add(name)
		return float(np.clip(v, -INT16_LIMIT, INT16_LIMIT))
	return v


def send(swarm, x_pred, u, mapping, N, mass, g, _warned=set()):
	"""Send one full-state setpoint per drone.

	mapping[i] is the formation index for the drone at list position i.
	"""
	u = inclination_protection(u, N, mass, g)
	for i, scf in enumerate(swarm.scf):
		pos, vel, acc, att, yawrate = unpack(x_pred, u, mapping[i], mass, g)
		qx, qy, qz, qw = euler_to_quat(*att)

		pos = tuple(_clamp(v, f'pos{i}', _warned) for v in pos)
		vel = tuple(_clamp(v, f'vel{i}', _warned) for v in vel)
		acc = tuple(_clamp(v, f'acc{i}', _warned) for v in acc)
		yr  = _clamp(np.degrees(yawrate), f'yawrate{i}', _warned)

		scf.cf.commander.send_full_state_setpoint(
			pos,
			vel,
			acc,
			(qx, qy, qz, qw),
			0.0, 0.0,                 # rollrate, pitchrate, not in the model
			yr)


def describe(x_pred, u, mapping, N, mass, g):
	"""Print what would be sent. For dry runs without hardware."""
	u = inclination_protection(u, N, mass, g)
	print(f'{"list":>4} {"idx":>4}  {"x":>7} {"y":>7} {"z":>7}  '
	      f'{"vx":>6} {"vy":>6} {"vz":>6}  '
	      f'{"ax":>6} {"ay":>6} {"az":>6}  '
	      f'{"roll":>6} {"pitch":>6} {"yaw":>6}')
	for i, j in enumerate(mapping):
		pos, vel, acc, att, yawrate = unpack(x_pred, u, j, mass, g)
		print(f'{i:4d} {j:4d}  '
		      f'{pos[0]:+7.3f} {pos[1]:+7.3f} {pos[2]:+7.3f}  '
		      f'{vel[0]:+6.2f} {vel[1]:+6.2f} {vel[2]:+6.2f}  '
		      f'{acc[0]:+6.2f} {acc[1]:+6.2f} {acc[2]:+6.2f}  '
		      f'{np.degrees(att[0]):+6.1f} {np.degrees(att[1]):+6.1f} '
		      f'{np.degrees(att[2]):+6.1f}')


if __name__ == '__main__':
	from arrays import Arrays

	a = Arrays()

	x0 = np.zeros(a.n)
	for i, off in enumerate(a.formation_offsets):
		t = a.xd0 + off
		x0[i*10 + 0] = t[0]
		x0[i*10 + 3] = t[1]
		x0[i*10 + 6] = t[2]

	mapping = np.arange(a.N)

	print('t = 0.00s  (positions should match the formation targets)')
	describe(a.predict(0, 0, x0), a.control(0, 0, x0),
	         mapping, a.N, a.mass, a.g)

	for k in (400, 1000, 1800):
		try:
			print(f'\nt = {k * a.dt:.2f}s')
			describe(a.predict(k, 0, x0), a.control(k, 0, x0),
			         mapping, a.N, a.mass, a.g)
		except IndexError:
			break

	q = euler_to_quat(0.0, 0.0, 0.0)
	assert np.allclose(q, (0, 0, 0, 1)), 'zero rotation should be identity quat'
	print('\nquaternion identity ok')
