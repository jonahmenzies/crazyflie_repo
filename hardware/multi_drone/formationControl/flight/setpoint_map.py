import numpy as np


def euler_to_quat(roll, pitch, yaw):
	"""Radians to quaternion (x, y, z, w). ZYX intrinsic, matching the firmware."""
	cr, sr = np.cos(roll  * 0.5), np.sin(roll  * 0.5)
	cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
	cy, sy = np.cos(yaw   * 0.5), np.sin(yaw   * 0.5)

	return (sr*cp*cy - cr*sp*sy,    # x
	        cr*sp*cy + sr*cp*sy,    # y
	        cr*cp*sy - sr*sp*cy,    # z
	        cr*cp*cy + sr*sp*sy)    # w


def unpack(x_pred, i):
	"""Slice drone i's ten elements out of the 50-vector.

	Order is [x, xd, theta, y, yd, phi, z, zd, psi, r] per params.
	"""
	b = i * 10
	pos = (x_pred[b + 0], x_pred[b + 3], x_pred[b + 6])
	vel = (x_pred[b + 1], x_pred[b + 4], x_pred[b + 7])
	att = (x_pred[b + 5], x_pred[b + 2], x_pred[b + 8])   # roll, pitch, yaw
	r   = x_pred[b + 9]
	return pos, vel, att, r


def send(swarm, x_pred, mapping):
	"""Send one full-state setpoint per drone.

	mapping[i] is the formation index for the drone at list position i,
	so drone i is told to fly formation slot mapping[i]'s predicted state.
	"""
	for i, scf in enumerate(swarm.scf):
		pos, vel, att, r = unpack(x_pred, mapping[i])
		qx, qy, qz, qw = euler_to_quat(*att)

		scf.cf.commander.send_full_state_setpoint(
			pos,
			vel,
			(0.0, 0.0, 0.0),          # acc, not in the model
			(qx, qy, qz, qw),
			0.0, 0.0,                 # rollrate, pitchrate, not in the model
			np.degrees(r))            # yawrate, firmware wants deg/s


def describe(x_pred, mapping):
	"""Print what would be sent. For dry runs without hardware."""
	print(f'{"list":>4} {"idx":>4}  {"x":>7} {"y":>7} {"z":>7}  '
	      f'{"vx":>6} {"vy":>6} {"vz":>6}  '
	      f'{"roll":>6} {"pitch":>6} {"yaw":>6}  {"r":>6}')
	for i, j in enumerate(mapping):
		pos, vel, att, r = unpack(x_pred, j)
		print(f'{i:4d} {j:4d}  '
		      f'{pos[0]:+7.3f} {pos[1]:+7.3f} {pos[2]:+7.3f}  '
		      f'{vel[0]:+6.2f} {vel[1]:+6.2f} {vel[2]:+6.2f}  '
		      f'{np.degrees(att[0]):+6.1f} {np.degrees(att[1]):+6.1f} '
		      f'{np.degrees(att[2]):+6.1f}  {np.degrees(r):+6.1f}')


if __name__ == '__main__':
	from arrays import Arrays

	a = Arrays()

	# start every drone exactly on its formation slot, at rest
	x0 = np.zeros(a.n)
	for i, off in enumerate(a.formation_offsets):
		t = a.xd0 + off
		x0[i*10 + 0] = t[0]
		x0[i*10 + 3] = t[1]
		x0[i*10 + 6] = t[2]

	mapping = np.arange(a.N)

	print('t = 0.00s  (should match the formation targets)')
	describe(a.predict(0, 0, x0), mapping)

	for k in (200, 600, 1200, 1800):
		print(f'\nt = {k * a.dt:.2f}s')
		describe(a.predict(k, 0, x0), mapping)

	q = euler_to_quat(0.0, 0.0, 0.0)
	assert np.allclose(q, (0, 0, 0, 1)), 'zero rotation should be identity quat'
	print('\nquaternion identity ok')
