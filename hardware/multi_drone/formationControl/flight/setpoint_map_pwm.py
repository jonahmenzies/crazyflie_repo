import numpy as np

from swarm import label

# send_setpoint(roll, pitch, yawrate, thrust). This is the paper's u
# applied exactly: u = (theta_d, phi_d, delta_F_d, r_d) goes straight to
# the inner-loop attitude controller that a_theta, a_phi and a_r describe.
# Nothing from x_pred is used.
#
# The cost is the thrust channel. The Crazyflie takes thrust as a uint16
# PWM count with no force interface, so the newton output of the control
# law has to be converted using a per-drone calibration. u_z spans about
# +/-0.01 N (3.5% of hover) while hover PWM repeats to only 0.6-1.3% within
# a session and drifts more across sessions — so the calibration noise is
# the same order as the control signal. Expect vertical drift.

# (hover_pwm, slope) per drone, measured with saftey_check/thrust_id.py.
# slope is dF/dPWM in newtons per count.
THRUST = {
	'radio://0/10/2M/E7E7E7E701':  (37253, 5.773e-06),
	'radio://0/60/2M/E7E7E7E706':  (38742, 6.332e-06),
	'radio://1/80/2M/E7E7E7E708':  (42333, 5.397e-06),
	'radio://1/90/2M/E7E7E7E709':  (39217, 4.437e-06),
	'radio://1/100/2M/E7E7E7E710': (47501, 4.221e-06),
}
THRUST_DEFAULT = (40000, 5.0e-06)
HOVER_N        = 0.26487        # mass * g

# Matches inclinationProtection in computeControl.cpp
TILT_MAX    = np.pi / 4.0
YAWRATE_MAX = 2.0 * np.pi


def thrust_cal(uri):
	if uri not in THRUST:
		print(f'  WARNING: no thrust calibration for drone {label(uri)} '
		      f'({uri}), using {THRUST_DEFAULT}')
	return THRUST.get(uri, THRUST_DEFAULT)


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


def thrust_to_pwm(F_total, pwm_hover, slope):
	"""Total body thrust in newtons -> Crazyflie thrust command (uint16).

	Anchored at the drone's measured hover point, with its own measured
	slope for deviations. Anchoring rather than using the raw fit intercept
	keeps hover exact, which matters because ~96% of the commanded force is
	the constant hover term.
	"""
	pwm = pwm_hover + (F_total - HOVER_N) / slope
	return int(np.clip(pwm, 0, 65535))


def unpack_u(u, i, mass, g, pwm_hover, slope):
	"""Drone i's four control inputs in firmware units.

	Control order per agent is [pitch, roll, u_z, yawrate], radians.
	Row 7 of Ai has no gravity term, so u_z is a net force with
	zdd = -u_z/mass, making total body thrust F = mass*g - u_z.
	"""
	b = i * 4
	pitch   = np.degrees(u[b + 0])
	roll    = np.degrees(u[b + 1])
	F_total = mass * g - u[b + 2]
	yawrate = np.degrees(u[b + 3])
	return roll, pitch, yawrate, thrust_to_pwm(F_total, pwm_hover, slope)


def send(swarm, x_pred, u, mapping, N, mass, g):
	"""Send one attitude+thrust command per drone.

	x_pred is accepted for signature compatibility with the other mappers
	but is not used — every channel comes from u.
	"""
	u = inclination_protection(u, N, mass, g)
	for i, scf in enumerate(swarm.scf):
		pwm_hover, slope = thrust_cal(scf.cf.link_uri)
		roll, pitch, yawrate, thrust = unpack_u(
			u, mapping[i], mass, g, pwm_hover, slope)
		scf.cf.commander.send_setpoint(roll, pitch, yawrate, thrust)


def describe(u, mapping, N, mass, g, uris=None):
	"""Print what would be sent. For checks without hardware."""
	u = inclination_protection(u, N, mass, g)
	print(f'{"slot":>4}  {"roll":>7} {"pitch":>7} {"yawrate":>8}  '
	      f'{"F (N)":>8} {"hover":>6} {"pwm":>7} {"d_pwm":>7}')
	for i, j in enumerate(mapping):
		pwm_hover, slope = thrust_cal(uris[i]) if uris else THRUST_DEFAULT
		roll, pitch, yawrate, thrust = unpack_u(
			u, j, mass, g, pwm_hover, slope)
		F = mass * g - u[j * 4 + 2]
		print(f'{i:4d}  {roll:+7.2f} {pitch:+7.2f} {yawrate:+8.2f}  '
		      f'{F:8.5f} {pwm_hover:6d} {thrust:7d} {thrust - pwm_hover:+7d}')


if __name__ == '__main__':
	from arrays import Arrays

	a = Arrays()
	uris = list(THRUST.keys())

	x0 = np.zeros(a.n)
	for i, off in enumerate(a.formation_offsets):
		t = a.xd0 + off
		x0[i*10 + 0] = t[0]
		x0[i*10 + 3] = t[1]
		x0[i*10 + 6] = t[2]

	mapping = np.arange(a.N)

	for k in (0, 400, 1000, 1800):
		try:
			print(f'\nt = {k * a.dt:.2f}s')
			describe(a.control(k, 0, x0), mapping, a.N, a.mass, a.g, uris)
		except IndexError:
			break

	print('\nhover check (each should return its own hover pwm):')
	for uri, (pwm, slope) in THRUST.items():
		out  = thrust_to_pwm(a.mass * a.g, pwm, slope)
		flag = 'ok' if abs(out - pwm) <= 1 else 'MISMATCH'
		print(f'  drone {label(uri):2d}  {pwm} -> {out}  {flag}')
