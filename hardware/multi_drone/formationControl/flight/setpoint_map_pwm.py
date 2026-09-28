import numpy as np

from swarm import label

# Sends the paper's u to each drone with send_setpoint(roll, pitch, yawrate, thrust).
# The Crazyflie takes thrust as a PWM count, not newtons, so each drone needs
# its own calibration. That calibration is about as noisy as u_z itself, so
# expect some vertical drift.


# ---- Thrust calibration ----

# (hover_pwm, slope) per drone, measured with safety_check/thrust_id.py
# slope is newtons per PWM count
THRUST = {
	'radio://0/10/2M/E7E7E7E701':  (37253, 5.773e-06),
	'radio://0/60/2M/E7E7E7E706':  (38742, 6.332e-06),
	'radio://1/80/2M/E7E7E7E708':  (42333, 5.397e-06),
	'radio://1/90/2M/E7E7E7E709':  (39217, 4.437e-06),
	'radio://1/100/2M/E7E7E7E710': (47501, 4.221e-06),
}
THRUST_DEFAULT = (40000, 5.0e-06)    # used if a drone has no calibration
HOVER_N = 0.26487                    # hover thrust, mass * g, newtons

# ---- Limits (same as inclinationProtection in computeControl.cpp) ----

TILT_MAX = np.pi / 4.0          # max pitch and roll, rad
YAWRATE_MAX = 2.0 * np.pi       # max yaw rate, rad/s


# Get a drone's (hover_pwm, slope), warning if it has no calibration
def thrust_cal(uri):
	if uri not in THRUST:
		print(f'  WARNING: no thrust calibration for drone {label(uri)} '
		      f'({uri}), using {THRUST_DEFAULT}')
	return THRUST.get(uri, THRUST_DEFAULT)


# Clamp every drone's u to the limits above. Returns a new array.
# Each drone's u is [pitch, roll, u_z, yawrate]
def inclination_protection(u, N, mass, g):
	u = u.copy()
	for i in range(N):
		b = i * 4
		u[b + 0] = np.clip(u[b + 0], -TILT_MAX, TILT_MAX)          # pitch
		u[b + 1] = np.clip(u[b + 1], -TILT_MAX, TILT_MAX)          # roll
		u[b + 2] = np.clip(u[b + 2], -mass * g, 2.0 * mass * g)    # thrust
		u[b + 3] = np.clip(u[b + 3], -YAWRATE_MAX, YAWRATE_MAX)    # yaw rate
	return u


# Convert total thrust in newtons to a PWM count (0 to 65535).
# Starts from the drone's measured hover PWM and adds the change in force.
def thrust_to_pwm(F_total, pwm_hover, slope):
	pwm = pwm_hover + (F_total - HOVER_N) / slope
	return int(np.clip(pwm, 0, 65535))


# Drone i's control inputs in the units the firmware wants:
# roll, pitch, yawrate in degrees, thrust as PWM.
# u_z is a net force (z acceleration = -u_z / mass), so total thrust = mass*g - u_z
def unpack_u(u, i, mass, g, pwm_hover, slope):
	b = i * 4
	pitch = np.degrees(u[b + 0])
	roll = np.degrees(u[b + 1])
	F_total = mass * g - u[b + 2]
	yawrate = np.degrees(u[b + 3])
	return roll, pitch, yawrate, thrust_to_pwm(F_total, pwm_hover, slope)


# Send one command to every drone. x_pred is not used, it is only here so
# the arguments match what flight_loop passes.
def send(swarm, x_pred, u, mapping, N, mass, g):
	u = inclination_protection(u, N, mass, g)
	for i, scf in enumerate(swarm.scf):
		pwm_hover, slope = thrust_cal(scf.cf.link_uri)
		roll, pitch, yawrate, thrust = unpack_u(u, mapping[i], mass, g, pwm_hover, slope)
		scf.cf.commander.send_setpoint(roll, pitch, yawrate, thrust)


# Print what send() would send, without any hardware
def describe(u, mapping, N, mass, g, uris=None):
	u = inclination_protection(u, N, mass, g)

	print(f'{"slot":>4}  {"roll":>7} {"pitch":>7} {"yawrate":>8}  '
	      f'{"F (N)":>8} {"hover":>6} {"pwm":>7} {"d_pwm":>7}')

	for i, j in enumerate(mapping):
		if uris:
			pwm_hover, slope = thrust_cal(uris[i])
		else:
			pwm_hover, slope = THRUST_DEFAULT

		roll, pitch, yawrate, thrust = unpack_u(u, j, mass, g, pwm_hover, slope)
		F = mass * g - u[j * 4 + 2]

		print(f'{i:4d}  {roll:+7.2f} {pitch:+7.2f} {yawrate:+8.2f}  '
		      f'{F:8.5f} {pwm_hover:6d} {thrust:7d} {thrust - pwm_hover:+7d}')


# Self test: python3 flight/setpoint_map_pwm.py (run from formationControl/)
if __name__ == '__main__':
	from arrays import Arrays

	a = Arrays()
	uris = list(THRUST.keys())

	# Every drone starts exactly on its formation target
	x0 = np.zeros(a.n)
	for i, offset in enumerate(a.formation_offsets):
		x, y, z = a.xd0 + offset
		x0[i*10 + 0] = x
		x0[i*10 + 3] = y
		x0[i*10 + 6] = z

	mapping = np.arange(a.N)

	# Commands at a few times during the first ping, stopping past its end
	for k in (0, 400, 1000, 1800):
		try:
			print(f'\nt = {k * a.dt:.2f}s')
			describe(a.control(k, 0, x0), mapping, a.N, a.mass, a.g, uris)
		except IndexError:
			break

	# Each drone's own hover force should give back its own hover PWM
	print('\nhover check (each should return its own hover pwm):')
	for uri, (pwm, slope) in THRUST.items():
		out = thrust_to_pwm(a.mass * a.g, pwm, slope)
		flag = 'ok' if abs(out - pwm) <= 1 else 'MISMATCH'
		print(f'  drone {label(uri):2d}  {pwm} -> {out}  {flag}')
