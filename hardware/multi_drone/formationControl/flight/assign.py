import numpy as np
from scipy.optimize import linear_sum_assignment


# Where each formation slot should be at t = 0 (reference start + offset)
def targets(arrays):
	return arrays.xd0[None, :] + arrays.formation_offsets


# Match drones to slots so the total distance is as small as possible.
# measured:    (N, 3) drone positions, in list order
# targets_xyz: (N, 3) slot positions
# Returns (mapping, distances). mapping[i] is the slot for the drone at list position i.
def solve(measured, targets_xyz):
	# cost[i, j] = distance from drone i to slot j
	cost = np.linalg.norm(measured[:, None, :] - targets_xyz[None, :, :], axis=2)

	# Hungarian algorithm picks the cheapest one-to-one matching
	rows, cols = linear_sum_assignment(cost)
	return cols, cost[rows, cols]


# Assign each drone to a formation slot and print the result.
# xy_only:  ignore height, use this on the ground where every drone sits at z ~ 0
# labels:   drone numbers from Swarm.labels(), only used to name drones in the table
# Returns mapping, where mapping[i] is the slot for the drone at list position i.
def assign(measured, arrays, xy_only=False, labels=None):
	slot_positions = targets(arrays)

	# Copies used for matching, with height zeroed if xy_only
	match_measured = measured.copy()
	match_targets = slot_positions.copy()
	if xy_only:
		match_measured[:, 2] = 0.0
		match_targets[:, 2] = 0.0

	mapping, dists = solve(match_measured, match_targets)

	# Print the table header
	axes = 'xy' if xy_only else 'xyz'
	head = 'drone' if labels is not None else 'list'
	print(f'Assignment ({axes}):')
	print(f'{head:>5} {"->":>3} {"slot":>5}  '
	      f'{"measured":>22}  {"target":>22}  {"dist":>6}')

	# One row per drone
	for i in range(len(mapping)):
		slot = mapping[i]
		mx, my, mz = measured[i]
		tx, ty, tz = slot_positions[slot]
		who = labels[i] if labels is not None else i

		print(f'{who:5d} {"->":>3} {slot:5d}  '
		      f'({mx:+6.2f},{my:+6.2f},{mz:+6.2f})  '
		      f'({tx:+6.2f},{ty:+6.2f},{tz:+6.2f})  '
		      f'{dists[i]:6.3f}')

	print(f'  total {dists.sum():.3f} m')
	return mapping


# Self test: python3 flight/assign.py (run from formationControl/)
# Shuffles the drones and checks the assignment finds the original order
if __name__ == '__main__':
	from arrays import Arrays

	a = Arrays()
	slot_positions = targets(a)

	print('formation targets at t=0:')
	for i, (x, y, z) in enumerate(slot_positions):
		print(f'  {i}: ({x:+.2f}, {y:+.2f}, {z:+.2f})')

	rng = np.random.default_rng(0)
	perm = rng.permutation(len(slot_positions))

	# Test 1: drones in the air, shuffled with a little position noise
	fake = slot_positions[perm] + rng.normal(0, 0.03, slot_positions.shape)
	print(f'\n3D: shuffled by {perm}')
	mapping = assign(fake, a)
	assert np.array_equal(mapping, perm), 'did not recover the permutation'

	# Test 2: same drones sitting on the floor
	ground = fake.copy()
	ground[:, 2] = rng.normal(0.02, 0.005, len(slot_positions))
	print(f'\nxy_only: same drones sitting on the floor')
	mapping = assign(ground, a, xy_only=True)
	assert np.array_equal(mapping, perm), 'did not recover the permutation'

	# Test 3: table with drone numbers instead of list positions
	print('\nlabelled table:')
	assign(ground, a, xy_only=True, labels=[1, 6, 8, 9])

	print('\nok')
