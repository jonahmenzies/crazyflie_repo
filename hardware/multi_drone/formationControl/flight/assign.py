import numpy as np
from scipy.optimize import linear_sum_assignment


def targets(arrays):
	"""Where each formation index should be at t=0: xd0 + offset."""
	return arrays.xd0[None, :] + arrays.formation_offsets


def solve(measured, targets_xyz):
	"""Hungarian match of measured positions to formation slots.

	measured    (N,3) positions in list order
	targets_xyz (N,3) where each index should be

	Returns (mapping, distances, cost) where mapping[i] is the formation
	index assigned to the drone at list position i.
	"""
	cost = np.linalg.norm(measured[:, None, :] - targets_xyz[None, :, :], axis=2)
	row, col = linear_sum_assignment(cost)
	return col, cost[row, col], cost


def ambiguity(cost, mapping):
	"""Ratio of second-best total cost to best. Near 1.0 means a coin flip.

	Forbids the cheapest pair of the winning assignment and re-solves; if
	the alternative costs almost the same, two drones are close enough that
	the solver's choice between them is arbitrary.
	"""
	best = cost[np.arange(len(mapping)), mapping].sum()

	dists = cost[np.arange(len(mapping)), mapping]
	i = int(np.argmin(dists))

	blocked = cost.copy()
	blocked[i, mapping[i]] = 1e6
	_, alt_col = linear_sum_assignment(blocked)
	alt = cost[np.arange(len(mapping)), alt_col].sum()

	return alt / best if best > 1e-9 else 1.0


def assign(measured, arrays, max_dist=0.3, min_ratio=1.15,
           verbose=True, xy_only=False):
	"""Full assignment with pre-arm guards. Raises if unsafe.

	xy_only ignores height when matching. Use it on the ground, where every
	drone sits at z near zero and the formation's z offsets would otherwise
	add a constant error to every pairing.

	Returns mapping, where mapping[i] is the formation index for the drone
	at list position i.
	"""
	tgt = targets(arrays)

	m, t = measured.copy(), tgt.copy()
	if xy_only:
		m[:, 2] = 0.0
		t[:, 2] = 0.0

	mapping, dists, cost = solve(m, t)
	ratio = ambiguity(cost, mapping)

	if verbose:
		axes = 'xy' if xy_only else 'xyz'
		print(f'Assignment ({axes}):')
		print(f'{"list":>4} {"->":>3} {"index":>5}  '
		      f'{"measured":>22}  {"target":>22}  {"dist":>6}')
		for i in range(len(mapping)):
			j = mapping[i]
			mm, tt = measured[i], tgt[j]
			print(f'{i:4d} {"->":>3} {j:5d}  '
			      f'({mm[0]:+6.2f},{mm[1]:+6.2f},{mm[2]:+6.2f})  '
			      f'({tt[0]:+6.2f},{tt[1]:+6.2f},{tt[2]:+6.2f})  '
			      f'{dists[i]:6.3f}')
		print(f'  total {dists.sum():.3f} m   ambiguity ratio {ratio:.3f}')

	far = [(i, d) for i, d in enumerate(dists) if d > max_dist]
	if far:
		raise RuntimeError(
			'drones too far from assigned slots: ' +
			', '.join(f'list {i} at {d:.2f}m' for i, d in far))

	if ratio < min_ratio:
		raise RuntimeError(
			f'assignment ambiguous (ratio {ratio:.3f} < {min_ratio}). '
			'Two drones are close enough that the match is arbitrary. '
			'Move them further apart.')

	return mapping


if __name__ == '__main__':
	from arrays import Arrays

	a = Arrays()
	tgt = targets(a)
	print('formation targets at t=0:')
	for i, t in enumerate(tgt):
		print(f'  {i}: ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f})')

	rng = np.random.default_rng(0)
	perm = rng.permutation(len(tgt))

	# full 3D, drones already at their slots
	fake = tgt[perm] + rng.normal(0, 0.03, tgt.shape)
	print(f'\n3D: shuffled by {perm}')
	mapping = assign(fake, a)
	assert np.array_equal(mapping, perm), 'did not recover the permutation'

	# on the ground, z near zero for everything
	ground = fake.copy()
	ground[:, 2] = rng.normal(0.02, 0.005, len(tgt))
	print(f'\nxy_only: same drones sitting on the floor')
	mapping = assign(ground, a, max_dist=0.4, xy_only=True)
	assert np.array_equal(mapping, perm), 'did not recover the permutation'

	print('\nok')
