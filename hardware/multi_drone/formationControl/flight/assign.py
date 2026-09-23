import numpy as np
from scipy.optimize import linear_sum_assignment


def targets(arrays):
	"""Where each formation index should be at t=0: xd0 + offset."""
	return arrays.xd0[None, :] + arrays.formation_offsets


def solve(measured, targets_xyz):
	"""Hungarian match of measured positions to formation slots.

	measured    (N,3) positions in list order
	targets_xyz (N,3) where each index should be

	Returns (mapping, distances) where mapping[i] is the formation index
	assigned to the drone at list position i.
	"""
	cost = np.linalg.norm(measured[:, None, :] - targets_xyz[None, :, :], axis=2)
	row, col = linear_sum_assignment(cost)
	return col, cost[row, col]


def assign(measured, arrays, xy_only=False, labels=None):
	"""Hungarian assignment of drones to formation slots.

	xy_only ignores height when matching. Use it on the ground, where every
	drone sits at z near zero and the formation's z offsets would otherwise
	add a constant error to every pairing.

	labels is the physical drone number per list position, from
	Swarm.labels(). Used only for the printed table. List position is
	whatever order the radios connected in and is discarded by the reorder
	that follows this call, so naming the airframe is the only way to tell
	later which hardware flew which slot. Omit it and the table falls back
	to list index, which is what the self-test below does.

	Returns mapping, where mapping[i] is the formation index for the drone
	at list position i.
	"""
	tgt = targets(arrays)

	m, t = measured.copy(), tgt.copy()
	if xy_only:
		m[:, 2] = 0.0
		t[:, 2] = 0.0

	mapping, dists = solve(m, t)

	axes = 'xy' if xy_only else 'xyz'
	head = 'drone' if labels is not None else 'list'
	print(f'Assignment ({axes}):')
	print(f'{head:>5} {"->":>3} {"slot":>5}  '
	      f'{"measured":>22}  {"target":>22}  {"dist":>6}')
	for i in range(len(mapping)):
		j = mapping[i]
		mm, tt = measured[i], tgt[j]
		who = labels[i] if labels is not None else i
		print(f'{who:5d} {"->":>3} {j:5d}  '
		      f'({mm[0]:+6.2f},{mm[1]:+6.2f},{mm[2]:+6.2f})  '
		      f'({tt[0]:+6.2f},{tt[1]:+6.2f},{tt[2]:+6.2f})  '
		      f'{dists[i]:6.3f}')
	print(f'  total {dists.sum():.3f} m')

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

	fake = tgt[perm] + rng.normal(0, 0.03, tgt.shape)
	print(f'\n3D: shuffled by {perm}')
	mapping = assign(fake, a)
	assert np.array_equal(mapping, perm), 'did not recover the permutation'

	ground = fake.copy()
	ground[:, 2] = rng.normal(0.02, 0.005, len(tgt))
	print(f'\nxy_only: same drones sitting on the floor')
	mapping = assign(ground, a, xy_only=True)
	assert np.array_equal(mapping, perm), 'did not recover the permutation'

	print('\nlabelled table:')
	assign(ground, a, xy_only=True, labels=[1, 6, 8, 9])

	print('\nok')
