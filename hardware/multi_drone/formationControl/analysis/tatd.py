# TATD of a flight: RMS distance of each drone from its reference over the
# last part of the flight (Jiang et al. 2020, Eq. 11). Same formula as the
# sim's computeTATD20. Scored for the measured and the predicted state.
#
#     python3 analysis/tatd.py logs/flight_<stamp>_full.csv [frac]     frac defaults to 0.2
#
# Run from formationControl/

import re
import sys

import numpy as np

path = sys.argv[1]
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2    # fraction of the flight scored

# Read the CSV by column name. Empty cells become NaN.
data = np.genfromtxt(path, delimiter=',', names=True)

# Number of drones, from the reference columns (rx0, rx1, ...)
N = sum(1 for c in data.dtype.names if re.fullmatch(r'rx\d+', c))
if N == 0:
	sys.exit(f'{path} has no reference columns (rx0, ...). '
	         f'Only logs written after they were added can be scored.')

# Window: the last FRAC of the rows (rounded like the C++ std::lround)
n_rows = len(data)
start = int(np.floor((1.0 - FRAC) * n_rows + 0.5))
window = data[start:]


# TATD for one state: 'm' = measured, 'p' = predicted
def tatd(tag):
	# Squared 3D position error, summed over every drone, one value per row
	err_sq = np.zeros(len(window))
	for i in range(N):
		for axis in 'xyz':
			err_sq += (window[f'{tag}{axis}{i}'] - window[f'r{axis}{i}'])**2

	# Only rows where this state was logged
	err_sq = err_sq[~np.isnan(err_sq)]
	if len(err_sq) == 0:
		return float('nan')
	return np.sqrt(err_sq.sum() / (N * len(err_sq)))


print(path.split('/')[-1])
print(f'  window: last {FRAC:.0%} of {n_rows} rows, '
      f't = {window["t"][0]:.2f} to {window["t"][-1]:.2f} s')
print(f'  TATD{round(FRAC*100)} measured   {tatd("m"):.4f} m')
print(f'  TATD{round(FRAC*100)} predicted  {tatd("p"):.4f} m')
