import numpy as np, json, csv

ARRAYS = 'arrays'
SIM_CSV = '../../../simulation/openLoop.csv'

m = json.load(open(f'{ARRAYS}/manifest.json'))
n, stride, dt = m['n'], m['stride'], m['dt']
N = n // 10

c00 = np.fromfile(f'{ARRAYS}/c_00.bin').reshape(-1, n, n)
e00 = np.fromfile(f'{ARRAYS}/e_00.bin').reshape(-1, n)
print(f'c_00 {c00.shape}   e_00 {e00.shape}   stride {stride}')

# x0 from params.cpp, drone-major: [x, xd, theta, y, yd, phi, z, zd, psi, r]
x0 = np.array([
    -1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  1.0,  0.0,
     1.0,  0.0,  0.0,  2.0, -1.0,  0.0,  3.0,  0.0,  0.5,  1.0,
     0.0,  3.0,  3.0,  4.0, -2.0,  0.0,  0.0,  3.0,  1.0,  1.0,
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    -1.0,  1.0,  1.0, -1.0,  1.0,  1.0,  2.0,  0.0, -0.5,  0.0,
])

reader = list(csv.reader(open(SIM_CSV)))
header, rows = reader[0], reader[1:]
print(f'csv header: {header[:6]} ...  ({len(header)} cols, {len(rows)} rows)')

print(f'\n{"row":>6} {"t":>8} {"err":>12}')
worst = 0.0
for j in range(c00.shape[0]):
    k = j * stride
    if k >= len(rows): break
    x_hat = c00[j] @ x0 + e00[j]
    err = 0.0
    for i in range(N):
        err = max(err, abs(x_hat[i*10+0] - float(rows[k][1+2*i])),
                       abs(x_hat[i*10+3] - float(rows[k][2+2*i])))
    worst = max(worst, err)
    if j < 60:
        print(f'{j:6d} {k*dt:8.2f} {err:12.3e}')

print(f'\nworst overall = {worst:.3e}')
