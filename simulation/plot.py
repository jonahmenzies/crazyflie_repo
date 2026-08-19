import csv, sys
import matplotlib.pyplot as plt

fname = sys.argv[1] if len(sys.argv) > 1 else "closedLoop.csv"
rows = list(csv.DictReader(open(fname)))
N = sum(1 for k in rows[0] if k.startswith("x") and k != "xd")

fig, ax = plt.subplots(figsize=(7,7))

for i in range(N):
    xs = [float(r[f"x{i}"]) for r in rows]
    ys = [float(r[f"y{i}"]) for r in rows]
    ax.plot(xs, ys, linewidth=1.0, label=f"drone {i}")
    ax.scatter(xs[0],  ys[0],  s=40, marker="o")
    ax.scatter(xs[-1], ys[-1], s=50, marker="D")

xd = [float(r["xd"]) for r in rows]
yd = [float(r["yd"]) for r in rows]
ax.plot(xd, yd, "k--", linewidth=2, label="desired")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(fname)
ax.set_aspect("equal")
ax.grid(True)
ax.legend()
plt.show()
