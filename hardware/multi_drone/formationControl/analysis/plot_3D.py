import csv, sys
import matplotlib.pyplot as plt

fname = sys.argv[1] if len(sys.argv) > 1 else "closedLoop.csv"
rows = list(csv.DictReader(open(fname)))
cols = rows[0].keys()

full = "tick_ms" in cols
if not full:
    sys.exit(f"{fname} has no z columns. Use a _full.csv from the flight logger.")


def get(r, key):
    v = r.get(key)
    return float(v) if v not in ("", None) else None


N = sum(1 for k in cols if k.startswith("px") and not k.startswith("pxd"))
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

for i in range(N):
    c = colours[i % len(colours)]
    xs = [get(r, f"px{i}") for r in rows]
    ys = [get(r, f"py{i}") for r in rows]
    zs = [get(r, f"pz{i}") for r in rows]

    ax.plot(xs, ys, zs, linewidth=1.0, color=c, label=f"drone {i}")
    ax.scatter(xs[0],  ys[0],  zs[0],  s=40, marker="o", color=c)
    ax.scatter(xs[-1], ys[-1], zs[-1], s=50, marker="D", color=c)

    m = [(get(r, f"mx{i}"), get(r, f"my{i}"), get(r, f"mz{i}")) for r in rows]
    m = [p for p in m if p[0] is not None]
    if m:
        ax.scatter([p[0] for p in m], [p[1] for p in m], [p[2] for p in m],
                   s=45, color=c, marker="x")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_title(f"{fname}\n(x = measured at pings)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
