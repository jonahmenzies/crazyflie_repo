import csv, sys
import matplotlib.pyplot as plt

fname = sys.argv[1] if len(sys.argv) > 1 else "closedLoop.csv"
rows = list(csv.DictReader(open(fname)))
cols = rows[0].keys()

full = "tick_ms" in cols          # logger's _full.csv has measured and predicted


def get(r, key):
    v = r[key]
    return float(v) if v not in ("", None) else None


if full:
    N = sum(1 for k in cols if k.startswith("px") and not k.startswith("pxd"))

    fig, ax = plt.subplots(figsize=(7, 7))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i in range(N):
        c = colours[i % len(colours)]

        xs = [get(r, f"px{i}") for r in rows]
        ys = [get(r, f"py{i}") for r in rows]
        ax.plot(xs, ys, linewidth=1.0, color=c, label=f"drone {i} predicted")

        mx = [(get(r, f"mx{i}"), get(r, f"my{i}")) for r in rows]
        mx = [p for p in mx if p[0] is not None]
        if mx:
            ax.scatter([p[0] for p in mx], [p[1] for p in mx],
                       s=45, color=c, marker="x", zorder=3)

        ax.scatter(xs[0],  ys[0],  s=40, marker="o", color=c)
        ax.scatter(xs[-1], ys[-1], s=50, marker="D", color=c)

    ax.set_title(f"{fname}   (x = measured at pings)")

else:
    N = sum(1 for k in cols if k.startswith("x") and k != "xd")

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(N):
        xs = [float(r[f"x{i}"]) for r in rows]
        ys = [float(r[f"y{i}"]) for r in rows]
        ax.plot(xs, ys, linewidth=1.0, label=f"drone {i}")
        ax.scatter(xs[0],  ys[0],  s=40, marker="o")
        ax.scatter(xs[-1], ys[-1], s=50, marker="D")

    xd = [get(r, "xd") for r in rows]
    yd = [get(r, "yd") for r in rows]
    if any(v is not None for v in xd):
        ax.plot(xd, yd, "k--", linewidth=2, label="desired")

    ax.set_title(fname)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal")
ax.grid(True)
ax.legend(fontsize=8)
plt.show()
