"""Result figures for the cold-chain EAL experiment (speed 0.7, test set).

    python scripts/make_figures.py                    # uses the cache if present
    python scripts/make_figures.py --recompute        # re-run models / baselines

Inputs:  fine-tuned checkpoint, pretrained n50 checkpoint, data/mdvrpltw/test/50.npz,
         results/pyvrp/mdvrpltw_test_50_5s.npz (scripts/pyvrp_benchmark.py)
Outputs: results/figures/*.png and *.pdf, results/figures/summary.md (table view)

Figures
  1 cost_split      distance + thermal per method (stacked bars, test-set means)
  2 gap_hist        per-instance gap of the fine-tuned model to PyVRP
  3 route_maps      one representative test instance: before / after fine-tuning / PyVRP
  4 lateness        share of stores served after each compartment's safe time
  5 learning_curves val cost per epoch, EAL vs from scratch (from the runs' metrics.csv)
  6 trip_stops      trips per instance by number of stores (1-stop, 2-stop, ...) and the mean
                    trip length in km, with the unit square rescaled to a --grid_km grid (default 50)

    python scripts/make_figures.py --grid_km 50       # side of the square service area in km
"""

import argparse
import os
import pickle
import sys

from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routefinder.baselines import pyvrp_coldchain as P  # noqa: E402

FINETUNED = "logs/eal/rf-eal-full/2026-11-02_20-22-32/checkpoints/epoch_008.ckpt"
PRETRAINED = "checkpoints/50/rf-transformer.ckpt"
DATA = "data/mdvrpltw/test/50.npz"
PYVRP = "results/pyvrp/mdvrpltw_test_50_5s.npz"
OUT = "results/figures"
LATENESS_SUBSET = 200
CURVES = {  # run_eal.py runs with metrics.csv (CSV logger)
    "EAL (pretrained)": "logs/eal/rf-eal-full/seed1_2026-11-02_23-39-41/metrics.csv",
    "From scratch": "logs/eal/rf-scratch/seed1_2026-11-03_01-11-56/metrics.csv",
}  # PyVRP routes are re-solved for this many instances (figure 4)

# ---- palette (dataviz reference palette, light mode; validated) ----
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3 (all-pairs safe)
CRITICAL = "#d03b3b"  # status: late delivery (always with a label)
CAT_NAMES = ["Frozen", "Chilled", "Tropical fruit", "Chilled veg."]


# ------------------------------------------------------------------ data
def route_times(inst, routes):
    """End-of-service time per store node and the start depot serving it."""
    D = inst["num_depots"]
    dist = np.linalg.norm(inst["locs"][:, None] - inst["locs"][None], axis=-1)
    time = dist / inst["speed"] * inst["traffic"]
    end = np.zeros(inst["locs"].shape[0])
    for dep, seq in routes:
        t, prev = 0.0, dep
        for j in seq:
            t = max(t + time[prev, j], inst["tw"][j, 0]) + inst["service"][j]
            end[j] = t
            prev = j
    return end[D:]


def model_routes(ckpt, insts, pretrained=False):
    import torch

    from routefinder.data.utils import get_dataloader
    from routefinder.envs.mtdvrp import ColdChainMTVRPGenerator, MTVRPEnv
    from scripts.validate_ckpt import to_routes
    from test import test

    if pretrained:
        import run_eal

        model = run_eal.adapt_model(run_eal.load_model(ckpt, "rf"), "eal-full")
    else:
        from routefinder.models import RouteFinderBase
        from routefinder.utils import trusted_checkpoint_globals

        with torch.serialization.safe_globals(trusted_checkpoint_globals(ckpt)):
            model = RouteFinderBase.load_from_checkpoint(ckpt, map_location="cpu", strict=False)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy = model.policy.to(dev).eval()
    env = MTVRPEnv(ColdChainMTVRPGenerator())
    acts = []
    for batch in get_dataloader(env.load_data(DATA), batch_size=250):
        acts += list(test(policy, env.reset(batch).to(dev), env, device=dev)["best_aug_actions"].cpu())
    D = insts[0]["num_depots"]
    return [to_routes(a.numpy(), D) for a in acts]


def nn_routes(insts):
    import torch

    from routefinder.envs.mtdvrp import ColdChainMTVRPGenerator, MTVRPEnv
    from routefinder.utils import greedy_policy
    from scripts.validate_ckpt import to_routes

    env = MTVRPEnv(ColdChainMTVRPGenerator(), check_solution=False)
    td = env.reset(env.load_data(DATA))
    acts = []
    while not td["done"].all():
        td = greedy_policy(td)
        acts.append(td["action"])
        td = env.step(td)["next"]
    A = torch.stack(acts, 1).numpy()
    return [to_routes(a, insts[0]["num_depots"]) for a in A]


def one_store_routes(inst):
    D = inst["num_depots"]
    dist = np.linalg.norm(inst["locs"][:, None] - inst["locs"][None], axis=-1)
    time = dist / inst["speed"] * inst["traffic"]
    routes = []
    for j in range(D, inst["locs"].shape[0]):
        best = None
        for d in range(D):
            t = time[d, j] + inst["service"][j]
            c = 2 * dist[d, j] + P.thermal_per_store(inst["cats"][j - D], t)
            if best is None or c < best[0]:
                best = (c, d)
        routes.append((best[1], [j]))
    return routes


_INSTS = None


def _init():
    global _INSTS
    _INSTS = P.load_instances(DATA)


def _pyvrp_best(args):
    b, runtime = args
    inst = _INSTS[b]
    best = None
    for mode in P.MODES:
        routes, _ = P.solve(inst, mode, runtime)
        d, th, _ = P.evaluate_routes(inst, routes)
        if best is None or d + th < best[0]:
            best = (d + th, routes)
    return b, best[1]


def trip_stats(insts, routes_per_inst, subset=None):
    """One row per trip: (instance, stops, trip length, store-to-store length), in unit-square
    distance. Trip length is the closed tour depot -> stores -> depot; the store-to-store part
    leaves out the two depot legs (0 for a 1-stop trip)."""
    rows = []
    for b in subset if subset is not None else range(len(routes_per_inst)):
        locs = insts[b]["locs"]
        for dep, seq in routes_per_inst[b]:
            path = locs[[dep, *seq, dep]]
            legs = np.linalg.norm(np.diff(path, axis=0), axis=1)
            rows.append((b, len(seq), legs.sum(), legs[1:-1].sum()))
    return np.array(rows)


def compute_trips(insts, methods=None, pv=None):
    """Per-trip stats for the trip_stops figure. Models and nearest neighbour on all test
    instances; PyVRP (routes are not stored by the benchmark) on the first LATENESS_SUBSET."""
    if methods is None:
        print("models (for trip stats) ...")
        methods = {"before": model_routes(PRETRAINED, insts, pretrained=True),
                   "finetuned": model_routes(FINETUNED, insts), "nn": nn_routes(insts)}
    if pv is None:
        print(f"PyVRP routes for {LATENESS_SUBSET} instances ...")
        with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
            pv = dict(pool.map(_pyvrp_best, [(b, 2.0) for b in range(LATENESS_SUBSET)]))
    trips = {k: trip_stats(insts, methods[k]) for k in ("before", "finetuned", "nn")}
    trips["pyvrp"] = trip_stats(insts, pv, subset=range(LATENESS_SUBSET))
    return {"rows": trips, "n_inst": {k: len(insts) for k in ("before", "finetuned", "nn")}
            | {"pyvrp": LATENESS_SUBSET}}


def compute():
    insts = P.load_instances(DATA)
    n = len(insts)
    methods = {}
    print("models ...")
    methods["before"] = model_routes(PRETRAINED, insts, pretrained=True)
    methods["finetuned"] = model_routes(FINETUNED, insts)
    print("heuristics ...")
    methods["nn"] = nn_routes(insts)
    methods["one_store"] = [one_store_routes(inst) for inst in insts]

    data = {"cost": {}, "end_times": {}}
    for name, R in methods.items():
        dist, th, bad = np.zeros(n), np.zeros(n), 0
        for b in range(n):
            dist[b], th[b], v = P.evaluate_routes(insts[b], R[b])
            bad += any(v.values())
        assert bad == 0, f"{name}: {bad} invalid solutions"
        data["cost"][name] = (dist, th)
    r = np.load(PYVRP)
    data["cost"]["pyvrp"] = (r["best_dist"], r["best_thermal"])
    data["cost"]["pyvrp_dist_only"] = (r["distance_dist"], r["distance_thermal"])

    # representative instance for the maps: median gap of the fine-tuned model to PyVRP
    ft = sum(data["cost"]["finetuned"])
    ref = sum(data["cost"]["pyvrp"])
    gap = 100 * (ft - ref) / ref
    b_map = int(np.argsort(gap)[n // 2])
    data["map_instance"] = b_map

    print(f"PyVRP routes for {LATENESS_SUBSET} instances + map instance ...")
    jobs = [(b, 2.0) for b in range(LATENESS_SUBSET)]
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        pv = dict(pool.map(_pyvrp_best, jobs))
        pv_map = dict(pool.map(_pyvrp_best, [(b_map, 5.0)]))[b_map]

    sub = range(LATENESS_SUBSET)
    for name in ("before", "finetuned"):
        data["end_times"][name] = np.stack([route_times(insts[b], methods[name][b]) for b in sub])
    data["end_times"]["pyvrp"] = np.stack([route_times(insts[b], pv[b]) for b in sub])
    data["map"] = dict(
        inst=insts[b_map],
        routes={"before": methods["before"][b_map], "finetuned": methods["finetuned"][b_map],
                "pyvrp": pv_map},
    )
    data["trips"] = compute_trips(insts, methods, pv)
    return data


# ------------------------------------------------------------------ plotting
def style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "legend.frameon": False,
    })


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote", os.path.join(OUT, name + ".png"))


LABELS = {
    "before": "Before fine-tuning",
    "nn": "Nearest neighbour",
    "pyvrp_dist_only": "PyVRP, distance only",
    "one_store": "One store per trip",
    "finetuned": "Fine-tuned (EAL)",
    "pyvrp": "PyVRP (reference)",
}


def fig_cost_split(data):
    import matplotlib.pyplot as plt

    order = ["before", "nn", "one_store", "pyvrp_dist_only", "finetuned", "pyvrp"]
    dist = [data["cost"][k][0].mean() for k in order]
    th = [data["cost"][k][1].mean() for k in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(order))[::-1]
    h = 0.62
    for yi, d, t in zip(y, dist, th):
        ax.barh(yi, d, height=h, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        ax.barh(yi, t, left=d, height=h, color=SERIES[1], edgecolor=SURFACE, linewidth=2)
        # thermal segments too thin to see get their value in the label
        extra = f"  (thermal {t:.2f})" if 0 < t < 0.5 else ""
        ax.text(d + t + 0.5, yi, f"{d + t:.2f}", va="center", ha="left", color=INK, fontsize=9.5)
        if extra:
            ax.text(d + t + 3.4, yi, extra.strip(), va="center", ha="left", color=MUTED, fontsize=8.5)
    ax.set_yticks(y, [LABELS[k] for k in order])
    for lbl, k in zip(ax.get_yticklabels(), order):
        lbl.set_color(INK if k in ("finetuned", "pyvrp") else INK2)
        if k in ("finetuned", "pyvrp"):
            lbl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean cost per instance (distance + thermal penalty)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, max(np.add(dist, th)) * 1.12)
    ax.set_title("Cost split on 1,000 test instances (speed 0.7)", pad=22)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])]
    ax.legend(handles, ["Distance", "Thermal penalty"], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, handlelength=1, handleheight=1)
    save(fig, "cost_split")
    plt.close(fig)


def fig_gap_hist(data):
    import matplotlib.pyplot as plt

    ft = sum(data["cost"]["finetuned"])
    ref = sum(data["cost"]["pyvrp"])
    gap = 100 * (ft - ref) / ref
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bins = np.arange(np.floor(gap.min()), np.ceil(gap.max()) + 1, 1.0)
    ax.hist(gap, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    med, mean = np.median(gap), gap.mean()
    top = ax.get_ylim()[1] * 1.2
    ax.set_ylim(0, top)
    ax.axvline(med, ymax=0.84, color=INK, linewidth=1.2)
    ax.text(med, top * 0.92, f"median {med:.1f}%  (mean {mean:.1f}%)", color=INK, fontsize=9,
            ha="center", va="bottom")
    ax.axvline(0, ymax=0.84, color=AXIS, linewidth=1, linestyle=(0, (3, 2)))
    better = int((gap < 0).sum())
    ax.text(-0.3, top * 0.92, f"model better on {better} instances", color=INK2,
            fontsize=8.5, ha="right", va="bottom")
    ax.set_xlabel("Gap of the fine-tuned model to PyVRP (%)")
    ax.set_ylabel("Instances")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Per-instance gap to PyVRP (1,000 test instances)")
    save(fig, "gap_hist")
    plt.close(fig)


def fig_route_maps(data):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    inst = data["map"]["inst"]
    D = inst["num_depots"]
    locs = inst["locs"]
    panels = [("before", "Before fine-tuning"), ("finetuned", "Fine-tuned (EAL)"), ("pyvrp", "PyVRP")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1))
    for ax, (key, title) in zip(axes, panels):
        routes = data["map"]["routes"][key]
        dist, th, _ = P.evaluate_routes(inst, routes)
        end = route_times(inst, routes)
        late = end > P._TS[0]  # past the frozen safe time: thermal > 0 for this store
        for dep, seq in routes:
            path = locs[[dep, *seq, dep]]
            ax.plot(path[:, 0], path[:, 1], color=SERIES[dep], linewidth=1.4, alpha=0.9,
                    solid_capstyle="round", zorder=2)
        stores = locs[D:]
        ax.scatter(stores[:, 0], stores[:, 1], s=22, color=SURFACE, edgecolor=INK2,
                   linewidth=0.9, zorder=3)
        if late.any():
            ax.scatter(stores[late, 0], stores[late, 1], s=70, facecolor="none",
                       edgecolor=CRITICAL, linewidth=1.6, zorder=4)
        for d in range(D):
            ax.scatter(*locs[d], marker="s", s=90, color=SERIES[d], edgecolor=SURFACE,
                       linewidth=2, zorder=5)
        ax.set_title(title, loc="left")
        ax.text(0, -0.08, f"cost {dist + th:.2f}  =  distance {dist:.2f} + thermal {th:.2f}\n"
                f"{len(routes)} trips, {int(late.sum())} stores late",
                transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
    handles = [Line2D([], [], color=SERIES[d], marker="s", markersize=7, linewidth=1.4,
                      markeredgecolor=SURFACE) for d in range(D)]
    handles += [Line2D([], [], linestyle="none", marker="o", markersize=8, markerfacecolor="none",
                       markeredgecolor=CRITICAL, markeredgewidth=1.6)]
    fig.legend(handles, [f"Depot {chr(65 + d)} and its trips" for d in range(D)]
               + ["Late: frozen goods past safe time"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle(f"Routes on a representative test instance (#{data['map_instance']}, "
                 "median gap to PyVRP)", x=0.125, ha="left", fontsize=11, fontweight="bold")
    save(fig, "route_maps")
    plt.close(fig)


def fig_lateness(data):
    import matplotlib.pyplot as plt

    keys = [("before", "Before fine-tuning"), ("finetuned", "Fine-tuned (EAL)"), ("pyvrp", "PyVRP")]
    rates = {k: 100 * (data["end_times"][k][..., None] > P._TS).mean((0, 1)) for k, _ in keys}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(4)
    w = 0.26
    for i, (k, label) in enumerate(keys):
        xs = x + (i - 1) * w
        ax.bar(xs, rates[k], width=w, color=SERIES[i], edgecolor=SURFACE, linewidth=2, label=label)
        for xi, v in zip(xs, rates[k]):
            ax.text(xi, v + 0.6, f"{v:.1f}" if v >= 0.05 else "0", ha="center", va="bottom",
                    fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{n}\n(safe {t:g})" for n, t in zip(CAT_NAMES, P._TS)])
    ax.tick_params(axis="x", length=0, colors=INK2)
    ax.set_ylabel("Stores served after safe time (%)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Late deliveries by compartment (first {LATENESS_SUBSET} test instances)", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9,
              handlelength=1, handleheight=1)
    save(fig, "lateness")
    plt.close(fig)


def read_curve(path):
    import csv

    pts = []
    for r in csv.DictReader(open(path)):
        if r.get("val/best_tour_length"):
            pts.append((int(r["step"]), float(r["val/best_tour_length"]) + float(r["val/best_thermal"])))
    # first row is the pre-training validate (step 0); the rest are end-of-epoch
    return pts[0][1], np.array([c for _, c in pts[1:]])


def fig_learning_curves():
    import matplotlib.pyplot as plt

    curves = {k: read_curve(v) for k, v in CURVES.items() if os.path.exists(v)}
    if len(curves) < 2:
        print("skip learning_curves (metrics.csv missing)")
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (name, (start, c)) in enumerate(curves.items()):
        ep = np.arange(1, len(c) + 1)
        ax.plot(ep, c, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        best = c.min()
        # higher curve labelled above its end point, lower curve below, so they never collide
        upper = c[-1] >= max(cc[-1] for _, cc in curves.values())
        ax.annotate(f"{name}: best {best:.2f} (untrained {start:.1f})",
                    xy=(len(c), c[-1]), xytext=(len(c), c[-1] + (0.35 if upper else -0.35)),
                    color=INK, fontsize=8.5, ha="right", va="bottom" if upper else "top")
    ax.set_xlim(0.6, len(c) + 0.4)
    lo = min(cc.min() for _, cc in curves.values())
    ax.set_ylim(lo - 0.9, None)  # room for the lower curve's label above the axis
    ax.set_xticks(np.arange(1, len(c) + 1))
    ax.set_xlabel("Epoch (157 training steps each)")
    ax.set_ylabel("Validation cost (128 instances)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Learning curves: pretrained EAL vs training from scratch", pad=10)
    save(fig, "learning_curves")
    plt.close(fig)


def trip_table(data, grid_km):
    """Per method and stop count: trips per instance, mean trip km, mean store-to-store km."""
    out = {}
    for k, rows in data["trips"]["rows"].items():
        n = data["trips"]["n_inst"][k]
        stops = rows[:, 1].astype(int)
        per = {}
        for s in range(1, stops.max() + 1):
            m = stops == s
            per[s] = (m.sum() / n, rows[m, 2].mean() * grid_km if m.any() else np.nan,
                      rows[m, 3].mean() * grid_km if m.any() else np.nan)
        out[k] = dict(per=per, trips=len(rows) / n, km=rows[:, 2].mean() * grid_km,
                      inter_km=rows[:, 3].mean() * grid_km, stops=stops.mean())
    return out


def fig_trip_stops(data, grid_km):
    import matplotlib.pyplot as plt

    if "trips" not in data:
        print("skip trip_stops (run with --recompute or --trips)")
        return
    tab = trip_table(data, grid_km)
    keys = [("before", "Before fine-tuning"), ("finetuned", "Fine-tuned (EAL)"), ("pyvrp", "PyVRP")]
    S = max(max(tab[k]["per"]) for k, _ in keys)
    x = np.arange(1, S + 1)
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    panels = [(0, "Trips per instance", "{:.1f}", "Trips per instance by number of stores"),
              (1, f"Mean trip length (km, {grid_km:g} km grid)", "{:.0f}",
               "Mean trip length by number of stores")]
    for ax, (j, ylabel, fmt, title) in zip(axes, panels):
        for i, (k, label) in enumerate(keys):
            vals = np.array([tab[k]["per"].get(s, (0, np.nan, np.nan))[j] for s in x])
            xs = x + (i - 1) * w
            ax.bar(xs, np.nan_to_num(vals), width=w, color=SERIES[i], edgecolor=SURFACE,
                   linewidth=2, label=label)
            for xi, v in zip(xs, vals):
                if np.isfinite(v) and v > 0:
                    ax.text(xi, v, " " + fmt.format(v), ha="center", va="bottom", fontsize=8,
                            color=INK2)
        ax.set_xticks(x, [f"{s}-stop" for s in x])
        ax.tick_params(axis="x", length=0, colors=INK2)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.set_title(title, pad=24)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9,
                   handlelength=1, handleheight=1)
    notes = "   ".join(f"{lbl}: {tab[k]['trips']:.1f} trips, {tab[k]['stops']:.2f} stores/trip, "
                       f"{tab[k]['km']:.1f} km/trip" for k, lbl in keys)
    fig.text(0.125, -0.04, notes + f"\nModels: {data['trips']['n_inst']['finetuned']:,} test "
             f"instances; PyVRP: first {data['trips']['n_inst']['pyvrp']} (re-solved at 2 s). "
             "Trip length = depot -> stores -> depot.", fontsize=8.5, color=INK2, va="top")
    save(fig, "trip_stops")
    plt.close(fig)


def summary_table(data, grid_km=50.0):
    ref = sum(data["cost"]["pyvrp"])
    lines = ["| Method | Distance | Thermal | Cost | Gap to PyVRP |", "|---|---|---|---|---|"]
    for k in ["pyvrp", "finetuned", "pyvrp_dist_only", "one_store", "nn", "before"]:
        d, t = data["cost"][k]
        gap = 100 * ((d + t) - ref) / ref
        lines.append(f"| {LABELS[k]} | {d.mean():.2f} | {t.mean():.3f} | {(d + t).mean():.2f} | "
                     f"{'—' if k == 'pyvrp' else f'{gap.mean():+.1f}%'} |")
    lines += ["", f"Late deliveries (% of stores past safe time, first {LATENESS_SUBSET} instances)", "",
              "| Method | " + " | ".join(CAT_NAMES) + " |", "|---" * 5 + "|"]
    for k in ("before", "finetuned", "pyvrp"):
        r = 100 * (data["end_times"][k][..., None] > P._TS).mean((0, 1))
        lines.append(f"| {LABELS[k]} | " + " | ".join(f"{v:.1f}" for v in r) + " |")
    if "trips" in data:
        tab = trip_table(data, grid_km)
        S = max(max(t["per"]) for t in tab.values())
        lines += ["", f"Trips by number of stores ({grid_km:g} km x {grid_km:g} km grid; trips per "
                  "instance, mean trip km in brackets; trip = depot -> stores -> depot)", "",
                  "| Method | " + " | ".join(f"{s}-stop" for s in range(1, S + 1))
                  + " | All trips | Stores/trip | Mean trip km | Mean store-to-store km |",
                  "|---" * (S + 5) + "|"]
        for k in ("before", "nn", "finetuned", "pyvrp"):
            t = tab[k]
            cells = [f"{t['per'][s][0]:.2f} ({t['per'][s][1]:.1f})" if s in t["per"]
                     and t["per"][s][0] > 0 else "0" for s in range(1, S + 1)]
            lines.append(f"| {LABELS[k]} | " + " | ".join(cells) + f" | {t['trips']:.2f} | "
                         f"{t['stops']:.2f} | {t['km']:.1f} | {t['inter_km']:.1f} |")
        lines.append(f"\nPyVRP row: first {data['trips']['n_inst']['pyvrp']} test instances "
                     "(re-solved at 2 s); other rows: all test instances.")
    path = os.path.join(OUT, "summary.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--trips", action="store_true",
                    help="recompute only the per-trip stats (trip_stops) and add them to the cache")
    ap.add_argument("--grid_km", type=float, default=50.0,
                    help="side of the square service area in km; unit-square distances are "
                         "multiplied by this for the trip-length numbers (default 50)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "figure_data.pkl")
    if args.recompute or not os.path.exists(cache):
        data = compute()
        pickle.dump(data, open(cache, "wb"))
    else:
        data = pickle.load(open(cache, "rb"))
        if args.trips or "trips" not in data:
            data["trips"] = compute_trips(P.load_instances(DATA))
            pickle.dump(data, open(cache, "wb"))
    style()
    fig_cost_split(data)
    fig_gap_hist(data)
    fig_route_maps(data)
    fig_lateness(data)
    fig_learning_curves()
    fig_trip_stops(data, args.grid_km)
    summary_table(data, args.grid_km)


if __name__ == "__main__":
    main()
