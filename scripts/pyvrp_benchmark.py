"""Solve the cold-chain test set with PyVRP and score every solution with the true cost
(distance + thermal penalty).

    python scripts/pyvrp_benchmark.py --data data/mdvrpltw/test/50.npz --runtime 2 --workers 7

Saves per-instance results to results/pyvrp/<data name>_<runtime>s.npz and prints a summary.
Compare with a model via scripts/validate_ckpt.py (same scoring code).
"""

import argparse
import os
import sys
import time

from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routefinder.baselines import pyvrp_coldchain as P  # noqa: E402

_INSTS = None


def _init(path):
    global _INSTS
    _INSTS = P.load_instances(path)


def _run(job):
    b, mode, runtime, seed = job
    inst = _INSTS[b]
    t = time.time()
    routes, feasible = P.solve(inst, mode, runtime, seed)
    elapsed = time.time() - t
    dist, thermal, viol = P.evaluate_routes(inst, routes)
    return b, mode, dist, thermal, feasible and not any(viol.values()), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mdvrpltw/test/50.npz")
    ap.add_argument("--runtime", type=float, default=2.0, help="seconds per instance and mode")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--num_instances", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/pyvrp")
    args = ap.parse_args()

    n = len(np.load(args.data)["locs"])
    n = min(n, args.num_instances or n)
    jobs = [(b, m, args.runtime, args.seed) for b in range(n) for m in P.MODES]
    print(f"{n} instances x {len(P.MODES)} modes x {args.runtime}s on {args.workers} workers")

    res = {m: dict(dist=np.zeros(n), thermal=np.zeros(n), ok=np.zeros(n, bool), time=np.zeros(n))
           for m in P.MODES}
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.data,)) as pool:
        for i, (b, m, d, th, ok, el) in enumerate(pool.imap_unordered(_run, jobs, chunksize=4)):
            r = res[m]
            r["dist"][b], r["thermal"][b], r["ok"][b], r["time"][b] = d, th, ok, el
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} solves, {time.time() - t0:.0f}s")

    cost = {m: res[m]["dist"] + res[m]["thermal"] for m in P.MODES}
    best_mode = np.stack([cost[m] for m in P.MODES]).argmin(0)
    pick = lambda k: np.stack([res[m][k] for m in P.MODES])[best_mode, np.arange(n)]  # noqa: E731
    best = dict(dist=pick("dist"), thermal=pick("thermal"), ok=pick("ok"))

    print(f"\nwall time {time.time() - t0:.0f}s")
    for name, r in [*((m, res[m]) for m in P.MODES), ("best of both", best)]:
        print(f"  {name:13s} distance {r['dist'].mean():.4f}  thermal {r['thermal'].mean():.4f}  "
              f"cost {(r['dist'] + r['thermal']).mean():.4f}  valid {r['ok'].sum()}/{n}")

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(args.data.replace("data/", "").replace("/", "_"))[0]
    out = os.path.join(args.out_dir, f"{name}_{args.runtime:g}s.npz")
    np.savez(
        out,
        **{f"{m}_{k}": v for m in P.MODES for k, v in res[m].items()},
        **{f"best_{k}": v for k, v in best.items()},
        runtime=args.runtime, seed=args.seed,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
