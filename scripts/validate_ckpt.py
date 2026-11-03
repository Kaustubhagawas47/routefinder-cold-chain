"""Independently check a model's best solutions on a cold-chain dataset and score them with
the same code as the PyVRP benchmark (routefinder.baselines.pyvrp_coldchain.evaluate_routes):
every store served once, vehicle + compartment capacity, time windows, distance limit,
return to start depot; distance and thermal recomputed in numpy.

    python scripts/validate_ckpt.py <ckpt> data/mdvrpltw/test/50.npz [--out results/model.npz]

Optionally compares against a PyVRP results file (--pyvrp results/pyvrp/....npz).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routefinder.baselines.pyvrp_coldchain import evaluate_routes, load_instances  # noqa: E402
from routefinder.data.utils import get_dataloader  # noqa: E402
from routefinder.envs.mtdvrp import ColdChainMTVRPGenerator, MTVRPEnv  # noqa: E402
from routefinder.models import RouteFinderBase  # noqa: E402
from routefinder.utils import trusted_checkpoint_globals  # noqa: E402
from test import test  # noqa: E402


def to_routes(actions, num_depots):
    routes, cur = [], None
    for x in actions:
        x = int(x)
        if x < num_depots:
            if cur and cur[1]:
                routes.append(cur)
            cur = (x, [])
        else:
            cur[1].append(x)
    if cur and cur[1]:
        routes.append(cur)
    return routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("data")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=250)
    ap.add_argument("--out", default=None, help="save per-instance results (npz)")
    ap.add_argument("--pyvrp", default=None, help="PyVRP results npz to compare with")
    args = ap.parse_args()

    with torch.serialization.safe_globals(trusted_checkpoint_globals(args.checkpoint)):
        model = RouteFinderBase.load_from_checkpoint(
            args.checkpoint, map_location="cpu", strict=False
        )
    dev = torch.device(args.device)
    policy = model.policy.to(dev).eval()
    env = MTVRPEnv(ColdChainMTVRPGenerator())

    acts, rews, tl, th = [], [], [], []
    t0 = time.time()
    for batch in get_dataloader(env.load_data(args.data), batch_size=args.batch_size):
        o = test(policy, env.reset(batch).to(dev), env, device=dev)
        acts += list(o["best_aug_actions"].cpu())
        rews.append(o["max_aug_reward"].cpu())
        tl.append(o["best_tour_length"].cpu())
        th.append(o["best_thermal"].cpu())
    inf_time = time.time() - t0
    tl, th = torch.cat(tl).numpy(), torch.cat(th).numpy()

    insts = load_instances(args.data)
    n = len(insts)
    D = insts[0]["num_depots"]
    dist, thermal = np.zeros(n), np.zeros(n)
    viol_tot, ok, sizes = {}, np.zeros(n, bool), []
    for b, inst in enumerate(insts):
        routes = to_routes(acts[b].numpy(), D)
        sizes += [len(s) for _, s in routes]
        dist[b], thermal[b], viol = evaluate_routes(inst, routes)
        ok[b] = not any(viol.values())
        for k, v in viol.items():
            viol_tot[k] = viol_tot.get(k, 0) + v

    print(f"instances {n}, inference {inf_time:.1f}s on {args.device}, "
          f"routes/instance {len(sizes) / n:.1f}, stores/route mean {np.mean(sizes):.2f} max {max(sizes)}")
    print(f"valid {ok.sum()}/{n}, violations: {viol_tot}")
    print(f"model-reported  distance {tl.mean():.4f} thermal {th.mean():.4f} cost {(tl + th).mean():.4f}")
    print(f"recomputed      distance {dist.mean():.4f} thermal {thermal.mean():.4f} "
          f"cost {(dist + thermal).mean():.4f}")
    print(f"max |diff| per instance: distance {np.abs(dist - tl).max():.2e} "
          f"thermal {np.abs(thermal - th).max():.2e}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(args.out, dist=dist, thermal=thermal, ok=ok, inference_time=inf_time)
        print(f"saved {args.out}")

    if args.pyvrp:
        r = np.load(args.pyvrp)
        ref = r["best_dist"] + r["best_thermal"]
        m = len(ref)
        cost = (dist + thermal)[:m]
        gap = 100 * (cost - ref) / ref
        print(f"\nvs PyVRP best ({m} instances, {float(r['runtime']):g}s/instance/mode): "
              f"model {cost.mean():.4f} vs PyVRP {ref.mean():.4f}, "
              f"gap mean {gap.mean():.2f}% median {np.median(gap):.2f}%, "
              f"model better on {(gap < -1e-6).sum()} instances")


if __name__ == "__main__":
    main()
