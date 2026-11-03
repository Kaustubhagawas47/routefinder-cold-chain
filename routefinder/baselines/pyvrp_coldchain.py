"""PyVRP reference solver for the cold-chain MDVRPLTW (3 depots, 4 compartments, traffic).

PyVRP models everything except the thermal penalty exactly:
- multiple depots, each vehicle returns to its start depot by the depot time window
- 5-dimensional capacity: total kg + the 4 compartments (frozen, chilled, tropical fruits,
  chilled vegetables), in integer kg
- separate matrices: cost = Euclidean distance, travel time = distance / speed * traffic factor
- store time windows (latest *start* of service, as in the env) and the per-route distance limit

The thermal penalty (env.thermal_penalty_fn) is not expressible in PyVRP's objective, so two
modes are offered and every solution is re-scored with the true cost (distance + thermal):
- "distance":    minimise distance, ignore thermal
- "thermal_cap": every store must finish service by min(T_SAFE) (frozen, 1.5), which makes
                 the thermal penalty zero; stores that cannot make that even with a direct
                 trip get their direct-trip time as the cap so the instance stays feasible
"""

import math

import numpy as np

from pyvrp import Client, Depot, ProblemData, VehicleType
from pyvrp import solve as pyvrp_solve
from pyvrp.stop import MaxRuntime

from routefinder.envs.mtdvrp.env import (
    THERMAL_BETA,
    THERMAL_CAT_RESCALE,
    THERMAL_SCALE,
    THERMAL_T_SAFE,
    THERMAL_WEIGHT,
)

CATEGORIES = ("frozen", "chilled", "tropical_fruits", "chilled_vegetables")
# kg; must match generate.py / ColdChainMTVRPGenerator
VEHICLE_CAPACITY = 645
COMPARTMENT_CAPACITIES = (75, 120, 180, 270)
SCALE = 10_000  # PyVRP needs integers: 1 distance/time unit = SCALE
MODES = ("distance", "thermal_cap")

_TS, _BE, _W, _CR = map(
    np.array, (THERMAL_T_SAFE, THERMAL_BETA, THERMAL_WEIGHT, THERMAL_CAT_RESCALE)
)


def load_instances(path):
    """Load an npz from generate.py into a list of per-instance dicts (numpy, float64)."""
    f = np.load(path)
    B = f["locs"].shape[0]
    D = int(f["num_depots"][0, 0])
    cats = np.stack([f[f"demand_{c}"] for c in CATEGORIES], -1)  # [B, N, 4] normalised
    out = []
    for b in range(B):
        out.append(
            dict(
                num_depots=D,
                locs=f["locs"][b].astype(np.float64),
                speed=float(f["speed"][b, 0]),
                traffic=f["traffic_matrix"][b].astype(np.float64),
                service=f["service_time"][b].astype(np.float64),  # [D+N], 0 at depots
                tw=f["time_windows"][b].astype(np.float64),  # [D+N, 2]
                dist_limit=float(f["distance_limit"][b, 0]),
                cats=cats[b].astype(np.float64),  # [N, 4]
                demand=f["demand_linehaul"][b].astype(np.float64),  # [N]
            )
        )
    return out


def _matrices(inst):
    locs = inst["locs"]
    dist = np.linalg.norm(locs[:, None] - locs[None], axis=-1)
    time = dist / inst["speed"] * inst["traffic"]
    return dist, time


def thermal_per_store(cats, end_time):
    """Thermal penalty of one store (cats [4]) whose service ends at end_time."""
    over = np.clip(end_time - _TS, 0.0, 5.0)
    return float((cats * _CR * _W * (1 - np.exp(-_BE * over))).sum() * THERMAL_SCALE)


def evaluate_routes(inst, routes, tol=1e-5):
    """Float re-check and cost of routes [(start_depot, [store node ids]), ...].
    Mirrors the env: arrival must be before the store's late TW, the vehicle must be back
    at its start depot by the depot's late TW, route length <= distance limit, total and
    per-compartment load <= 1 (normalised)."""
    D = inst["num_depots"]
    N = inst["locs"].shape[0]
    dist, time = _matrices(inst)
    viol = dict(coverage=0, vehicle_cap=0, comp_cap=0, tw=0, depot_tw=0, dist_limit=0)
    served = sorted(j for _, seq in routes for j in seq)
    viol["coverage"] = int(served != list(range(D, N)))
    total_dist = total_thermal = 0.0
    for dep, seq in routes:
        idx = np.array(seq) - D
        viol["vehicle_cap"] += int(inst["demand"][idx].sum() > 1 + tol)
        viol["comp_cap"] += int((inst["cats"][idx].sum(0) > 1 + tol).any())
        t = d = 0.0
        prev = dep
        for j in seq:
            d += dist[prev, j]
            arr = max(t + time[prev, j], inst["tw"][j, 0])
            viol["tw"] += int(arr > inst["tw"][j, 1] + tol)
            t = arr + inst["service"][j]
            total_thermal += thermal_per_store(inst["cats"][j - D], t)
            prev = j
        d += dist[prev, dep]
        viol["depot_tw"] += int(t + time[prev, dep] > inst["tw"][dep, 1] + tol)
        viol["dist_limit"] += int(d > inst["dist_limit"] + tol)
        total_dist += d
    return total_dist, total_thermal, viol


def instance2data(inst, mode="distance"):
    assert mode in MODES, mode
    D = inst["num_depots"]
    N = inst["locs"].shape[0]
    dist, time = _matrices(inst)

    # Conservative integer rounding so that PyVRP-feasible is also float-feasible
    dist_i = np.round(dist * SCALE).astype(np.int64)
    time_i = np.ceil(time * SCALE).astype(np.int64)
    np.fill_diagonal(dist_i, 0)
    np.fill_diagonal(time_i, 0)
    service_i = np.ceil(inst["service"] * SCALE).astype(np.int64)
    late = inst["tw"][:, 1].copy()

    if mode == "thermal_cap":
        # latest start so that service ends by the lowest safe time (frozen)
        cap = min(THERMAL_T_SAFE) - inst["service"]
        # fallback: fastest direct trip from any depot that can still get back in time
        direct = time[:D, :]  # [D, N]
        back_ok = direct + inst["service"][None] + time[:, :D].T <= late[:D, None]
        direct = np.where(back_ok, direct, np.inf).min(0)
        late[D:] = np.minimum(late[D:], np.maximum(cap[D:], direct[D:] + 1e-4))

    late_i = np.floor(late * SCALE).astype(np.int64) - 1  # env uses strict "<"
    max_dist = int(math.floor(inst["dist_limit"] * SCALE)) - 4  # rounding slack (<=4 arcs)

    kg = np.rint(inst["cats"] * np.array(COMPARTMENT_CAPACITIES)).astype(np.int64)  # [N, 4]
    coords = np.rint(inst["locs"] * SCALE).astype(np.int64)

    depots = [Depot(x=int(coords[d, 0]), y=int(coords[d, 1])) for d in range(D)]
    clients = [
        Client(
            x=int(coords[j, 0]),
            y=int(coords[j, 1]),
            delivery=[int(kg[j - D].sum()), *map(int, kg[j - D])],
            service_duration=int(service_i[j]),
            tw_early=int(np.floor(inst["tw"][j, 0] * SCALE)),
            tw_late=int(late_i[j]),
        )
        for j in range(D, N)
    ]
    vehicle_types = [
        VehicleType(
            num_available=N - D,
            capacity=[VEHICLE_CAPACITY, *COMPARTMENT_CAPACITIES],
            start_depot=d,
            end_depot=d,
            tw_early=0,
            tw_late=int(late_i[d]),
            max_distance=max_dist,
        )
        for d in range(D)
    ]
    return ProblemData(clients, depots, vehicle_types, [dist_i], [time_i])


def solve(inst, mode="distance", max_runtime=5.0, seed=0):
    """Returns (routes, feasible). routes: [(start_depot, [store node ids]), ...]"""
    data = instance2data(inst, mode)
    res = pyvrp_solve(data, stop=MaxRuntime(max_runtime), seed=seed, display=False)
    sol = res.best
    routes = [(int(r.start_depot()), [int(v) for v in r.visits()]) for r in sol.routes()]
    return routes, bool(sol.is_feasible())
