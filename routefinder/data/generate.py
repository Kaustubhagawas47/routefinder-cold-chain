import os

import numpy as np
from scipy.cluster.vq import kmeans2

from rl4co.data.utils import check_extension
from rl4co.utils.pylogger import get_pylogger

from routefinder.envs.mtdvrp.generator import COLDCHAIN_SPEED

log = get_pylogger(__name__)


def get_vehicle_capacity(num_loc):
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap


def generate_mtvrp_data(
    dataset_size,
    num_loc=20,
    min_loc=0.0,
    max_loc=1.0,
    max_time=4.6,
    max_distance_limit=2.8,  # 2sqrt(2) ~= 2.8
    speed=COLDCHAIN_SPEED,
    num_depots=3,  # number of depots, only used for multi-depot problems
    variant="MDVRPLTW",
    vehicle_capacity=645,
    frozen_capacity=75,
    chilled_capacity=120,
    tropical_fruits_capacity=180,
    chilled_vegetables_capacity=270
):
    
    if variant.upper() == "MDVRPLTW":
     num_depots = num_depots
    else:
        raise ValueError(f"Unknown variants: {variant}")
    
    # Generate locations
    dark_stores= np.random.uniform(min_loc, max_loc, (dataset_size, num_loc, 2))
    depots= np.zeros((dataset_size, num_depots, 2))
    for i in range(dataset_size):
        # kmeans2 always returns num_depots centroids (kmeans may drop empty clusters)
        centroids, _ = kmeans2(dark_stores[i], num_depots, minit="++")
        depots[i]=centroids
    
    locs= np.concatenate([depots, dark_stores], axis=1)

    # Generate demands
    vehicle_capacity= vehicle_capacity
    frozen_capacity=frozen_capacity
    chilled_capacity=chilled_capacity
    tropical_fruits_capacity=tropical_fruits_capacity
    chilled_vegetables_capacity=chilled_vegetables_capacity

   #For demand_frozen, mean= 25 kg, sd=2.5
    frozen_raw= np.random.normal(loc=25, scale=2.5, size=(dataset_size,num_loc))
    demand_frozen= np.clip(np.round(frozen_raw),18, 30).astype(int)
    demand_frozen_norm= demand_frozen/frozen_capacity
   
   #For demand_chilled, mean=40, sd=2.5
    chilled_raw= np.random.normal(loc=40, scale=2.5, size=(dataset_size,num_loc))
    demand_chilled= np.clip(np.round(chilled_raw), 30, 50).astype(int) 
    demand_chilled_norm= demand_chilled/chilled_capacity
   
   #For demand_tropical fruits,  mean =60, sd=2.5
    tropical_fruits_raw= np.random.normal(loc=60, scale=2.5, size=(dataset_size,num_loc))
    demand_tropical_fruits= np.clip(np.round(tropical_fruits_raw), 50, 70).astype(int) 
    demand_tropical_fruits_norm= demand_tropical_fruits/tropical_fruits_capacity
   
   #For demand_chilled_vegetables, mean=90, sd= 2.5
    chilled_vegetables_raw= np.random.normal(loc=90, scale=2.5, size=(dataset_size,num_loc))
    demand_chilled_vegetables= np.clip(np.round(chilled_vegetables_raw), 80, 100).astype(int) 
    demand_chilled_vegetables_norm= demand_chilled_vegetables/chilled_vegetables_capacity
   
   # Aggregating the demands into a single tensor
    demand_multi=np.stack([demand_frozen,demand_chilled,demand_tropical_fruits,demand_chilled_vegetables], axis=-1)

    aggregate_demand_kg= demand_multi.sum(axis=-1)
    aggregate_demand_norm= aggregate_demand_kg/vehicle_capacity
    
    # Generate time windows and service time
    time_windows = None
    service_time = None

    # time window for depots
    tw_depots=np.zeros((dataset_size, num_depots, 2))
    tw_depots[:,:,1]= max_time
    # time window for dark stores
    tw_locs= np.zeros((dataset_size, num_loc, 2))
    tw_locs[:,:,1]= 3
    # Concatenating the time windows
    time_windows=np.concatenate([tw_depots, tw_locs], axis=1)
    
    # Calculating service times
    a, b = 0.15, 0.18
    service_time = a + (b - a) * np.random.rand(dataset_size, num_loc)
    # pad service time until the number of depots
    service_time = np.pad(service_time, ((0, 0), (num_depots, 0)))

    # Generate distance limits: dist_lower_bound = 2 * max(depot_to_location_distance),
    # max = min(dist_lower_bound, max_distance_limit). Ensures feasible yet challenging
    # constraints, with each instance having a unique, meaningful limit.
    
    # Calculate the maximum distance from each depot to any dark store; the min over depots
    # (same as MTVRPGenerator.generate_distance_limit) is enough to ensure feasibility
    depot_to_store = np.linalg.norm(
        locs[:, :num_depots, None] - locs[:, None, num_depots:], axis=-1
    )  # [B, num_depots, N]
    max_dist = depot_to_store.max(axis=-1).min(axis=-1)

    # Calculate the minimum distance limit (2 * max_distance)
    distance_lower_bound = 2 * max_dist + 1e-6  # Add epsilon to avoid zero distance

    # Ensure max_distance_limit is not exceeded
    max_distance_limit = np.maximum(max_distance_limit, distance_lower_bound + 1e-6)

    # Generate distance limits between min_distance_limits and max_distance_limit
    distance_limit = np.random.uniform(
            distance_lower_bound,
            np.full_like(distance_lower_bound, max_distance_limit),
            (dataset_size,),
        )[:, None]
    
     # Generate speed
    speed = np.full((dataset_size, 1), speed)

    # Traffic matrix
    total_nodes= num_depots+num_loc

    traffic_matrix= np.ones((dataset_size, total_nodes, total_nodes), dtype=np.float32)
    
    congested_mask= np.random.choice([True, False], size=(dataset_size, total_nodes, total_nodes), p=[0.20,0.80])
    delay_factor= np.random.uniform(1.4,2.5, size=(dataset_size, total_nodes, total_nodes)).astype(np.float32)

    traffic_matrix= np.where(congested_mask, delay_factor, traffic_matrix)

    for i in range(total_nodes):
        traffic_matrix[:, i, i]= 1.0


    data = {
        "locs": locs.astype(np.float32),
        "speed": speed.astype(np.float32),
        "num_depots": np.full((dataset_size, 1), num_depots).astype(np.int32),
        "time_windows": time_windows.astype(np.float32),
        "service_time": service_time.astype(np.float32),
        "distance_limit": distance_limit.astype(np.float32),
        "demand_frozen": demand_frozen_norm.astype(np.float32),
        "demand_chilled": demand_chilled_norm.astype(np.float32),
        "demand_tropical_fruits": demand_tropical_fruits_norm.astype(np.float32),
        "demand_chilled_vegetables": demand_chilled_vegetables_norm.astype(np.float32),
        "demand_linehaul": aggregate_demand_norm.astype(np.float32),
        "traffic_matrix": traffic_matrix.astype(np.float32),
    }

    return data


def generate_dataset(
    filename=None,
    data_dir="data",
    name=None,
    problem="MDVRPLTW",
    dataset_size=1000,
    graph_sizes=[50],
    overwrite=True,
    seed=1234,
    disable_warning=False,
    **kwargs,
):
    """We keep a similar structure as in Kool et al. 2019 but save and load the data as npz
    This is way faster and more memory efficient than pickle and also allows for easy transfer to TensorDict
    """

    fname = filename
    if isinstance(graph_sizes, int):
        graph_sizes = [graph_sizes]
    for graph_size in graph_sizes:
        datadir = os.path.join(data_dir, problem)
        os.makedirs(datadir, exist_ok=True)

        if filename is None:
            fname = os.path.join(
                datadir,
                "{}{}_seed{}.npz".format(
                    graph_size,
                    "_{}".format(name) if name is not None else "",
                    seed,
                ),
            )
        else:
            fname = check_extension(filename, extension=".npz")

        # Generate any needed directories
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if not overwrite and os.path.isfile(check_extension(fname, extension=".npz")):
            if not disable_warning:
                log.info(
                    "File {} already exists! Run with -f option to overwrite. Skipping...".format(
                        fname
                    )
                )
            continue

        # Set seed
        np.random.seed(seed)

        # Automatically generate dataset
        dataset = generate_mtvrp_data(
            dataset_size=dataset_size, num_loc=graph_size, variant=problem, **kwargs
        )

        # A function can return None in case of an error or a skip
        if dataset is not None:
            # Save to disk as dict
            log.info("Saving {} data to {}".format(problem.upper(), fname))
            np.savez(fname, **dataset)
