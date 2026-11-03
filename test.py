import argparse
import os
import pickle
import time
import warnings

import numpy as np
import torch

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from routefinder.data.utils import get_dataloader
from routefinder.envs import MTVRPEnv
from routefinder.envs.mtdvrp import ColdChainMTVRPGenerator
from routefinder.envs.mtdvrp import MTVRPEnv as ColdChainEnv
from routefinder.models import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE
from routefinder.utils import trusted_checkpoint_globals

# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


def test(
    policy,
    td,
    env,
    num_augment=8,
    augment_fn="dihedral8",  # or symmetric. Default is dihedral8 for reported eval
    num_starts=None,
    device="cuda",
):

    costs_bks = td.get("costs_bks", None)

    with torch.inference_mode():
        with (
            torch.amp.autocast("cuda")
            if "cuda" in str(device)
            else torch.inference_mode()
        ):  # Use mixed precision if supported
            n_start = (
                env.get_num_starts(td, phase="test") if num_starts is None else num_starts
            )

            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)

            # Evaluate policy. The cold-chain env leaves the cost terms (tour length,
            # thermal penalty) of every rollout in env.last_reward_components.
            env.last_reward_components = None
            out = policy(td, env, phase="test", num_starts=n_start, return_actions=True)
            best_components = env.last_reward_components
            env.last_reward_components = None
            if best_components is not None:
                best_components = {
                    k: unbatchify(v, (num_augment, n_start))
                    for k, v in best_components.items()
                }

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            reward = unbatchify(out["reward"], (num_augment, n_start))

            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})
                if best_components is not None:
                    best_components = {
                        k: v.gather(-1, max_idxs[..., None]).squeeze(-1)
                        for k, v in best_components.items()
                    }

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})
                if best_components is not None:
                    best_components = {
                        k: v.gather(1, max_idxs[:, None]).squeeze(1)
                        for k, v in best_components.items()
                    }

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # note: torch.abs is here as a temporary fix, since we forgot to
                    # convert rewards to costs. Does not affect the results.
                    gap_to_bks = (
                        100
                        * (-max_aug_reward - torch.abs(costs_bks))
                        / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if best_components is not None:
                # cost terms of the best solution per instance (behind max_aug_reward)
                out.update({f"best_{k}": v for k, v in best_components.items()})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 69420})  # Dummy value

            return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem name: cvrp, vrptw, etc. or all",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Problem size: 50, 100, for automatic loading",
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda", help="cuda, mps or cpu")
    parser.add_argument(
        "--remove-mixed-backhaul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove mixed backhaul instances. Use --no-remove-mixed-backhaul to keep them.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save results to results/main/{size}/{checkpoint",
    )

    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif opts.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if opts.datasets is not None:
        data_paths = opts.datasets.split(",")
    else:
        # list recursively all npz files in data/
        data_paths = []
        for root, _, files in os.walk("data"):
            for file in files:
                # print(file)
                if "test" not in root:
                    continue
                if file.endswith(".npz"):
                    # mixed-backhaul variant dirs contain "mb" (e.g. data/vrpmbtw/test)
                    variant_dir = os.path.basename(os.path.dirname(root))
                    if opts.remove_mixed_backhaul and "mb" in variant_dir:
                        continue
                    # if name in 50 or 100, append
                    if str(opts.size) in file:
                        if file == "50.npz" or file == "100.npz":
                            data_paths.append(os.path.join(root, file))
        assert len(data_paths) > 0, "No datasets found. Check the data directory."
        data_paths = sorted(sorted(data_paths), key=lambda x: len(x))
        print(f"Found {len(data_paths)} datasets on the following paths: {data_paths}")

    # Load model
    print("Loading checkpoint from ", opts.checkpoint)
    if "mvmoe" in opts.checkpoint:
        BaseLitModule = MVMoE
    elif "mtpomo" in opts.checkpoint:
        BaseLitModule = MTPOMO
    elif "moe" in opts.checkpoint:
        BaseLitModule = RouteFinderMoE
    else:
        BaseLitModule = RouteFinderBase

    with torch.serialization.safe_globals(trusted_checkpoint_globals(opts.checkpoint)):
        model = BaseLitModule.load_from_checkpoint(
            opts.checkpoint, map_location="cpu", strict=False
        )

    stock_env = MTVRPEnv()
    coldchain_env = ColdChainEnv(ColdChainMTVRPGenerator())
    policy = model.policy.to(device).eval()  # Use mixed precision if supported

    results = {}
    for dataset in tqdm(data_paths):

        print(f"Loading {dataset}")
        # Cold-chain data carries a traffic matrix and needs the multi-depot env
        with np.load(dataset) as f:
            env = coldchain_env if "traffic_matrix" in f.files else stock_env
        td_test = env.load_data(dataset)  # this also adds the bks cost if available
        dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        start = time.time()
        res = []
        for batch in dataloader:
            td_test = env.reset(batch).to(device)
            o = test(policy, td_test, env, device=device)
            res.append(o)
        out = {}
        out["max_aug_reward"] = torch.cat([o["max_aug_reward"] for o in res])
        # gap is only meaningful when best-known solutions were loaded
        has_bks = all(torch.is_tensor(o["gap_to_bks"]) for o in res)
        gap = torch.cat([o["gap_to_bks"] for o in res]).mean().item() if has_bks else None

        inference_time = time.time() - start

        dataset_name = dataset.split("/")[-3].split(".")[0].upper()
        gap_str = f"{gap:.3f}%" if gap is not None else "n/a (no BKS)"

        # Cost split (cold-chain only): cost = tour length + thermal penalty
        split = {}
        for k in ("tour_length", "thermal"):
            if all(f"best_{k}" in o for o in res):
                split[k] = torch.cat([o[f"best_{k}"] for o in res]).float().mean().item()
        split_str = (
            f" (Distance: {split['tour_length']:.3f} + Thermal: {split['thermal']:.3f})"
            if len(split) == 2
            else ""
        )
        print(
            f"{dataset_name} | Cost: {-out['max_aug_reward'].mean().item():.3f}{split_str} | Gap: {gap_str} | Inference time: {inference_time:.3f} s"
        )

        if results.get(dataset_name, None) is None:
            results[dataset_name] = {}
        results[dataset_name]["cost"] = -out["max_aug_reward"].mean().item()
        results[dataset_name]["gap"] = gap
        for k, v in split.items():
            results[dataset_name][k] = v
        results[dataset_name]["inference_time"] = inference_time

    if opts.save_results:
        # Save results with checkpoint name under results/main/
        checkpoint_name = opts.checkpoint.split("/")[-1].split(".")[0]
        savedir = f"results/main/{opts.size}/"
        os.makedirs(savedir, exist_ok=True)
        pickle.dump(results, open(savedir + checkpoint_name + ".pkl", "wb"))
