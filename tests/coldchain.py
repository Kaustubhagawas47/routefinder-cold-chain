"""Tests for the cold-chain MDVRPLTW setup (3 depots, compartments, traffic, thermal penalty).

The stock single-depot tests (env.py, policy.py, training.py) still cover routefinder.envs.mtvrp.
"""

import os
import sys

import numpy as np
import pytest
import torch

from rl4co.utils.trainer import RL4COTrainer

from routefinder.data.generate import generate_mtvrp_data
from routefinder.envs.mtdvrp import ColdChainMTVRPGenerator, MTVRPEnv
from routefinder.envs.mtdvrp.selectstartnodes import RandomStartNodes
from routefinder.envs.mtvrp import MTVRPEnv as StockEnv
from routefinder.models import RouteFinderBase, RouteFinderPolicy
from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull
from routefinder.models.env_embeddings.mtvrp.init import (
    MTVRPInitEmbeddingFull,
    MTVRPInitEmbeddingRouteFinder,
)
from routefinder.models.finetuning.eal import efficient_adapter_layers
from routefinder.models.reward_normalization import (
    BaseValues,
    ExponentialMean,
    RewardNormalization,
)
from routefinder.utils import greedy_policy, rollout, trusted_checkpoint_globals

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_eal  # noqa: E402

NUM_DEPOTS, NUM_LOC = 3, 10


@pytest.fixture
def env():
    gen = ColdChainMTVRPGenerator(num_depot=NUM_DEPOTS, num_loc=NUM_LOC)
    return MTVRPEnv(gen, check_solution=True)


@pytest.fixture
def td(env):
    torch.manual_seed(0)
    return env.reset(env.generator(4))


def full_policy():
    return RouteFinderPolicy(
        init_embedding=MTVRPInitEmbeddingFull(), context_embedding=MTVRPContextEmbeddingFull()
    )


def test_env_rollout_valid(env, td):
    actions = rollout(env, td.clone(), greedy_policy)
    reward = env.get_reward(td, actions)  # check_solution=True validates the tours
    assert reward.shape == (4,) and torch.isfinite(reward).all() and (reward < 0).all()


def test_npz_data_loads(env, tmp_path):
    np.random.seed(0)
    data = generate_mtvrp_data(dataset_size=4, num_loc=NUM_LOC, num_depots=NUM_DEPOTS)
    path = tmp_path / "cc.npz"
    np.savez(path, **data)
    td = env.reset(env.load_data(str(path)))
    assert td["traffic_matrix"].shape == (4, NUM_DEPOTS + NUM_LOC, NUM_DEPOTS + NUM_LOC)
    actions = rollout(env, td.clone(), greedy_policy)
    assert torch.isfinite(env.get_reward(td, actions)).all()


def test_num_starts_by_phase(env, td):
    assert env.get_num_starts(td) == NUM_LOC  # training: sample from all starts
    assert env.get_num_starts(td, phase="train") == NUM_LOC
    assert env.get_num_starts(td, phase="val") == NUM_DEPOTS  # greedy: one per depot
    assert env.get_num_starts(td, phase="test") == NUM_DEPOTS


def test_greedy_same_depot_start_is_identical(env, td):
    """Why val/test only use one start per depot: repeats give the same solution."""
    policy = full_policy().eval()
    with torch.no_grad():
        out = policy(
            td.clone(), env, phase="test", decode_type="multistart_greedy",
            num_starts=2 * NUM_DEPOTS, return_actions=True,
        )
    reward = out["reward"].view(2 * NUM_DEPOTS, -1)  # [num_starts, batch]
    assert torch.allclose(reward[:NUM_DEPOTS], reward[NUM_DEPOTS:])


def test_random_start_nodes_are_depots(td):
    starts = RandomStartNodes()(td, num_starts=50, backup_n_starts=50)
    assert starts.shape == (50 * td.shape[0],)
    assert (starts < NUM_DEPOTS).all()


@pytest.mark.parametrize("phase,decode_type", [("train", "multistart_sampling"), ("test", "multistart_greedy")])
def test_full_policy_multistart(env, td, phase, decode_type):
    n = env.get_num_starts(td, phase=phase)
    out = full_policy()(td.clone(), env, phase=phase, decode_type=decode_type, num_starts=n)
    assert out["reward"].shape == (4 * n,) and torch.isfinite(out["reward"]).all()


def test_single_depot_embedding_rejects_multi_depot(td):
    with pytest.raises(ValueError, match="MTVRPInitEmbeddingFull"):
        MTVRPInitEmbeddingRouteFinder()(td)


def test_traffic_features_are_excess_delay(td):
    cust = MTVRPInitEmbeddingFull()._cust_feats(td)
    traffic = cust[..., -3:]  # mean in, max in, mean out
    assert (traffic >= 0).all() and (traffic < 1.5 + 1e-6).all()


def test_reward_normalization_coldchain_bucket(td):
    hits = [v for v in vars(BaseValues()) if RewardNormalization.get_problem_mask(v, td).any()]
    assert hits == ["coldchain"]
    rewards = -torch.rand(4, NUM_LOC) - 1
    norm, _ = ExponentialMean(alpha=0.25)(td, rewards, "div")
    assert torch.allclose(norm.mean(), torch.tensor(-1.0), atol=1e-5)


def test_eal_zero_pads_new_features():
    class Holder:
        policy = RouteFinderPolicy()  # pretrained-style single-depot embeddings

    old = Holder.policy
    old_cust = old.encoder.init_embedding.project_customers_feats.weight.detach().clone()
    old_glob = old.encoder.init_embedding.project_global_feats.weight.detach().clone()
    old_ctx = old.decoder.context_embedding.project_context.weight.detach().clone()

    new = efficient_adapter_layers(Holder(), adapter_only=False).policy
    cust = new.encoder.init_embedding.project_customers_feats.weight
    glob = new.encoder.init_embedding.project_global_feats.weight
    ctx = new.decoder.context_embedding.project_context.weight
    assert torch.equal(cust[:, : old_cust.shape[1]], old_cust) and (cust[:, old_cust.shape[1]:] == 0).all()
    assert torch.equal(glob[:, : old_glob.shape[1]], old_glob) and (glob[:, old_glob.shape[1]:] == 0).all()
    assert torch.equal(ctx[:, : old_ctx.shape[1]], old_ctx) and (ctx[:, old_ctx.shape[1]:] == 0).all()


def test_eal_finetune_checkpoint_reloads(tmp_path):
    """Pretrained (stock env, single-depot policy) -> EAL -> train -> checkpoint -> reload."""
    pretrained = RouteFinderBase(
        StockEnv(generator_params={"num_loc": NUM_LOC, "variant_preset": "all"}),
        RouteFinderPolicy(),
        batch_size=4, train_data_size=8, val_data_size=4, test_data_size=4,
    )
    model = run_eal.adapt_model(pretrained, "eal-full")
    model.env = MTVRPEnv(ColdChainMTVRPGenerator(num_depot=NUM_DEPOTS, num_loc=NUM_LOC))
    model.val_metrics = model.test_metrics = ["reward"]
    run_eal.sync_hparams(model)

    trainer = RL4COTrainer(
        max_epochs=1, accelerator="cpu", devices=1, logger=False,
        default_root_dir=str(tmp_path), enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model)
    ckpt = str(tmp_path / "eal.ckpt")
    trainer.save_checkpoint(ckpt)

    with torch.serialization.safe_globals(trusted_checkpoint_globals(ckpt)):
        reloaded = RouteFinderBase.load_from_checkpoint(ckpt, map_location="cpu")
    assert isinstance(reloaded.policy.encoder.init_embedding, MTVRPInitEmbeddingFull)
    assert isinstance(reloaded.env, MTVRPEnv)
    for k, v in model.policy.state_dict().items():
        assert torch.equal(v.cpu(), reloaded.policy.state_dict()[k])


@pytest.mark.parametrize("phase", ["train", "val"])
def test_cost_components_logged(env, phase):
    """reward = -(tour_length + thermal); best_* follow the solution behind max_aug_reward."""
    model = RouteFinderBase(
        env, full_policy(), batch_size=4, val_data_size=4,
        metrics={
            "train": ["reward", "tour_length", "thermal"],
            "val": ["max_aug_reward", "best_tour_length", "best_thermal"],
        },
    )
    torch.manual_seed(0)
    batch = env.generator(4)
    with torch.no_grad():
        out = model.shared_step(batch, 0, phase=phase)
    if phase == "train":
        total = out["train/tour_length"] + out["train/thermal"]
        assert torch.allclose(total, -out["train/reward"], atol=1e-5)
    else:
        total = out["val/best_tour_length"] + out["val/best_thermal"]
        assert torch.allclose(total, -out["val/max_aug_reward"], atol=1e-5)
    assert env.last_reward_components is None  # consumed, not left stale


def test_pyvrp_coldchain_solutions_are_valid(env, tmp_path):
    """PyVRP reference solver: solutions pass the float re-check, the thermal-cap mode
    removes thermal, and evaluate_routes agrees with the env's reward."""
    pytest.importorskip("pyvrp")
    from routefinder.baselines import pyvrp_coldchain as P

    data = generate_mtvrp_data(dataset_size=2, num_loc=NUM_LOC, num_depots=NUM_DEPOTS)
    path = tmp_path / "cc.npz"
    np.savez(path, **data)
    insts = P.load_instances(str(path))
    td = env.reset(env.load_data(str(path)))
    for b, inst in enumerate(insts):
        costs = {}
        for mode in P.MODES:
            routes, feasible = P.solve(inst, mode, max_runtime=0.3)
            dist, thermal, viol = P.evaluate_routes(inst, routes)
            assert feasible and not any(viol.values()), (mode, viol)
            costs[mode] = (dist, thermal)
            # same solution through the env gives the same cost
            # replay through the env: every action must be allowed by the env's mask
            actions = torch.tensor([x for dep, seq in routes for x in (dep, *seq)])[None]
            state = td[b : b + 1].clone()
            for a in actions[0]:
                assert state["action_mask"][0, a], f"{mode}: env masks action {int(a)}"
                state.set("action", a[None])
                state = env.step(state)["next"]
            reward = env.get_reward(state, actions)
            assert torch.isclose(-reward, torch.tensor(dist + thermal, dtype=torch.float32), atol=1e-3)
        assert costs["thermal_cap"][1] <= costs["distance"][1] + 1e-6
