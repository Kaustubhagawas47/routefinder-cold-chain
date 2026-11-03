"""Fine-tune a pretrained RouteFinder checkpoint on the cold-chain MDVRPLTW
(3 depots, 50 dark stores, 4 compartments, traffic, thermal penalty).

Example (smoke test):
    python run_eal.py --experiment eal-full --max_epochs 1 \
        --limit_train_batches 2 --limit_val_batches 2 --no_wandb
"""

import os
import time

import torch

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from rl4co.utils.callbacks.speed_monitor import SpeedMonitor
from rl4co.utils.trainer import RL4COTrainer

from routefinder.envs.mtdvrp import ColdChainMTVRPGenerator, MTVRPEnv
from routefinder.models.baselines.mvmoe.model import MVMoE
from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingFull
from routefinder.models.finetuning.baselines import adapter_layers, model_from_scratch
from routefinder.models.finetuning.eal import efficient_adapter_layers
from routefinder.models.model import (
    RouteFinderBase,
    RouteFinderMoE,
    RouteFinderSingleVariantSampling,
)
from routefinder.utils import trusted_checkpoint_globals

NUM_DEPOTS = 3
NUM_LOC = 50
DATA_DIR = "./data/"
VAL_FILE = f"mdvrpltw/val/{NUM_LOC}.npz"
TEST_FILE = f"mdvrpltw/test/{NUM_LOC}.npz"
DATALOADER_NAME = f"coldchain{NUM_LOC}"




def load_model(path, model_type):
    if "rf" in model_type:
        cls = RouteFinderMoE if "moe" in model_type else RouteFinderBase
    elif model_type == "mvmoe":
        cls = MVMoE
    elif model_type == "mtpomo":
        cls = RouteFinderSingleVariantSampling
    else:
        raise ValueError(f"Model type not recognized: {model_type}")
    with torch.serialization.safe_globals(trusted_checkpoint_globals(path)):
        return cls.load_from_checkpoint(path, map_location="cpu")


def adapt_model(model, train_type):
    if "eal" in train_type:
        # number of zero-padded feature columns is inferred from old/new layer sizes
        return efficient_adapter_layers(
            model,
            init_embedding_cls=MTVRPInitEmbeddingFull,
            context_embedding_cls=MTVRPContextEmbeddingFull,
            adapter_only="adapter" in train_type,
        )
    if "al" in train_type:
        return adapter_layers(
            model,
            init_embedding_cls=MTVRPInitEmbeddingFull,
            context_embedding_cls=MTVRPContextEmbeddingFull,
            adapter_only="adapter" in train_type,
        )
    if train_type == "scratch":
        return model_from_scratch(
            model,
            init_embedding_cls=MTVRPInitEmbeddingFull,
            context_embedding_cls=MTVRPContextEmbeddingFull,
        )
    raise ValueError(
        f"Training type not recognized: {train_type}. Choose from ['eal', 'al', 'scratch']"
    )


def sync_hparams(model):
    """Lightning recorded the hyperparameters when the *pretrained* model was built, so
    they still hold the single-depot policy and the stock env. Point them at the adapted
    policy and cold-chain env; otherwise a fine-tuned checkpoint rebuilds the old
    architecture on load_from_checkpoint and the weights don't fit."""
    model.hparams["policy"] = model.policy
    model.hparams["env"] = model.env
    model.hparams["metrics"] = {
        **model.hparams.get("metrics", {}),
        "train": model.train_metrics,
        "val": model.val_metrics,
        "test": model.test_metrics,
    }


def build_env():
    generator = ColdChainMTVRPGenerator(num_depot=NUM_DEPOTS, num_loc=NUM_LOC)  # speed: COLDCHAIN_SPEED
    return MTVRPEnv(
        generator,
        check_solution=False,
        select_start_nodes_fn="all",
        data_dir=DATA_DIR,
        val_file=[VAL_FILE],
        test_file=[TEST_FILE],
        val_dataloader_names=[DATALOADER_NAME],
        test_dataloader_names=[DATALOADER_NAME],
    )


def main(args, train_type, run_idx=0):
    # run i of --num_runs uses seed + i (model init for scratch, training data, sampling)
    seed = None
    if args.seed is not None:
        seed = args.seed + run_idx
        seed_everything(seed, workers=True)
    resume_epoch = None
    if args.resume:
        # Already-adapted (fine-tuned) checkpoint: load it as saved, no EAL conversion.
        # Lightning restores optimizer state, epoch and step in trainer.fit(ckpt_path=...)
        with torch.serialization.safe_globals(trusted_checkpoint_globals(args.resume)):
            resume_epoch = torch.load(args.resume, map_location="cpu")["epoch"]
        if args.max_epochs <= resume_epoch + 1:
            raise ValueError(
                f"{args.resume} has finished epoch {resume_epoch} ({resume_epoch + 1} epochs); "
                f"--max_epochs is the new total, so it must be > {resume_epoch + 1}"
            )
        model = load_model(args.resume, args.model_type)
    else:
        model = load_model(args.checkpoint, args.model_type)
        model = adapt_model(model, train_type)

    model.optimizer_kwargs["lr"] = args.lr
    model.env = build_env()
    model.setup()
    model.data_cfg["batch_size"] = args.batch_size
    model.data_cfg["val_batch_size"] = args.val_batch_size
    model.data_cfg["test_batch_size"] = args.val_batch_size
    model.data_cfg["train_data_size"] = args.train_data_size
    # The checkpoint's metric list includes gap_to_bks, but there are no
    # best-known solutions for this problem (it would just log a dummy 100)
    model.val_metrics = [
        "reward", "max_reward", "max_aug_reward", "best_tour_length", "best_thermal",
    ]
    model.test_metrics = list(model.val_metrics)
    # cost terms: reward = -(tour_length + thermal)
    model.train_metrics = list(model.train_metrics) + ["tour_length", "thermal"]
    sync_hparams(model)

    run_dir = os.path.join(
        args.output_dir,
        f"{args.model_type}-{train_type}",
        (f"resume-epoch{resume_epoch}_" if resume_epoch is not None else "")
        + (f"seed{seed}_" if seed is not None else "")
        + time.strftime("%Y-%m-%d_%H-%M-%S"),
    )

    # Always log to CSV (<run_dir>/metrics.csv) so --no_wandb runs keep their curves
    loggers = [CSVLogger(save_dir=run_dir, name="", version="")]
    if not args.no_wandb:
        from lightning.pytorch.loggers import WandbLogger

        loggers.append(
            WandbLogger(
                project="routefinder-eal-coldchain",
                name=f"{args.model_type}-{train_type}",
                reinit=True,
                tags=[args.model_type, train_type, "coldchain"],
            )
        )
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="epoch_{epoch:03d}",
        monitor="val/reward",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        checkpoint_cb,
        RichModelSummary(max_depth=3),
        SpeedMonitor(intra_step_time=True, inter_step_time=True, epoch_time=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = RL4COTrainer(
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_val_batches,
        logger=loggers,
        log_every_n_steps=10,
        callbacks=callbacks,
        default_root_dir=run_dir,
    )

    # Starting point: zero-shot (just after the embedding swap), or the resumed checkpoint
    trainer.validate(model)
    if args.test_zero_shot and not args.resume:
        print("Testing zero-shot generalization (before fine-tuning)")
        trainer.test(model)

    if args.resume:
        print(f"Resuming {args.resume} after epoch {resume_epoch}, training to {args.max_epochs} epochs")
    # Note: when resuming into a new run_dir, "best" (and the final test) is chosen among
    # the resumed epochs only; the starting checkpoint's score is the validate row above
    safe = trusted_checkpoint_globals(args.resume) if args.resume else []
    with torch.serialization.safe_globals(safe):
        trainer.fit(model, ckpt_path=args.resume)

    # Test the best checkpoint (by val/reward), not just the last weights
    best = checkpoint_cb.best_model_path or None
    safe = trusted_checkpoint_globals(best) if best else []
    with torch.serialization.safe_globals(safe):
        trainer.test(model, ckpt_path=best)

    print(f"Finished {train_type} run {run_idx + 1}. Best checkpoint: {best}")
    print(f"Metrics: {os.path.join(run_dir, 'metrics.csv')}")
    if not args.no_wandb:
        import wandb

        wandb.finish()


def _limit(v):
    # Lightning accepts an int (number of batches) or a float (fraction)
    f = float(v)
    return int(f) if f.is_integer() and f > 1 else f


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # mvmoe / mtpomo checkpoints use single-depot MTVRPInitEmbedding (different feature
    # layout), so EAL into the cold-chain Full embeddings is not supported for them
    parser.add_argument(
        "--model_type", type=str, default="rf", choices=["rf", "rf-moe"], help="Model type"
    )
    parser.add_argument("--experiment", type=str, default="eal-full", help="eal-full, al-full, scratch or all")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/50/rf-transformer.ckpt")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="run i of --num_runs uses seed + i; unset = not reproducible")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--train_data_size", type=int, default=10_000)
    parser.add_argument("--limit_train_batches", type=_limit, default=1.0)
    parser.add_argument("--limit_val_batches", type=_limit, default=1.0)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32", help="16-mixed is unreliable on MPS")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--output_dir", type=str, default="logs/eal", help="Checkpoints go to <output_dir>/<model>-<exp>/<time>/"
    )
    parser.add_argument("--test_zero_shot", action="store_true")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="continue training a fine-tuned checkpoint from this run_eal.py (optimizer, epoch and "
        "step restored); --max_epochs is the new total; --experiment only names the run folder",
    )
    args = parser.parse_args()

    available_experiments = ["eal-full", "eal-adapter", "al-full", "al-adapter", "scratch"]
    exps = ["eal-full", "al-full", "scratch"] if args.experiment == "all" else [args.experiment]
    assert all(e in available_experiments for e in exps), (
        f"Invalid experiment: {exps}. Choose from {available_experiments} or 'all'"
    )

    if args.resume:
        assert len(exps) == 1 and args.num_runs == 1, "--resume continues one run"

    for exp in exps:
        for i in range(args.num_runs):
            print(f"Training {exp}, run {i + 1}/{args.num_runs}")
            main(args, exp, i)
