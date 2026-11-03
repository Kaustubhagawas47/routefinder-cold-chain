import argparse
import logging

from routefinder.data.generate import generate_dataset

logging.basicConfig(level=logging.INFO)

# Cold-chain MDVRPLTW is the only problem generate.py supports
PROBLEM = "MDVRPLTW"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--val_data_size", type=int, default=128
    )  # to make validation faster use 128
    parser.add_argument("--test_data_size", type=int, default=1000)
    parser.add_argument("--num_nodes", type=int, nargs="+", default=[50])
    parser.add_argument("--num_depots", type=int, default=3)
    parser.add_argument("--val_seed", type=int, default=4321)
    parser.add_argument("--test_seed", type=int, default=1234)
    args = parser.parse_args()

    # Print config
    print("Config:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    seeds = {"val": args.val_seed, "test": args.test_seed}

    for phase, seed in seeds.items():
        for size in args.num_nodes:
            generate_dataset(
                problem=PROBLEM,
                data_dir=args.data_dir,
                filename=args.data_dir + f"/{PROBLEM.lower()}/{phase}/{size}.npz",
                dataset_size=(
                    args.val_data_size if phase == "val" else args.test_data_size
                ),
                graph_sizes=size,
                seed=seed,
                num_depots=args.num_depots,
            )
