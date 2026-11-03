import torch

from rl4co.envs import RL4COEnvBase


def get_select_start_nodes_fn(select_start_nodes, **kwargs):
    if select_start_nodes == "random":
        return RandomStartNodes(**kwargs)
    elif select_start_nodes == "all":  # default from POMO
        return AllSelectStartNodes(**kwargs)
    else:
        raise ValueError(f"Unknown select_start_nodes: {select_start_nodes}")


class SelectStartNodes:
    def __init__(self, num_starts=None):
        self.num_starts = num_starts

    def __call__(self, td, num_starts, backup_n_starts, **kwargs):
        if self.num_starts is not None:
            num_starts = self.num_starts
        if isinstance(
            num_starts, RL4COEnvBase
        ):  # to avoid current RL4CO problematic API, to fix
            num_starts = backup_n_starts
        return self._select(td, num_starts)

    def _select(self, td, num_starts):
        raise NotImplementedError("Implement this method in a subclass")

    def get_num_starts(self, td, phase="train"):
        """Number of multi-starts.

        With several depots every start is a depot, so starts cycle through the depots.
        Greedy decoding (val/test) from the same depot always gives the same solution, so
        outside training we only use one start per depot. Training samples, so all
        num_loc starts are kept (more samples for the POMO shared baseline).
        """
        if self.num_starts is not None:
            return self.num_starts
        num_depots = td["num_depots"].max().item()
        if num_depots > 1 and phase != "train":
            return num_depots
        return td["locs"].shape[-2] - num_depots


class RandomStartNodes(SelectStartNodes):
    def _select(self, td, num_starts):
        # With several depots a route must start at a depot, so only sample depots.
        # Flattened in the same [num_starts * batch] order as AllSelectStartNodes.
        num_depots = td["num_depots"].max().item()
        high = num_depots if num_depots > 1 else td["locs"].size(1)
        return torch.randint(0, high, (num_starts * td.shape[0],), device=td.device)


class AllSelectStartNodes(SelectStartNodes):
    """Note: in multi-depot variants, we should start from all possible depots"""

    def _select(self, td, num_starts):
        num_depots = td["num_depots"].max().item()
        num_loc = td["locs"].shape[-2] - num_depots
        if num_depots > 1:
            selected = (
                torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
                % num_depots
            )
            # NOTE: we take num_starts same as num_loc; however, here
            # we actually start from all depots (multiple times)
        else:
            selected = (
                torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
                % num_loc
                + 1
            )
        return selected
