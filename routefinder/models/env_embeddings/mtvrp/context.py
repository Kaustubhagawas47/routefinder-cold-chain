import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = step_context_dim if step_context_dim is not None else embed_dim
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class MTVRPContextEmbedding(EnvContext):
    """Context embedding MTVRP.
    - current time
    - used capacity
    - open route
    - remaining distance (set to default_remain_dist if positive infinity)

    Note that the distance limit (L) and open routes (O) are only embedding during decoding
    in this version
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(MTVRPContextEmbedding, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 10
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        mask = td["used_capacity_backhaul"] == 0
        used_capacity = torch.where(
            mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
        )
        used_capacity_frozen= torch.where(
            mask, td["used_capacity_frozen"], torch.zeros_like(td["used_capacity_frozen"])
        )
        used_capacity_chilled= torch.where(
            mask, td["used_capacity_chilled"], torch.zeros_like(td["used_capacity_chilled"])
        )
        used_capacity_tropical_fruits= torch.where(
            mask, td["used_capacity_tropical_fruits"], torch.zeros_like(td["used_capacity_tropical_fruits"])
        )
        used_capacity_chilled_vegetables= torch.where(
            mask, td["used_capacity_chilled_vegetables"], torch.zeros_like(td["used_capacity_chilled_vegetables"])
        )
        available_load = td["vehicle_capacity"] - used_capacity
        available_frozen= td["frozen_capacity"] - used_capacity_frozen
        available_chilled= td["chilled_capacity"] - used_capacity_chilled
        available_tropical_fruits= td["tropical_fruits_capacity"] - used_capacity_tropical_fruits
        available_chilled_vegetables= td["chilled_vegetables_capacity"] - used_capacity_chilled_vegetables
        remaining_dist = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"],
            posinf=self.default_remain_dist,
        )
        
        #outgoing traffic calculations
        curr_node= td["current_node"].squeeze(-1)
        batch_idx= torch.arange(td.batch_size[0], device=td.device)

        curr_outgoing_traffic= td["traffic_matrix"][batch_idx, curr_node, :]
        
        curr_mean_delay= curr_outgoing_traffic.mean(dim=-1, keepdim=True)
        curr_max_delay= curr_outgoing_traffic.max(dim=-1, keepdim=True).values
        
        context_feats = torch.cat(
            (
                available_load,
                td["current_time"],
                td["open_route"].float(),
                remaining_dist,
                available_frozen,
                available_chilled,
                available_tropical_fruits,
                available_chilled_vegetables,
                curr_mean_delay,
                curr_max_delay,
            ),
            -1,
        )
        return context_feats


class RouteFinderContextEmbedding(EnvContext):
    """Context embedding MTVRP.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)

    We do not need to embed the open route here since it is done encoder-side.
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(RouteFinderContextEmbedding, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 3
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        mask = td["used_capacity_backhaul"] == 0
        used_capacity = torch.where(
            mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
        )
        available_load = td["vehicle_capacity"] - used_capacity
        remaining_dist = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"],
            posinf=self.default_remain_dist,
        )
        context_feats = torch.cat(
            (
                available_load,
                td["current_time"],
                remaining_dist,
            ),
            -1,
        )
        return context_feats


# Simple wrapper for better naming. We can replace with this name after getting
# the final models
class MTVRPContextEmbeddingRouteFinder(RouteFinderContextEmbedding):
    def __init__(self, *args, **kwargs):
        super(MTVRPContextEmbeddingRouteFinder, self).__init__(*args, **kwargs)


class MTVRPContextEmbeddingM(MTVRPContextEmbeddingRouteFinder):
    """Context embedding MTVRP with mixed backhaul.
    This is for the zero-shot or few-short on backhaul_class 2 instances.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 4)
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        context_feats = super(MTVRPContextEmbeddingM, self)._state_embedding(
            embeddings, td
        )
        # this will be 0 and tell the model we are *not* doing VRPMPD if backhaul class is not 2
        available_load_vrpmpd = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        ) * (td["backhaul_class"] == 2)

        # Note: now we need the projection to have embed_dim + 4 features!
        return torch.cat(
            (
                context_feats,
                available_load_vrpmpd,
            ),
            -1,
        )


def _gather_node(x, idx):
    """Select row `idx` along the node axis (dim -2) of x with shape [*batch, N, F],
    idx with shape [*batch]. Returns [*batch, F]."""
    idx = idx.long()[..., None, None].expand(*idx.shape, 1, x.size(-1))
    return x.gather(-2, idx).squeeze(-2)


class MTVRPContextEmbeddingFull(MTVRPContextEmbeddingRouteFinder):
    """Context embedding MTVRP with full features, including the Mixed Backhaul (MB) variants
    and Multi-depot (MD) variants as well.
    In practice, we use the same features as the MTVRPContextEmbeddingRouteFinder, but we add:
    - available load for MB variants (backhaul class 2)
    - locations of the depot where we started the route, since we may need to return there
    - available capacity of the 4 compartments (frozen, chilled, tropical fruits, chilled vegetables)
    - mean and max excess traffic delay (factor - 1) on the arcs leaving the current node
    New features are appended after the pretrained ones so that EAL zero-padding keeps the
    pretrained weights aligned with their features.
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(
            self, embed_dim=embed_dim, step_context_dim=embed_dim + 3 + 1 + 2 + 4 + 2
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        context_feats = super(MTVRPContextEmbeddingFull, self)._state_embedding(
            embeddings, td
        )
        # this will be 0 and tell the model we are *not* doing VRPMPD if backhaul class is not 2
        available_load_vrpmpd = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        ) * (td["backhaul_class"] == 2)

        # td may have batch shape [B] or [B, num_starts] (multistart decoding), so gather
        # explicitly along the node axis instead of relying on gather_by_index's default dim
        start_depot_location = _gather_node(td["locs"], td["current_depot"])

        # Available capacity per compartment (capacities are normalised to 1 by default)
        available_cats = torch.cat(
            (
                td["frozen_capacity"] - td["used_capacity_frozen"],
                td["chilled_capacity"] - td["used_capacity_chilled"],
                td["tropical_fruits_capacity"] - td["used_capacity_tropical_fruits"],
                td["chilled_vegetables_capacity"] - td["used_capacity_chilled_vegetables"],
            ),
            -1,
        )

        # Excess traffic delay (factor - 1) on the arcs leaving the current node
        curr_outgoing_traffic = _gather_node(td["traffic_matrix"], td["current_node"]) - 1.0
        curr_mean_delay = curr_outgoing_traffic.mean(dim=-1, keepdim=True)
        curr_max_delay = curr_outgoing_traffic.max(dim=-1, keepdim=True).values

        return torch.cat(
            (
                context_feats,
                available_load_vrpmpd,
                start_depot_location,
                available_cats,
                curr_mean_delay,
                curr_max_delay,
            ),
            -1,
        )
