from copy import deepcopy

import torch

from routefinder.models.env_embeddings.mtvrp.context import MTVRPContextEmbeddingFull
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingFull

from .utils import freeze_backbone


def _zero_pad_weight(old_weight, new_layer, num_new_feats, name):
    """Keep the old weight columns and append zero columns so it matches `new_layer`."""
    inferred = new_layer.in_features - old_weight.shape[-1]
    if num_new_feats is None:
        num_new_feats = inferred
    assert num_new_feats == inferred >= 0, (
        f"{name}: old layer has {old_weight.shape[-1]} input features and new layer has "
        f"{new_layer.in_features}, so {inferred} new features are needed (got {num_new_feats})"
    )
    return torch.cat(
        [
            old_weight,
            torch.zeros(
                (old_weight.shape[0], num_new_feats),
                dtype=old_weight.dtype,
                device=old_weight.device,
            ),
        ],
        dim=-1,
    )


def efficient_adapter_layers(
    model,
    init_embedding_cls=MTVRPInitEmbeddingFull,
    context_embedding_cls=MTVRPContextEmbeddingFull,
    init_embedding_num_new_feats=None,
    context_embedding_num_new_feats=None,
    adapter_only=True,
):
    """Efficient Active Layers.
    Keep the model the same, replace the embeddings with new zero-padded embeddings for unseen features.

    Args:
        model: the model to be adapted
        init_embedding_cls: the new init embedding class
        context_embedding_cls: the new context embedding class
        init_embedding_num_new_feats: the number of new customer features to be added to the init embedding (initiate with zeros).
            If None, inferred from the old and new layer sizes
        context_embedding_num_new_feats: the number of new features to be added to the context embedding (initiate with zeros).
            If None, inferred from the old and new layer sizes
        adapter_only: if True, only the new embeddings are trained, otherwise the whole model is trained.
    """

    print("Using Efficient Adapter Layers (EAL)")

    policy = model.policy
    embed_dim = policy.decoder.context_embedding.embed_dim

    policy_new = deepcopy(policy)

    init_embedding_new_feat = init_embedding_cls(embed_dim=embed_dim)
    context_embedding_new_feat = context_embedding_cls(embed_dim=embed_dim)

    policy_new.encoder.init_embedding = init_embedding_new_feat
    policy_new.decoder.context_embedding = context_embedding_new_feat

    policy_new = policy_new.to(next(policy.parameters()).device)

    # Init Embedding with EAL (encoder)
    init_embedding_old = deepcopy(policy.encoder.init_embedding)
    # use previous k weights and pad l new number of weights with zeros
    init_embed_new = init_embedding_cls(embed_dim=embed_dim)
    proj_glob_params_new = _zero_pad_weight(
        init_embedding_old.project_global_feats.weight.data,
        init_embed_new.project_global_feats,
        None,  # e.g. RouteFinder (5) -> Full (6): backhaul_class
        "init global feats",
    )
    proj_cust_params_new = _zero_pad_weight(
        init_embedding_old.project_customers_feats.weight.data,
        init_embed_new.project_customers_feats,
        init_embedding_num_new_feats,
        "init customer feats",
    )
    init_embed_new.project_global_feats.weight.data = proj_glob_params_new
    init_embed_new.project_customers_feats.weight.data = proj_cust_params_new

    # Context Embedding with EAL (decoder)
    context_embedding_old = deepcopy(policy.decoder.context_embedding)
    # use previous k weights and pad l new number of weights with zeros
    context_embed_new = context_embedding_cls(embed_dim=embed_dim)
    context_embed_new.project_context.weight.data = _zero_pad_weight(
        context_embedding_old.project_context.weight.data,
        context_embed_new.project_context,
        context_embedding_num_new_feats,
        "context feats",
    )

    # Replace above into the policy
    policy_new.encoder.init_embedding = init_embed_new
    policy_new.decoder.context_embedding = context_embed_new

    # If not full, then we freeze the backbone
    if adapter_only:
        policy_new = freeze_backbone(policy_new)

    model.policy = policy_new
    return model
