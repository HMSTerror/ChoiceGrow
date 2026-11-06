"""Precompute neighbors, shared background, and native community/catalog scopes from training data.

structure.pt contains read-only tensors for the mathematical modules; artifacts.json records the protocol.
Validation and test targets are excluded from graph construction, frequency counts, and community discovery.
"""
from pathlib import Path

import numpy as np
import torch

from data import load_split, metadata, prepared_root, write_json
from neighbors import build_neighbors
from scope import community_root_blocks, native_memberships


def build_artifacts(output, config, device):
    """Build the structural artifacts for one experiment; device is retained for API compatibility.

    Neighbor tables have shape [N,K], base has shape [N], and blocks has shape [2,N].
    Build neighbors from training transitions, background from training targets, then native communities.
    The partition seed is independent of the training seed so training runs can share a partition.
    """
    n = metadata(output)["item_count"]
    training = load_split(output, "train")
    destination = config["destination"]
    ids, weights = build_neighbors(training["seq"], training["target"], n,
                destination["neighbor_count"], destination["neighbor_lags"], destination["neighbor_self_prior"])
    counts = torch.from_numpy(np.load(prepared_root(output) / "prepared" / "train_target_counts.npy")).float()
    base = (counts + 1).sqrt()
    # Background counts include training targets only, unlike the history-plus-target long-tail definition.
    settings = config["diffusion"]
    if len(settings["level_rates"]) != 2:
        raise ValueError("The main method uses native-community/catalog two-level rates")
    modules = native_memberships(training, n, settings["partition_seed"], settings["infomap_restarts"])
    blocks = community_root_blocks(modules["memberships"], n)
    # Preserve native Infomap leaf communities without merging them into blocks of size 64 or 1024.
    write_json(Path(output) / "native_memberships.json", modules)
    artifact = {"base": base / base.sum(), "neighbor_ids": ids,
                "neighbor_weights": weights, "blocks": blocks}
    torch.save(artifact, Path(output) / "structure.pt")
    write_json(Path(output) / "artifacts.json", {
        "statistics_split": "train", "destination": "history_neighbors_with_shared_background",
        "neighbor_mix": destination["neighbor_mix"],
        "base_count_definition": "training next-item targets",
        "partition": "native_infomap_leaf_communities_and_catalog", "partition_seed": settings["partition_seed"],
        "partition_method": "directed native Infomap leaf modules and catalog; no packing or splitting",
        "infomap_selection": modules["selected"],
        "infomap_restarts": settings["infomap_restarts"],
        "block_counts": [int(torch.unique(level).numel()) for level in blocks],
        "neighbor_count": ids.shape[1], "neighbor_lags": destination["neighbor_lags"],
        "neighbor_self_prior": destination["neighbor_self_prior"]})
