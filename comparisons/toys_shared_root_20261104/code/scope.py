"""Native community/catalog nesting and within-block weighted projections P_{u,l}.

Build a directed graph from training interactions and preserve its native Infomap leaf communities.
Add a catalog-wide root level; at runtime, renormalize rho_u within each source item's block.
"""
from collections import Counter

import torch
from torch import nn


def directed_edges(training, item_count):
    """Append targets to valid training histories and count adjacent directed transitions, excluding padding."""
    edges = Counter()
    for history, target in zip(training["seq"].tolist(), training["target"].tolist()):
        chain = [v for v in history if v < item_count] + [target]
        edges.update(zip(chain[:-1], chain[1:]))
    return edges


def native_memberships(training, item_count, seed, restarts):
    """Repeat Infomap and select the partition with the shortest training-graph codelength.

    seed and restarts control candidate partitions; recommendation validation metrics are not used.
    Return each item's complete community path and trial codelengths for reproducibility.
    """
    from infomap import Infomap
    if restarts < 1:
        raise ValueError("At least one Infomap restart is required")
    edges = directed_edges(training, item_count)
    best, trials = None, []
    for restart in range(restarts):
        trial_seed = seed + restart + 1
        model = Infomap(f"--directed --silent --num-trials 1 --seed {trial_seed}")
        for item in range(item_count):
            model.add_node(item)
        for (source, target), weight in sorted(edges.items()):
            model.add_link(source, target, float(weight))
        model.run()
        paths = {int(k): tuple(int(v) for v in path) for k, path in model.get_multilevel_modules().items()}
        trial = {"restart": restart, "seed": trial_seed, "codelength": float(model.codelength)}
        trials.append(trial)
        if best is None or trial["codelength"] < best[0]:
            best = trial["codelength"], paths, trial
    return {"selection": "minimum training graph codelength", "selected": best[2],
            "trials": trials, "memberships": best[1]}


def community_root_blocks(memberships, item_count):
    """Map native leaf-community paths to contiguous IDs and return [2,N] community/catalog labels.

    The first row preserves native communities; the second is all zeros for one shared catalog root.
    """
    paths = {int(item): tuple(path) for item, path in memberships.items()}
    if set(paths) != set(range(item_count)) or any(not path for path in paths.values()):
        raise ValueError("Every catalog item must have a native community path")
    labels = {path: index for index, path in enumerate(sorted(set(paths.values())))}
    community = torch.tensor([labels[paths[item]] for item in range(item_count)], dtype=torch.long)
    return torch.stack((community, torch.zeros_like(community)))


class NestedPartition(nn.Module):
    """Encapsulate fine-to-coarse partition labels, block masses, and weighted projections.

    projection(i,l)[j] = 1{C_l(i)=C_l(j)} * rho_u[j] / rho_u(C_l(i)).
    Compute only requested projection rows, without an N-by-N matrix; the main method uses two levels.
    """

    def __init__(self, blocks):
        """Register [L,N] partition labels as a buffer that follows .to(device)."""
        super().__init__()
        self.register_buffer("blocks", blocks)
        self.level_count = len(blocks)

    def block_sum(self, values, level):
        """Sum [B,N] values by level-specific block and copy each total back to its members."""
        ids = self.blocks[level].expand(len(values), -1)
        totals = torch.zeros_like(values).scatter_add(1, ids, values)
        return totals.gather(1, ids)

    def masses(self, rho):
        """Precompute rho_u block masses for all levels once per diffusion path."""
        return tuple(self.block_sum(rho, level) for level in range(self.level_count))

    def projection(self, source, level, rho, block_mass):
        """Return [B,N] within-block conditional rows P_{u,l}[source,:]."""
        ids = self.blocks[level]
        mask = ids[None] == ids[source, None]
        denominator = block_mass[level].gather(1, source[:, None])
        return rho * mask / denominator

    def bridge(self, level, rho, block_mass, base):
        """Return b P_{u,l}, the bridge term induced by composing shared resets and projections."""
        return rho * self.block_sum(base.expand_as(rho), level) / block_mass[level]
