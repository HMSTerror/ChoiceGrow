"""Neighbor tables and the user-specific destination distribution rho_u.

Estimate directed history-item-to-target weights from training samples, then combine the resulting
neighbor distributions using the current history. Defaults allocate 2% to neighbors and 98% to background.
Neighbor construction uses neither validation/test data nor a separately trained SASRec model.
"""
from collections import Counter, defaultdict

import torch
from torch import nn


def build_neighbors(sequence, targets, item_count, count=16, lags=3, self_prior=1.0):
    """Build neighbor IDs and normalized weights [N,K], where K=min(count,N).

    Each training sample contributes its target from the last lags valid history positions, weighted by 1/lag.
    Add a self-transition prior and keep the K highest counts, breaking ties by item ID.
    Items without training statistics fall back to themselves rather than an empty distribution.
    """
    rows = defaultdict(Counter)
    for history, target in zip(sequence.tolist(), targets.tolist()):
        valid = [item for item in history if 0 <= item < item_count]
        for lag, source in enumerate(reversed(valid[-lags:]), 1):
            rows[source][target] += 1.0 / lag
    width = min(count, item_count)
    ids = torch.arange(item_count)[:, None].expand(-1, width).clone()
    weights = torch.zeros(item_count, width)
    weights[:, 0] = 1.0
    for source, counts in rows.items():
        counts[source] += self_prior
        selected = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:width]
        ids[source, :len(selected)] = torch.tensor([item for item, _ in selected])
        weights[source].zero_()
        counts = torch.tensor([value for _, value in selected], dtype=torch.float32)
        weights[source, :len(selected)] = counts / counts.sum()
    return ids, weights


class NeighborDestination(nn.Module):
    """rho_u = (1-neighbor_mix)*shared_background + neighbor_mix*history_neighbors.

    Table construction uses the last lags positions of each training sample; application combines all valid
    positions in the current input history with exp(-history_decay*age), giving older items less weight.
    This module contains device-aware buffers and no parameters updated by backpropagation.
    """

    def __init__(self, artifact, settings):
        """Register the precomputed background and neighbor tables without changing dtypes or item IDs."""
        super().__init__()
        self.settings = settings
        for name in ("base", "neighbor_ids", "neighbor_weights"):
            self.register_buffer(name, artifact[name])
        self.item_count = len(self.base)

    def forward(self, sequence):
        """Map left-padded histories [B,H] to strictly positive, normalized destination distributions [B,N]."""
        base = self.base.expand(len(sequence), -1)
        if self.settings["neighbor_mix"] == 0:
            return base
        valid = sequence < self.item_count  # Padding ID N does not contribute to the neighbor mixture.
        safe = sequence.clamp_max(self.item_count - 1)
        age = sequence.shape[1] - 1 - torch.arange(sequence.shape[1], device=sequence.device)
        weights = valid * torch.exp(-self.settings["history_decay"] * age)
        ids = self.neighbor_ids[safe]
        neighbor_weights = self.neighbor_weights[safe] * weights[..., None]
        # Multiple history positions may share neighbors; accumulate their probability mass with scatter_add.
        recent = torch.zeros_like(base).scatter_add(1, ids.flatten(1), neighbor_weights.flatten(1))
        recent = recent / weights.sum(-1, keepdim=True)
        mix = self.settings["neighbor_mix"]
        rho = (1 - mix) * base + mix * recent
        return rho / rho.sum(-1, keepdim=True)
