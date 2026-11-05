"""Checkpoint I/O independent of the class hierarchy, compatible with existing model/rate_gate keys."""
from copy import deepcopy
from pathlib import Path

import torch

from data import metadata
from models import Denoiser


def make_denoiser(output, config, process):
    """Read the actual catalog size and construct a denoiser without mutating the caller's configuration."""
    settings = deepcopy(config)
    settings["catalog_count"] = metadata(output)["item_count"]
    return Denoiser(process.item_count, settings)


class CheckpointStore:
    """Save the EMA denoiser and current gate from the same epoch.

    Keep the original model/rate_gate keys instead of serializing the combined object's state_dict.
    Module boundaries therefore do not change checkpoint keys. These lightweight
    evaluation weights omit optimizer state; training_state.TrainingStateStore
    owns the separate complete snapshot used to resume training.
    """

    def __init__(self, output, config):
        """Bind the experiment output directory and its complete configuration."""
        self.output, self.config = Path(output), config

    @property
    def best_path(self):
        """Return the existing filename for the validation-selected checkpoint."""
        return self.output / "choice_best.pt"

    def save(self, filename, model, process, epoch, metrics, ema_updates):
        """Save weights and selection metadata; learned-rate runs also save the gate parameters."""
        state = {"model": model.state_dict(), "epoch": epoch, "validation": metrics,
                 "config": self.config, "objective": "score_entropy", "ema_updates": ema_updates}
        policy = process.generator.rate_policy
        if policy.is_learned:
            state["rate_gate"] = policy.gate.state_dict()
        torch.save(state, self.output / filename)

    def load_best(self, process, device):
        """Load tensor weights and strictly restore the gate; return state for restoring the denoiser."""
        state = torch.load(self.best_path, map_location=device, weights_only=True)
        policy = process.generator.rate_policy
        if policy.is_learned:
            if "rate_gate" not in state:
                raise ValueError("Learned-rate checkpoint is missing its rate gate")
            policy.gate.load_state_dict(state["rate_gate"], strict=True)
        return state
