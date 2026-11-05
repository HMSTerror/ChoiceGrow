"""Complete epoch-boundary snapshots for continuing the same training run."""
# Note: docs/decisions/implemented/feature/2026-11-05-training-resume.md
from copy import deepcopy
from pathlib import Path
import random
import time

import numpy as np
import torch


def validate_resume_config(saved, requested):
    """Allow a new stopping horizon while preserving the numerical experiment."""
    original, current = deepcopy(saved), deepcopy(requested)
    for config in (original, current):
        for name in ("epochs", "patience"):
            config["training"].pop(name, None)
    if original != current:
        raise ValueError("Resume only allows changing training.epochs and training.patience; "
                         "all other settings must match the saved run")


def atomic_save(value, path):
    """Replace the complete file only after torch.save succeeds."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingStateStore:
    """Persist live/EMA models, AdamW, RNGs, shuffling, and selection together.

    Lightweight choice_best/newst files remain compatible evaluation checkpoints.
    This separate snapshot is the authority for resuming at a complete epoch.
    """

    def __init__(self, output):
        self.output = Path(output)
        self.path = self.output / "training_state.pt"

    def save(self, trainer, epoch, best, best_epoch, best_seconds, elapsed_seconds,
             *, fixed_horizon=False):
        """Capture after the epoch log is written without consuming random draws."""
        numpy_state = np.random.get_state()
        log = self.output / "choice_log.jsonl"
        state = {
            "format_version": 2, "epoch_boundary": True, "epoch": epoch,
            "config": deepcopy(trainer.config), "device": str(trainer.device),
            "fixed_horizon": fixed_horizon,
            "live_model": trainer.model.state_dict(),
            "ema_model": trainer.ema.model.state_dict(), "ema_updates": trainer.ema.updates,
            "optimizer": trainer.optimizer.state_dict(), "process": trainer.process.state_dict(),
            "best_validation": best, "best_epoch": best_epoch,
            "selected_checkpoint_seconds": best_seconds, "elapsed_seconds": elapsed_seconds,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            "numpy_rng": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                          "position": numpy_state[2], "has_gauss": numpy_state[3],
                          "cached_gaussian": numpy_state[4]},
            "python_rng": random.getstate(), "shuffle_rng": trainer.batches.generator.get_state(),
            "log_text": log.read_bytes().decode("utf-8") if log.exists() else "",
            "best_checkpoint": torch.load(trainer.checkpoints.best_path, map_location="cpu", weights_only=True)
                               if best is not None else None,
        }
        atomic_save(state, self.path)

    def read(self, config, *, fixed_horizon=False):
        """Read on CPU so RNG tensors stay on CPU even when training uses CUDA."""
        if not self.path.exists():
            raise FileNotFoundError(f"No complete training state: {self.path}. "
                                    "EMA-only checkpoints cannot exactly resume training")
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 2 or not state.get("epoch_boundary"):
            raise ValueError("Unsupported or incomplete training-state snapshot")
        validate_resume_config(state["config"], config)
        if state["fixed_horizon"] != fixed_horizon:
            raise ValueError("The fixed-horizon/early-stopping protocol must match the saved run")
        limit = config["training"]["epochs"]
        if limit and limit <= state["epoch"]:
            raise ValueError(f"Requested epoch limit must exceed saved epoch {state['epoch']}")
        return state

    def restore(self, trainer, state):
        """Restore optimizer and RNGs after constructing the matching model/loader."""
        if str(trainer.device) != state["device"]:
            raise ValueError("Resume on the same device as the saved training run")
        trainer.model.load_state_dict(state["live_model"])
        trainer.ema.model.load_state_dict(state["ema_model"])
        trainer.ema.updates = state["ema_updates"]
        trainer.process.load_state_dict(state["process"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.batches.generator.set_state(state["shuffle_rng"])
        numpy_state = state["numpy_rng"]
        np.random.set_state((numpy_state["name"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                             numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def restore_outputs(self, trainer, state):
        """Rewind an interrupted log and archive stale evaluation/summary files.

        The full snapshot embeds its selected best checkpoint. A newer lightweight
        checkpoint left by an incomplete epoch must not leak into the resumed run.
        """
        log = self.output / "choice_log.jsonl"
        names = ["training.json", "report.json", "recommendations.npz", "long_tail.json",
                 "validation_report.json", "validation_recommendations.npz", "long_tail_val.json"]
        stale = [self.output / name for name in names if (self.output / name).exists()]
        saved_log = state["log_text"].encode("utf-8")
        if log.exists() and log.read_bytes() != saved_log:
            stale.append(log)
        if stale:
            archive = self.output / "resume_history" / str(time.time_ns())
            archive.mkdir(parents=True)
            for path in stale:
                path.replace(archive / path.name)
        log.write_bytes(saved_log)
        if state["best_checkpoint"] is not None:
            atomic_save(state["best_checkpoint"], trainer.checkpoints.best_path)
        else:
            trainer.checkpoints.best_path.unlink(missing_ok=True)
        trainer.checkpoints.save("choice_newst.pt", trainer.ema.model, trainer.process,
                                 state["epoch"], None, trainer.ema.updates)
