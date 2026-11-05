"""Training orchestration: model state, optimization, validation early stopping, and result storage.

ChoiceDiffusion handles mathematics, Evaluator handles metrics, and CheckpointStore handles weights.
Preserve initialization, AdamW parameter order, clipping, EMA updates, and RNG call order.
"""
from copy import deepcopy
from itertools import count
import json
from pathlib import Path
import time

import numpy as np
import torch

from checkpoint import CheckpointStore, make_denoiser
from data import loader, load_split, write_json
from diffusion import ChoiceDiffusion
from evaluate import Evaluator
from training_state import TrainingStateStore


def seed_all(seed):
    """Seed NumPy and PyTorch; gate initialization is isolated from the training RNG state."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_epochs(limit):
    """Count epochs from one; limit=0 has no hard limit and stops through validation in fit."""
    if limit < 0:
        raise ValueError("Epoch limits must be nonnegative")
    return count(1) if limit == 0 else range(1, limit + 1)


def append_log(path, record):
    """Append one JSONL record and print it immediately for background-run progress inspection."""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def better(metrics, previous):
    """Accept only strictly improved validation NDCG@10; ties do not reset early stopping."""
    return previous is None or metrics["ndcg@10"] > previous["ndcg@10"]


class EMA:
    """Exponential moving average of the denoiser; the rate gate uses live parameters without EMA."""
    def __init__(self, model, decay):
        """Create a non-trainable evaluation copy and track the number of EMA updates."""
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        """Average parameters after each optimizer step, using the original ramp-up capped by decay."""
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for shadow, current in zip(self.model.parameters(), model.parameters()):
            shadow.lerp_(current, 1 - decay)


class Trainer:
    """Own all state for one independent training run; fit is the training entry point."""
    def __init__(self, output, config, device):
        """Create diffusion, denoiser, EMA, optimizer, and independently shuffled batches in the original order."""
        seed_all(config["seed"])
        self.output, self.config, self.device = Path(output), config, device
        self.settings = config["training"]
        self.process = ChoiceDiffusion.from_directory(output, config, device)
        self.model = make_denoiser(output, config, self.process).to(device)
        self.ema = EMA(self.model, self.settings["ema_decay"])
        policy = self.process.generator.rate_policy
        gate_parameters = list(policy.gate.parameters()) if policy.is_learned else []
        # Keep denoiser parameters before gate parameters for joint clipping and optimization.
        self.parameters = list(self.model.parameters()) + gate_parameters
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.settings["learning_rate"], weight_decay=0.)
        self.training, self.validation = load_split(output, "train"), load_split(output, "val")
        self.batches = loader(self.training, self.settings["batch_size"], shuffle=True, seed=config["seed"])
        self.checkpoints = CheckpointStore(output, config)
        self.evaluator = Evaluator(config, device)

    def train_epoch(self):
        """Train one epoch and return the sample-weighted mean score-entropy loss.

        sequence has shape [B,H] and target has shape [B]; loss samples noise and computes the exact kernel.
        Backpropagate and clip denoiser/gate gradients together, then update only the denoiser's EMA.
        """
        self.model.train()
        total, examples = 0., 0
        for sequence, _, target in self.batches:
            loss = self.process.loss(self.model, sequence.to(self.device), target.to(self.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite score-entropy loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, self.settings["gradient_clip"])
            self.optimizer.step()
            self.ema.update(self.model)
            total += float(loss.detach()) * len(target)
            examples += len(target)
        return total / examples

    def fit(self, resume=False):
        """Train or continue from the last complete epoch, preserving early-stop history.

        patience measures elapsed epochs, not validation calls; validation uses fixed random streams.
        Evaluation weights remain separate from the complete training_state.pt snapshot.
        """
        settings = self.settings
        store = TrainingStateStore(self.output)
        if resume:
            state = store.read(self.config)
            store.restore(self, state)
            store.restore_outputs(self, state)
            best, best_epoch = state["best_validation"], state["best_epoch"]
            best_seconds, elapsed = state["selected_checkpoint_seconds"], state["elapsed_seconds"]
            start_epoch = state["epoch"] + 1
        else:
            if any((self.output / name).exists() for name in
                   ("choice_best.pt", "choice_newst.pt", "choice_log.jsonl", "training_state.pt")):
                raise FileExistsError(f"Training artifacts already exist in {self.output}; use --resume or a new output")
            best, best_epoch, best_seconds, elapsed, start_epoch = None, 0, 0., 0., 1
        started = time.perf_counter()
        epochs = count(start_epoch) if settings["epochs"] == 0 else range(start_epoch, settings["epochs"] + 1)
        for epoch in epochs:
            mean_loss = self.train_epoch()
            metrics, shares = None, None
            if epoch % settings["validation_interval"] == 0 or epoch == settings["epochs"]:
                # Validate and save the same pair: EMA denoiser plus current gate parameters.
                metrics = self.evaluator.evaluate(self.ema.model, self.process, self.validation,
                                                  "val", settings["validation_streams"])
                shares = self.evaluator.rate_statistics(self.process, self.validation)
                if better(metrics, best):
                    best, best_epoch = metrics, epoch
                    best_seconds = elapsed + time.perf_counter() - started
                    self.checkpoints.save("choice_best.pt", self.ema.model, self.process, epoch, metrics, self.ema.updates)
            self.checkpoints.save("choice_newst.pt", self.ema.model, self.process, epoch, metrics, self.ema.updates)
            # Retain the legacy newst filename and overwrite it each epoch instead of accumulating large checkpoints.
            append_log(self.output / "choice_log.jsonl", {"stage": "choice", "epoch": epoch,
                "score_entropy": mean_loss, "validation": metrics, "best_epoch": best_epoch, "rate_shares": shares})
            store.save(self, epoch, best, best_epoch, best_seconds, elapsed + time.perf_counter() - started)
            if metrics is not None and epoch - best_epoch >= settings["patience"]:
                break
        report = {"epochs_completed": epoch, "best_epoch": best_epoch, "best_validation": best,
            "selection_split": "val", "objective": "score_entropy", "epoch_limit": settings["epochs"] or None,
            "stop_reason": "early_stopping" if epoch - best_epoch >= settings["patience"] else "epoch_limit",
            "selected_checkpoint_seconds": best_seconds, "elapsed_seconds": elapsed + time.perf_counter() - started,
            "dynamic_rate_gate": self.process.generator.rate_policy.is_learned, "gate_ema": False,
            "resumed_from_epoch": start_epoch - 1 if resume else None,
            "full_training_state": "training_state.pt", "full_state_interval": 1}
        write_json(self.output / "training.json", report)
        return report
