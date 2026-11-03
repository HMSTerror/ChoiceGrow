"""Build train-only neighbors/native communities, train score entropy, and test best."""
from contextlib import contextmanager
from copy import deepcopy
from itertools import count
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from data import loader, load_split, metadata, prepared_root, write_json
from diffusion import ChoiceProcess, DynamicChoiceProcess
from models import Denoiser
from neighbors import build_neighbors
from scope import community_root_blocks, native_memberships


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_epochs(limit):
    """Zero leaves the horizon open; validation patience ends training."""
    if limit < 0:
        raise ValueError("Epoch limits must be nonnegative")
    return count(1) if limit == 0 else range(1, limit + 1)


def append_log(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def ranking_totals(scores, target):
    ranked = scores.topk(min(10, scores.shape[1]), -1).indices
    matches = ranked == target[:, None]
    rank = matches.to(torch.int64).argmax(-1) + 1
    found = matches.any(-1)
    result = {}
    for cutoff in (5, 10):
        hit = found & (rank <= cutoff)
        result[f"hr@{cutoff}"] = float(hit.sum())
        result[f"ndcg@{cutoff}"] = float((hit / torch.log2(rank.float() + 1)).sum())
    return result


def better(metrics, previous):
    return previous is None or metrics["ndcg@10"] > previous["ndcg@10"]


def build_artifacts(output, config, device):
    n = metadata(output)["item_count"]
    training = load_split(output, "train")
    destination = config["destination"]
    ids, weights = build_neighbors(training["seq"], training["target"], n,
                destination["neighbor_count"], destination["neighbor_lags"], destination["neighbor_self_prior"])
    counts = torch.from_numpy(np.load(prepared_root(output) / "prepared" / "train_target_counts.npy")).float()
    base = (counts + 1).sqrt()
    settings = config["diffusion"]
    if len(settings["level_rates"]) != 2:
        raise ValueError("The main method uses native-community/catalog two-level rates")
    modules = native_memberships(training, n, settings["partition_seed"], settings["infomap_restarts"])
    blocks = community_root_blocks(modules["memberships"], n)
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


def load_process(output, config, device):
    artifact = torch.load(Path(output) / "structure.pt", map_location="cpu", weights_only=True)
    process_type = DynamicChoiceProcess if config["diffusion"].get("dynamic_rate_gate", False) else ChoiceProcess
    return process_type(artifact, config).to(device)


def checkpoint_state(model, process, config, epoch, metrics, ema_updates):
    state = {"model": model.state_dict(), "epoch": epoch, "validation": metrics,
             "config": config, "objective": "score_entropy", "ema_updates": ema_updates}
    if hasattr(process, "rate_gate"):
        state["rate_gate"] = process.rate_gate.state_dict()
    return state


def restore_process(process, state):
    if hasattr(process, "rate_gate"):
        if "rate_gate" not in state:
            raise ValueError("Learned-rate checkpoint is missing its rate gate")
        process.rate_gate.load_state_dict(state["rate_gate"], strict=True)


@torch.no_grad()
def rate_statistics(process, values, config, device):
    if not hasattr(process, "rate_gate"):
        return None
    shares = []
    for sequence, _, _ in loader(values, config["training"]["eval_batch_size"]):
        rates = process.context(sequence.to(device)).level_rates
        shares.append((rates / rates.sum(-1, keepdim=True)).cpu())
    shares = torch.cat(shares).numpy()
    return {"community": {"mean": float(shares[:, 0].mean()), "std": float(shares[:, 0].std()),
                           "min": float(shares[:, 0].min()), "max": float(shares[:, 0].max())},
            "root_mean": float(shares[:, 1].mean())}


def make_denoiser(output, config, process):
    settings = deepcopy(config)
    settings["catalog_count"] = metadata(output)["item_count"]
    return Denoiser(process.item_count, settings)


@contextmanager
def evaluation_rng(seed, split, stream, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        payload = f"{seed}:{split}:{stream}".encode("utf-8")
        torch.manual_seed(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1))
        yield


@torch.no_grad()
def evaluate_diffusion(model, process, values, config, device, split, streams, predictions_path=None):
    model.eval()
    results, predictions = [], []
    for stream in range(streams):
        totals, substeps, cap_count, batches = {}, 0., 0, []
        with evaluation_rng(config["seed"], split, stream, device):
            for sequence, _, target in loader(values, config["training"]["eval_batch_size"]):
                scores, telemetry = process.sample_scores(model, sequence.to(device),
                                                         config["training"]["guidance_strength"])
                if scores.shape[1] != process.item_count or not torch.isfinite(scores).all():
                    raise FloatingPointError("Expected finite full-catalog ranking scores")
                scores = scores[:, :model.catalog_count]
                if predictions_path is not None:
                    batches.append(scores.topk(min(10, model.catalog_count), -1).indices.cpu().numpy())
                for key, value in ranking_totals(scores, target.to(device)).items():
                    totals[key] = totals.get(key, 0.) + value
                substeps += telemetry["mean_reverse_substeps"] * len(target)
                cap_count += telemetry["exit_cap_activations"]
        if predictions_path is not None:
            predictions.append(np.concatenate(batches))
        results.append({**{key: value / len(values["target"]) for key, value in totals.items()},
                        "mean_reverse_substeps": substeps / len(values["target"]),
                        "exit_cap_activations": cap_count})
    if predictions_path is not None:
        np.savez_compressed(predictions_path, top10=np.stack(predictions), target=values["target"].numpy())
    metrics = {key: sum(row[key] for row in results) / streams for key in results[0]}
    return {**metrics, "examples": len(values["target"]), "streams": streams, "per_stream": results}


class EMA:
    def __init__(self, model, decay):
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for shadow, current in zip(self.model.parameters(), model.parameters()):
            shadow.lerp_(current, 1 - decay)


def train_diffusion(output, config, device):
    seed_all(config["seed"])
    settings = config["training"]
    process = load_process(output, config, device)
    model = make_denoiser(output, config, process).to(device)
    ema = EMA(model, settings["ema_decay"])
    gate_parameters = list(process.rate_gate.parameters()) if hasattr(process, "rate_gate") else []
    parameters = list(model.parameters()) + gate_parameters
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0.)
    training, validation = load_split(output, "train"), load_split(output, "val")
    batches = loader(training, settings["batch_size"], shuffle=True, seed=config["seed"])
    checkpoint = Path(output) / "choice_best.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Diffusion checkpoint already exists: {checkpoint}; choose a new output")
    best, best_epoch, best_seconds, started = None, 0, 0., time.perf_counter()
    objective = "score_entropy"
    for epoch in training_epochs(settings["epochs"]):
        model.train()
        total, examples = 0., 0
        for sequence, _, target in batches:
            loss = process.loss(model, sequence.to(device), target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite score-entropy loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"])
            optimizer.step()
            ema.update(model)
            total += float(loss.detach()) * len(target)
            examples += len(target)
        metrics, shares = None, None
        if epoch % settings["validation_interval"] == 0 or epoch == settings["epochs"]:
            metrics = evaluate_diffusion(ema.model, process, validation, config,
                                         device, "val", settings["validation_streams"])
            shares = rate_statistics(process, validation, config, device)
            if better(metrics, best):
                best, best_epoch = metrics, epoch
                best_seconds = time.perf_counter() - started
                torch.save(checkpoint_state(ema.model, process, config, epoch, metrics, ema.updates), checkpoint)
        torch.save(checkpoint_state(ema.model, process, config, epoch, metrics, ema.updates),
                   Path(output) / "choice_newst.pt")
        append_log(Path(output) / "choice_log.jsonl", {"stage": "choice", "epoch": epoch,
                   "score_entropy": total / examples, "validation": metrics, "best_epoch": best_epoch,
                   "rate_shares": shares})
        if metrics is not None and epoch - best_epoch >= settings["patience"]:
            break
    write_json(Path(output) / "training.json", {"epochs_completed": epoch, "best_epoch": best_epoch,
               "best_validation": best, "selection_split": "val", "objective": objective,
               "epoch_limit": settings["epochs"] or None,
               "stop_reason": "early_stopping" if epoch - best_epoch >= settings["patience"] else "epoch_limit",
               "selected_checkpoint_seconds": best_seconds, "elapsed_seconds": time.perf_counter() - started,
               "dynamic_rate_gate": bool(gate_parameters), "gate_ema": False})


def final_evaluation(output, config, device):
    process = load_process(output, config, device)
    state = torch.load(Path(output) / "choice_best.pt", map_location=device, weights_only=True)
    restore_process(process, state)
    model = make_denoiser(output, config, process).to(device)
    model.load_state_dict(state["model"])
    testing = load_split(output, "test")
    result = evaluate_diffusion(model, process, testing, config,
                                device, "test", config["training"]["test_streams"],
                                predictions_path=Path(output) / "recommendations.npz")
    report = {"dataset": config["dataset"], "seed": config["seed"], "smoke": config["smoke"],
              "method": "choicegrow", "experiment": config.get("experiment", "neighbor_community_root"),
              "partition": config.get("partition", "native_infomap_leaf_communities_and_catalog"),
              "evaluation": "full_catalog", "best_epoch": state["epoch"], "selection_split": "val",
              "objective": state["objective"], "neighbor_mix": config["destination"]["neighbor_mix"], "test": result,
              "dynamic_rate_gate": hasattr(process, "rate_gate"),
              "test_rate_shares": rate_statistics(process, testing, config, device)}
    write_json(Path(output) / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report
