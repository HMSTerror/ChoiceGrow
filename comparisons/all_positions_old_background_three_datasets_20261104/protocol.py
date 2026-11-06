"""Frozen experiment protocol shared by the three-dataset fits and Toys factorial.

Copied from the completed all-position comparison. No candidate selection is included.
Training and inference numerics are unchanged.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CODE = ROOT / "code"
REFERENCE = PROJECT / "runs/raw_seed100_lr0006_patience150_20261103"
sys.path.insert(0, str(CODE))

import numpy as np
import torch
from configuration import load_config
from data import load_split, metadata, prepared_root, write_json
from evaluate import Evaluator
from neighbors import build_neighbors
from train import Trainer

SCORES = ("hr@5", "hr@10", "ndcg@5", "ndcg@10")



def now():
    return datetime.now().astimezone().isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(directory):
    """Derive the best validation result from the entire log, tolerating an unfinished last line."""
    latest, best = 0, None
    path = Path(directory) / "choice_log.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest = max(latest, row.get("epoch", 0))
            value = row.get("validation")
            if value and (best is None or value["ndcg@10"] > best["ndcg@10"]):
                best = {"epoch": row["epoch"], "ndcg@10": value["ndcg@10"]}
    return {"latest_epoch": latest, "best_validation": best}


def prefix_counts(training, n):
    """Count B,C,D,E once for [A,B,C,D]->E; leave model-training examples unchanged."""
    counts = torch.zeros(n, dtype=torch.long)
    for history, target in zip(training["seq"].tolist(), training["target"].tolist()):
        chain = [x for x in history if 0 <= x < n] + [target]
        for item in chain[1:]:
            counts[item] += 1
    return counts


def make_config(dataset, seed, arm, neighbors, background):
    """Keep the current raw-data protocol; only the authorized factors differ."""
    config = read_json(REFERENCE / dataset / "resolved_config.json")
    config["training"].pop("denoiser", None)  # Removed label never affected the model.
    config.update(dataset=dataset, seed=seed, experiment=arm,
                  neighbor_statistics=neighbors, background_statistics=background,
                  partition="native_infomap_leaf_communities_and_catalog" if arm.endswith("community")
                  else "catalog_root_only")
    config["destination"]["neighbor_mix"] = .02 if arm.startswith("neighbor") else 0.
    settings = config["diffusion"]
    if arm.endswith("root"):
        # A one-level root has no learnable share. Preserve the same total Lambda.
        settings["level_rates"] = [settings["total_budget"]]
        settings.pop("initial_shares", None)
        settings["dynamic_rate_gate"] = False
    return config


def signature(config):
    """Compare settings that determine training numerics, ignoring descriptive/path-only labels."""
    training = deepcopy(config["training"])
    training.pop("denoiser", None)
    diffusion = deepcopy(config["diffusion"])
    if diffusion.pop("rate_scale", 1.) != 1.:
        raise ValueError("A non-unit legacy rate_scale is incompatible")
    return {key: deepcopy(config[key]) for key in
            ("dataset", "seed", "history_length", "hidden_size", "smoke", "tf32", "threads")} | {
                "training": training, "destination": config["destination"], "diffusion": diffusion}


def split_hashes(directory):
    data = prepared_root(directory) / "prepared"
    return {split: sha256(data / f"{split}.npz") for split in ("train", "val", "test")}


def prepare_artifact(directory, config):
    """Reuse the exact prepared splits/communities; independently choose b and neighbor counts."""
    directory = Path(directory)
    source = REFERENCE / config["dataset"]
    write_json(directory / "data_source.json", {"path": str(prepared_root(source).resolve())})
    training, n = load_split(directory, "train"), metadata(directory)["item_count"]
    original = torch.load(source / "structure.pt", map_location="cpu", weights_only=True)
    artifact = {key: value.clone() for key, value in original.items()}
    background = config["background_statistics"]
    if background == "prefix":
        counts = prefix_counts(training, n)
        values = (counts.float() + 1).sqrt()
        artifact["base"] = values / values.sum()
        np.save(directory / "prefix_target_counts.npy", counts.numpy())
    else:
        assert background == "target"
        counts = torch.bincount(training["target"], minlength=n)
        expected = (counts.float() + 1).sqrt()
        assert torch.equal(expected / expected.sum(), original["base"])
    destination = config["destination"]
    if config["neighbor_statistics"] == "all_positions":
        artifact["neighbor_ids"], artifact["neighbor_weights"] = build_neighbors(
            training["seq"], training["target"], n, destination["neighbor_count"],
            destination["neighbor_lags"], destination["neighbor_self_prior"], "all_positions")
    if config["experiment"].endswith("root"):
        artifact["blocks"] = torch.zeros((1, n), dtype=torch.long)
    else:
        assert torch.equal(artifact["blocks"], original["blocks"])
        shutil.copy2(source / "native_memberships.json", directory / "native_memberships.json")
    assert artifact["neighbor_ids"].shape == original["neighbor_ids"].shape
    assert torch.isfinite(artifact["neighbor_weights"]).all()
    torch.testing.assert_close(artifact["neighbor_weights"].sum(1), torch.ones(n))
    torch.save(artifact, directory / "structure.pt")
    write_json(directory / "resolved_config.json", config)
    write_json(directory / "config.json", config)
    write_json(directory / "artifacts.json", {
        "statistics_split": "train", "reference": str(source),
        "neighbor_statistics": config["neighbor_statistics"], "background_statistics": background,
        "background_count_total": int(counts.sum()), "training_samples": len(training["target"]),
        "training_sample_expansion": False, "neighbor_lags": destination["neighbor_lags"],
        "neighbor_shape": list(artifact["neighbor_ids"].shape),
        "base_l1_difference": float((artifact["base"] - original["base"]).abs().sum()),
        "block_counts": [int(level.unique().numel()) for level in artifact["blocks"]],
        "partition_seed": config["diffusion"]["partition_seed"],
        "prepared_split_sha256": split_hashes(source)})
    return artifact


def verify_reuse(source, directory, config, artifact):
    """Require matching data, training settings, structural tensors, and complete selected weights."""
    source = Path(source)
    if not all((source / name).exists() for name in
               ("training.json", "choice_best.pt", "resolved_config.json", "structure.pt")):
        raise ValueError("Source training is not complete")
    old = read_json(source / "resolved_config.json")
    if signature(old) != signature(config):
        raise ValueError("Training configuration differs")
    if split_hashes(source) != split_hashes(directory):
        raise ValueError("Prepared data splits differ")
    original = torch.load(source / "structure.pt", map_location="cpu", weights_only=True)
    keys = ("base", "blocks") if config["destination"]["neighbor_mix"] == 0 else tuple(artifact)
    for key in keys:
        if not torch.equal(original[key], artifact[key]):
            raise ValueError(f"Structural tensor differs: {key}")
    state = torch.load(source / "choice_best.pt", map_location="cpu", weights_only=True)
    if signature(state["config"]) != signature(config):
        raise ValueError("Checkpoint configuration differs")
    training = read_json(source / "training.json")
    if state["epoch"] != training["best_epoch"] or state["validation"] != training["best_validation"]:
        raise ValueError("Checkpoint and completed training report disagree")
    if config["diffusion"]["dynamic_rate_gate"] and "rate_gate" not in state:
        raise ValueError("Learned checkpoint lacks its gate")
    logged = progress(source)["best_validation"]
    if logged != {"epoch": training["best_epoch"], "ndcg@10": training["best_validation"]["ndcg@10"]}:
        raise ValueError("Full validation log and training report disagree")


def tail_metrics(directory):
    """Measure full-catalog accuracy and long-tail exposure using one fixed train-only popularity rule."""
    training = load_split(directory, "train")
    n = metadata(directory)["item_count"]
    items = np.concatenate((training["seq"].numpy().ravel(), training["target"].numpy()))
    counts = np.bincount(items[(items >= 0) & (items < n)], minlength=n)
    boundary = int(np.sort(counts)[::-1][math.ceil(.2 * n) - 1])
    tail = counts < boundary
    with np.load(Path(directory) / "recommendations.npz") as predictions:
        ranked, target = predictions["top10"], predictions["target"]
    np.testing.assert_array_equal(target, load_split(directory, "test")["target"].numpy())
    assert ranked.shape == (3, len(target), 10)
    assert ranked.min() >= 0 and ranked.max() < n
    report = read_json(Path(directory) / "report.json")
    result = {"examples": len(target), "streams": 3, "catalog_items": n,
              "tail_catalog_items": int(tail.sum()), "target_groups": {},
              "protocol": {"counts": "training history plus target occurrences",
                           "tail_rule": f"frequency < {boundary}", "boundary_ties": "head",
                           "aggregation": "stream means, then rows; seed means computed separately"}}
    for k in (5, 10):
        top = ranked[:, :, :k]
        hits = top == target[None, :, None]
        found, rank = hits.any(-1), hits.argmax(-1) + 1
        values = {f"hr@{k}": found.mean(0), f"ndcg@{k}": (found / np.log2(rank + 1)).mean(0)}
        for key, value in values.items():
            result[key] = float(value.mean())
            assert abs(result[key] - report["test"][key]) < 1e-7
        result[f"tail_exposure@{k}"] = float(tail[top].mean())
        result[f"tail_list_rate@{k}"] = float(tail[top].any(-1).mean())
        result[f"tail_catalog_coverage@{k}"] = float(np.mean([
            len(np.unique(stream[tail[stream]])) / tail.sum() for stream in top]))
        for label, mask in (("tail", tail[target]), ("head", ~tail[target])):
            result["target_groups"].setdefault(label, {"examples": int(mask.sum())})
            result["target_groups"][label].update({key: float(value[mask].mean()) if mask.any() else None
                                                       for key, value in values.items()})
    return result


def finish(directory, reused=False):
    """Complete inference/analysis without retraining a completed fit."""
    directory = Path(directory)
    config = load_config(directory / "resolved_config.json")
    if not ((directory / "report.json").exists() and (directory / "recommendations.npz").exists()):
        Evaluator(config, torch.device("cuda:0")).test_best(directory)
    report = read_json(directory / "report.json")
    training = read_json(directory / "training.json")
    assert report["best_epoch"] == training["best_epoch"] and report["test"]["streams"] == 3
    write_json(directory / "long_tail.json", tail_metrics(directory))
    write_json(directory / "worker_status.json", {"state": "complete", "updated_at": now(),
               "pid": os.getpid(), "reused_training": reused or (directory / "reuse.json").exists()})


def reuse_training(source, directory, config, artifact):
    verify_reuse(source, directory, config, artifact)
    source, directory = Path(source), Path(directory)
    for name in ("choice_best.pt", "training.json", "choice_log.jsonl"):
        shutil.copy2(source / name, directory / name)
    if (source / "report.json").exists() and (source / "recommendations.npz").exists():
        report = read_json(source / "report.json")
        assert report["best_epoch"] == read_json(source / "training.json")["best_epoch"]
        assert report["test"]["streams"] == 3
        for name in ("report.json", "recommendations.npz"):
            shutil.copy2(source / name, directory / name)
    write_json(directory / "reuse.json", {"source": str(source),
               "checkpoint_sha256": sha256(source / "choice_best.pt"),
               "verified": ["configuration", "checkpoint", "data_splits", "active_structural_tensors", "full_validation_log"],
               "source_configuration": read_json(source / "resolved_config.json")})
    if (directory / "recommendations.npz").exists():
        finish(directory, reused=True)


def flatten(tail):
    result = {key: tail[key] for key in SCORES}
    result.update({"tail_" + key: tail["target_groups"]["tail"][key] for key in SCORES})
    for k in (5, 10):
        for key in ("tail_exposure", "tail_list_rate", "tail_catalog_coverage"):
            result[f"{key}@{k}"] = tail[f"{key}@{k}"]
    return result


def moments(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n": len(values), "values": [float(value) for value in values]}


def worker(spec):
    """Train a fresh job once, or resume evaluation after an already completed fit."""
    job = read_json(spec)
    output = Path(job["output"])
    config = load_config(output / "resolved_config.json")
    torch.set_num_threads(config["threads"])
    if config["tf32"]:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    try:
        if (output / "cancellation.json").exists():
            raise RuntimeError("This job has been cancelled; no training or evaluation was started")
        if not (output / "training.json").exists():
            if any((output / name).exists() for name in ("choice_best.pt", "choice_newst.pt", "choice_log.jsonl")):
                raise RuntimeError("Incomplete training has no optimizer state; refusing to restart or claim a resume")
            write_json(output / "worker_status.json", {"state": "training", "pid": os.getpid(), "updated_at": now()})
            Trainer(output, config, torch.device("cuda:0")).fit()
        write_json(output / "worker_status.json", {"state": "evaluating", "pid": os.getpid(), "updated_at": now()})
        finish(output)
    except BaseException:
        write_json(output / "worker_status.json", {"state": "failed", "pid": os.getpid(), "updated_at": now(),
                   "traceback": traceback.format_exc()})
        raise


def pid_alive(pid):
    """Check liveness without os.kill(pid,0), which is unsafe on Windows."""
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return ctypes.get_last_error() == 5  # Access denied is not evidence of exit.
    try:
        code = wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)
