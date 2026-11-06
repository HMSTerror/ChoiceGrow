"""Screen neighbor/background statistics, select on validation, then run the Toys factorial.

The frozen code/ is independent of the main project and earlier comparisons.
No task retries incomplete training: checkpoints contain no optimizer state.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CODE = ROOT / "code"
OUTPUT = ROOT / "runs"
REFERENCE = PROJECT / "runs/raw_seed100_lr0006_patience150_20261103"
PREFIX = PROJECT / "comparisons/prefix_background_seed100_20261103/runs"
sys.path.insert(0, str(CODE))

import numpy as np
import torch
from configuration import load_config
from data import load_split, metadata, prepared_root, write_json
from evaluate import Evaluator
from neighbors import build_neighbors, neighbor_counts
from train import Trainer

DATASETS = ("Toys", "Beauty")
SEEDS = (100, 101, 102)
ARMS = ("shared_root", "neighbor_root", "shared_community", "neighbor_community")
VARIANTS = (
    ("last_target", "target"), ("last_target", "prefix"),
    ("all_positions", "target"), ("all_positions", "prefix"),
)
SCORES = ("hr@5", "hr@10", "ndcg@5", "ndcg@10")


def now():
    return datetime.now().astimezone().isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant_name(neighbors, background):
    return f"{neighbors}__{background}"


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


def create_job(phase, dataset, seed, arm, neighbors, background, candidates=()):
    label = variant_name(neighbors, background)
    relative = Path("screening") / dataset / label if phase == "screening" else Path("factorial") / arm / f"seed_{seed}"
    directory = OUTPUT / relative
    directory.mkdir(parents=True, exist_ok=False)
    config = make_config(dataset, seed, arm, neighbors, background)
    artifact = prepare_artifact(directory, config)
    reuse, rejected = None, []
    for candidate in candidates:
        try:
            verify_reuse(candidate, directory, config, artifact)
        except (ValueError, FileNotFoundError) as error:
            rejected.append({"source": str(candidate), "reason": str(error)})
            continue
        reuse_training(candidate, directory, config, artifact)
        reuse = str(candidate)
        break
    job = {"phase": phase, "dataset": dataset, "seed": seed, "arm": arm,
           "neighbors": neighbors, "background": background, "variant": label,
           "output": str(directory), "reuse_source": reuse, "rejected_reuse": rejected}
    write_json(directory / "job.json", job)
    return job | {"state": "complete" if (directory / "long_tail.json").exists() else "queued",
                  "spec": str(directory / "job.json"), "pid": None}


def prepare():
    OUTPUT.mkdir(exist_ok=False)
    jobs = []
    # Run the two backgrounds together within each dataset for direct progress comparisons.
    for neighbors, background in VARIANTS:
        for dataset in DATASETS:
            old = REFERENCE / dataset if background == "target" else PREFIX / dataset
            candidates = (old,) if neighbors == "last_target" else ()
            jobs.append(create_job("screening", dataset, 100, "neighbor_community", neighbors, background, candidates))
    # Put the two Toys fits first, followed by the two Beauty fits.
    jobs.sort(key=lambda job: (job["state"] != "complete", job["dataset"] != "Toys", job["background"] != "target"))
    manifest = {"created_at": now(), "phase": "screening", "datasets": list(DATASETS),
                "seed": 100, "workers": 2, "selection_metric": "mean_relative_best_validation_ndcg10",
                "selection_datasets": list(DATASETS), "selection_uses_test_metrics": False,
                "selection_candidates": [variant_name(*v) for v in VARIANTS],
                "factorial_seeds": list(SEEDS), "factorial_arms": list(ARMS),
                "training_sample_expansion": False, "jobs": jobs,
                "snapshot_sha256": {p.name: sha256(p) for p in sorted(CODE.iterdir()) if p.is_file()}}
    write_json(OUTPUT / "manifest.json", manifest)
    write_json(OUTPUT / "status.json", {"state": "prepared", "phase": "screening", "jobs": jobs})
    # Document why the earlier Choice factorial cannot be reused for the current raw-data protocol.
    prior = Path("E:/Choice/runs/toys_native_community_real_20261102")
    if prior.exists():
        old = read_json(prior / "resolved_config.json")
        current = read_json(REFERENCE / "Toys/resolved_config.json")
        write_json(OUTPUT / "reuse_audit.json", {
            "older_choice_source": str(prior), "eligible": False,
            "source_split_hashes": split_hashes(prior), "current_split_hashes": split_hashes(REFERENCE / "Toys"),
            "old_learning_rate": old["training"]["learning_rate"],
            "current_learning_rate": current["training"]["learning_rate"],
            "old_patience": old["training"]["patience"], "current_patience": current["training"]["patience"],
            "reason": "Different data splits, learning rate, patience, and TF32 protocol"})
    summarize(manifest)
    print(json.dumps({"state": "prepared", "screening_jobs": len(jobs),
                      "training_reused": sum(job["reuse_source"] is not None for job in jobs),
                      "new_training": sum(job["reuse_source"] is None for job in jobs)}), flush=True)


def choose_variant(manifest):
    """Rank candidates using validation only; stable ties prefer the earlier declared candidate."""
    table = {}
    for neighbors, background in VARIANTS:
        label = variant_name(neighbors, background)
        table[label] = {}
        for dataset in DATASETS:
            job = next(job for job in manifest["jobs"] if job["phase"] == "screening"
                       and job["dataset"] == dataset and job["variant"] == label)
            training = read_json(Path(job["output"]) / "training.json")
            table[label][dataset] = {"best_epoch": training["best_epoch"],
                                    "ndcg@10": training["best_validation"]["ndcg@10"]}
    baseline = variant_name(*VARIANTS[0])
    scores = {label: sum(values[d]["ndcg@10"] / table[baseline][d]["ndcg@10"] - 1
                        for d in DATASETS) / len(DATASETS) for label, values in table.items()}
    winner = max(scores, key=scores.get)
    return {"selected_at": now(), "selected_variant": winner, "criterion": manifest["selection_metric"],
            "formula": "mean_dataset(best_val_NDCG10(candidate)/best_val_NDCG10(original)-1)",
            "scores": scores, "validation_results": table, "uses_test_metrics": False,
            "tie_break": "declared candidate order: original, background-only, neighbors-only, both",
            "interpretation": "Exploratory selection on seed100; reused seed100 is not independent confirmation"}


def prepare_factorial(manifest):
    selection = choose_variant(manifest)
    write_json(OUTPUT / "selection.json", selection)
    neighbors, background = selection["selected_variant"].split("__")
    candidates = [Path(job["output"]) for job in manifest["jobs"] if job["phase"] == "screening"
                  and job["dataset"] == "Toys"]
    candidates += [REFERENCE / "Toys", PREFIX / "Toys"]
    jobs = []
    for seed in SEEDS:
        for arm in ("neighbor_community", "shared_community", "neighbor_root", "shared_root"):
            jobs.append(create_job("factorial", "Toys", seed, arm, neighbors, background, candidates))
    manifest["jobs"].extend(jobs)
    manifest.update(phase="factorial", selection=selection,
                    factorial_reused_training=sum(job["reuse_source"] is not None for job in jobs),
                    factorial_new_training=sum(job["reuse_source"] is None for job in jobs))
    write_json(OUTPUT / "manifest.json", manifest)
    print(json.dumps({"phase": "factorial", "selected": selection["selected_variant"],
                      "reused_training": manifest["factorial_reused_training"],
                      "new_training": manifest["factorial_new_training"]}), flush=True)


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


def summarize(manifest):
    rows, aggregates = [], {}
    for job in manifest["jobs"]:
        directory = Path(job["output"])
        if not (directory / "long_tail.json").exists():
            continue
        row = {key: job[key] for key in ("phase", "dataset", "seed", "arm", "variant", "reuse_source")}
        row.update(output=str(directory), metrics=flatten(read_json(directory / "long_tail.json")),
                   validation=progress(directory)["best_validation"])
        rows.append(row)
    for arm in ARMS:
        completed = [row for row in rows if row["phase"] == "factorial" and row["arm"] == arm]
        aggregates[arm] = {"completed_seeds": [row["seed"] for row in completed],
                           "metrics": {key: moments([row["metrics"][key] for row in completed])
                                       for key in completed[0]["metrics"]} if completed else {}}
    effects = {}
    for name, left, right in (
        ("community_without_neighbors", "shared_community", "shared_root"),
        ("community_with_neighbors", "neighbor_community", "neighbor_root"),
        ("neighbors_at_root", "neighbor_root", "shared_root"),
        ("neighbors_at_community", "neighbor_community", "shared_community")):
        a = {row["seed"]: row["metrics"] for row in rows if row["phase"] == "factorial" and row["arm"] == left}
        b = {row["seed"]: row["metrics"] for row in rows if row["phase"] == "factorial" and row["arm"] == right}
        paired = sorted(a.keys() & b.keys())
        effects[name] = {"paired_seeds": paired, "difference": {
            key: moments([a[seed][key] - b[seed][key] for seed in paired]) for key in a[paired[0]]} if paired else {}}
    summary = {"updated_at": now(), "phase": manifest["phase"], "completed": len(rows),
               "total_jobs": len(manifest["jobs"]), "rows": rows, "factorial": aggregates, "effects": effects,
               "aggregation": "three sampling-stream means within each training run, then three seed means/sample std"}
    write_json(OUTPUT / "summary.json", summary)
    lines = ["# Neighbor/background comparison and Toys factorial", "",
             "Selection uses validation only. Three sampling streams are not independent training runs.", "",
             "| Phase | Dataset | Variant/arm | Seed | Best val NDCG@10 | Test NDCG@10 | Tail NDCG@10 | Tail exposure@10 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['phase']} | {row['dataset']} | {row['variant'] if row['phase']=='screening' else row['arm']} | "
                     f"{row['seed']} | {row['validation']['ndcg@10']:.6f} | {m['ndcg@10']:.6f} | {m['tail_ndcg@10']:.6f} | {m['tail_exposure@10']:.2%} |")
    lines += ["", "Factorial seed means and sample standard deviations:", "",
              "| Arm | Completed seeds | HR@5 | HR@10 | NDCG@5 | NDCG@10 | Tail NDCG@10 | Tail exposure@10 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, aggregate in aggregates.items():
        if not aggregate["metrics"]:
            continue
        values = []
        for key in (*SCORES, "tail_ndcg@10", "tail_exposure@10"):
            m = aggregate["metrics"][key]
            values.append(f"{m['mean']:.6f} +/- {m['std']:.6f}" if m["std"] is not None else f"{m['mean']:.6f} (one seed)")
        lines.append(f"| {arm} | {aggregate['completed_seeds']} | " + " | ".join(values) + " |")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def coordinate():
    control = OUTPUT / "execution_control.json"
    if control.exists() and read_json(control).get("hold_new_jobs", False):
        raise RuntimeError("User has held subsequent jobs and automatic selection; existing workers finish independently")
    manifest = read_json(OUTPUT / "manifest.json")
    lock = OUTPUT / "launcher.lock"
    if lock.exists():
        old_pid = int(lock.read_text())
        if pid_alive(old_pid):
            raise RuntimeError(f"Coordinator already active: {old_pid}")
        lock.unlink()
    for job in manifest["jobs"]:
        if job["state"] == "running" and pid_alive(job.get("pid")):
            raise RuntimeError(f"Existing worker still active: {job['pid']}; do not duplicate it")
        status = Path(job["output"]) / "worker_status.json"
        if status.exists():
            job["state"] = read_json(status)["state"]
        if job["state"] in ("running", "training", "evaluating"):
            # A finished fit can safely finish inference. Partial training remains failed for manual diagnosis.
            job["state"] = "queued" if (Path(job["output"]) / "training.json").exists() else "failed"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    (OUTPUT / "launcher.pid").write_text(str(os.getpid()), encoding="ascii")
    active = {}
    try:
        while True:
            for index, (process, stdout, stderr) in list(active.items()):
                job = manifest["jobs"][index]
                if (Path(job["output"]) / "cancellation.json").exists() and process.poll() is None:
                    process.terminate()
                code = process.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                cancelled = (Path(job["output"]) / "cancellation.json").exists()
                job.update(state="cancelled" if cancelled else "complete" if code == 0 else "failed",
                           exit_code=code, finished_at=now())
                del active[index]
                summarize(manifest)
            for index, job in enumerate(manifest["jobs"]):
                if len(active) >= 2:
                    break
                if job["state"] != "queued":
                    continue
                output = Path(job["output"])
                if (output / "cancellation.json").exists():
                    job["state"] = "cancelled"
                    continue
                stdout = (output / "worker.log").open("a", encoding="utf-8")
                stderr = (output / "worker.err.log").open("a", encoding="utf-8")
                command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker-job", job["spec"]]
                process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                job.update(state="running", pid=process.pid, command=command, started_at=now())
                active[index] = process, stdout, stderr
            phase_jobs = [job for job in manifest["jobs"] if job["phase"] == manifest["phase"]]
            done = not active and all(job["state"] not in ("queued", "running") for job in phase_jobs)
            failed = any(job["state"] in ("failed", "cancelled") for job in phase_jobs)
            if done and manifest["phase"] == "screening" and not failed:
                prepare_factorial(manifest)
                summarize(manifest)
                continue
            state = ("failed" if failed else "complete") if done else "running"
            for job in manifest["jobs"]:
                job["progress"] = progress(job["output"])
            write_json(OUTPUT / "manifest.json", manifest)
            write_json(OUTPUT / "status.json", {"state": state, "phase": manifest["phase"],
                       "coordinator_pid": os.getpid(), "updated_at": now(), "jobs": manifest["jobs"]})
            if done:
                summarize(manifest)
                return
            time.sleep(5)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker-job", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker_job:
        worker(args.worker_job)
    elif args.summary_only:
        summarize(read_json(OUTPUT / "manifest.json"))
    else:
        coordinate()
