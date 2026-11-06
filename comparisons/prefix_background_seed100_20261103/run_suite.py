"""Prepare and launch the three seed-100 background-only comparisons, with two GPU workers."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CODE = ROOT / "code"
OUTPUT = ROOT / "runs"
REFERENCE = PROJECT / "runs/raw_seed100_lr0006_patience150_20261103"
DATASETS = ("Toys", "Beauty", "Sports")
sys.path.insert(0, str(CODE))
from artifacts import build_artifacts, prefix_target_counts
from configuration import load_config
from data import load_split, metadata, prepared_root, write_json
from evaluate import Evaluator
from train import Trainer


def now():
    return datetime.now().astimezone().isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_best(output):
    """Read the full validation history, never confuse the latest epoch with the best."""
    path = Path(output) / "choice_log.jsonl"
    latest, best = 0, None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A worker may currently be appending its last line.
            latest = max(latest, row.get("epoch", 0))
            metrics = row.get("validation")
            if metrics and (best is None or metrics["ndcg@10"] > best["ndcg@10"]):
                best = {"epoch": row["epoch"], "ndcg@10": metrics["ndcg@10"]}
    return {"latest_epoch": latest, "best_validation": best}


def prepare():
    """Freeze settings, reuse baseline inputs, and verify the one-factor change before training."""
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Suite already prepared; inspect its manifest instead of overwriting it")
    defaults = load_config(PROJECT / "config.json")
    jobs = []
    for dataset in DATASETS:
        source, output = REFERENCE / dataset, OUTPUT / dataset
        if not (source / "report.json").exists():
            raise FileNotFoundError(f"Baseline is not complete: {source}")
        config = read_json(source / "resolved_config.json")
        comparable = deepcopy(config)
        comparable["training"].pop("denoiser", None)
        expected = deepcopy(defaults)
        expected.update(dataset=dataset, device="cuda:0", method="choicegrow",
                        experiment="neighbor_community_root", partition="native_infomap_leaf_communities_and_catalog")
        assert comparable == expected, f"Baseline settings differ from current defaults: {dataset}"
        config = comparable
        config.update(data_root=str(PROJECT / "dataset/processed"),
                      comparison_reference=str(source),
                      background_statistics="all_prefix_targets_of_retained_training_sequences")
        output.mkdir(exist_ok=False)
        write_json(output / "config.json", config)
        write_json(output / "resolved_config.json", config)
        write_json(output / "data_source.json", {"path": str(prepared_root(source)),
                                                "reason": "Reuse exactly the baseline's prepared splits"})
        structural = build_artifacts(output, config, torch.device("cpu"))
        training = load_split(output, "train")
        n = metadata(output)["item_count"]
        literal = np.zeros(n, dtype=np.int64)
        for history, target in zip(training["seq"].tolist(), training["target"].tolist()):
            chain = [item for item in history if 0 <= item < n] + [target]
            for label in chain[1:]:
                literal[label] += 1
        np.testing.assert_array_equal(literal, prefix_target_counts(training["seq"], training["target"], n).numpy())
        assert literal.sum() == training["length"].sum().item()
        input_files = {name: sha256(prepared_root(source) / "prepared" / f"{name}.npz")
                       for name in ("train", "val", "test")}
        verification = {"dataset": dataset, "reference": str(source),
                        "prepared_split_sha256": input_files,
                        "training_samples": len(training["target"]),
                        "old_target_count": len(training["target"]),
                        "new_prefix_target_count": int(literal.sum()),
                        "prefix_counts_match_literal_expansion": True,
                        "unchanged_tensors": structural["unchanged_artifact_tensors"],
                        "base_l1_difference": structural["base_l1_difference"],
                        "same_training_settings": True}
        write_json(output / "verification.json", verification)
        jobs.append({"dataset": dataset, "state": "queued", "pid": None, "output": str(output),
                     "command": [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", dataset],
                     "stdout": str(OUTPUT / f"{dataset}.log"),
                     "stderr": str(OUTPUT / f"{dataset}.err.log")})
        print(json.dumps(verification), flush=True)
    write_json(OUTPUT / "manifest.json", {"created_at": now(), "reference_suite": str(REFERENCE),
               "seed": 100, "parallel_workers": 2, "training_sample_expansion": False,
               "change": "background counts only; labels of each nonempty prefix, excluding the first item",
               "smoothing": "normalize(sqrt(count+1))", "jobs": jobs})
    write_json(OUTPUT / "status.json", {"state": "prepared", "updated_at": now(), "jobs": jobs})


def tail_metrics(directory, reference):
    """Compute unchanged long-tail metrics from cached full-catalog recommendations."""
    training = load_split(reference, "train")
    n = metadata(reference)["item_count"]
    items = np.concatenate((training["seq"].numpy().ravel(), training["target"].numpy()))
    counts = np.bincount(items[(items >= 0) & (items < n)], minlength=n)
    boundary = int(np.sort(counts)[::-1][math.ceil(.2 * n) - 1])
    tail = counts < boundary
    testing = load_split(reference, "test")
    with np.load(Path(directory) / "recommendations.npz") as cache:
        top10, target = cache["top10"], cache["target"]
    np.testing.assert_array_equal(target, testing["target"].numpy())
    assert top10.shape == (3, len(target), 10)
    assert top10.min() >= 0 and top10.max() < n
    groups = {"tail": tail[target], "head": ~tail[target]}
    result = {"examples": len(target), "streams": len(top10), "tail_catalog_items": int(tail.sum()),
              "catalog_items": n, "target_groups": {},
              "protocol": {"counts": "training history plus target occurrences", "tail_rule": f"frequency < {boundary}",
                           "boundary_ties": "head", "aggregation": "streams averaged before test rows"}}
    report = read_json(Path(directory) / "report.json")
    for k in (5, 10):
        ranked = top10[:, :, :k]
        hits = ranked == target[None, :, None]
        found = hits.any(-1)
        rank = hits.argmax(-1) + 1
        rows = {f"hr@{k}": found.mean(0), f"ndcg@{k}": (found / np.log2(rank + 1)).mean(0)}
        for key, values in rows.items():
            result[key] = float(values.mean())
            assert abs(result[key] - report["test"][key]) < 1e-7
        result[f"tail_exposure@{k}"] = float(tail[ranked].mean())
        result[f"tail_list_rate@{k}"] = float(tail[ranked].any(-1).mean())
        for group, mask in groups.items():
            result["target_groups"].setdefault(group, {"examples": int(mask.sum())})
            result["target_groups"][group].update({key: float(values[mask].mean()) if mask.any() else None
                                                  for key, values in rows.items()})
    return result


def summarize():
    """Persist paired baseline/comparison metrics as datasets finish; no extra inference is required."""
    results, lines = {}, ["# Seed-100 background-count comparison", "",
        "One training seed; test metrics average three sampling streams.", "",
        "| Dataset | Baseline NDCG@10 | Prefix NDCG@10 | Relative change | Baseline tail NDCG@10 | Prefix tail NDCG@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        reference, output = REFERENCE / dataset, OUTPUT / dataset
        original = read_json(reference / "report.json")
        old_tail = tail_metrics(reference, reference)
        row = {"reference": str(reference), "comparison": str(output),
               "baseline_test": original["test"], "baseline_long_tail": old_tail,
               "progress": log_best(output), "comparison_test": None}
        if (output / "report.json").exists():
            report = read_json(output / "report.json")
            new_tail = tail_metrics(output, reference)
            write_json(output / "long_tail.json", new_tail)
            old, new = original["test"]["ndcg@10"], report["test"]["ndcg@10"]
            row.update(comparison_test=report["test"], comparison_long_tail=new_tail,
                       overall_ndcg10_difference=new-old, overall_ndcg10_relative_percent=100*(new/old-1),
                       tail_ndcg10_difference=new_tail["target_groups"]["tail"]["ndcg@10"]-old_tail["target_groups"]["tail"]["ndcg@10"],
                       tail_exposure10_difference=new_tail["tail_exposure@10"]-old_tail["tail_exposure@10"])
            lines.append(f"| {dataset} | {old:.6f} | {new:.6f} | {100*(new/old-1):+.2f}% | "
                         f"{old_tail['target_groups']['tail']['ndcg@10']:.6f} | {new_tail['target_groups']['tail']['ndcg@10']:.6f} |")
        else:
            lines.append(f"| {dataset} | {original['test']['ndcg@10']:.6f} | pending | pending | "
                         f"{old_tail['target_groups']['tail']['ndcg@10']:.6f} | pending |")
        results[dataset] = row
    write_json(OUTPUT / "comparison.json", {"updated_at": now(), "seed": 100, "datasets": results})
    (OUTPUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def worker(dataset):
    """Train once from scratch using the frozen configuration, then test the best checkpoint."""
    output = OUTPUT / dataset
    config = load_config(output / "config.json")
    torch.set_num_threads(config["threads"])
    if config["tf32"]:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    write_json(output / "worker_status.json", {"state": "training", "pid": os.getpid(), "updated_at": now()})
    Trainer(output, config, device).fit()
    write_json(output / "worker_status.json", {"state": "evaluating", "pid": os.getpid(), "updated_at": now()})
    Evaluator(config, device).test_best(output)
    write_json(output / "long_tail.json", tail_metrics(output, REFERENCE / dataset))
    write_json(output / "worker_status.json", {"state": "complete", "pid": os.getpid(), "updated_at": now()})


def coordinate():
    """Launch each dataset once, at most two concurrently, and keep durable status and comparisons."""
    with (OUTPUT / "launcher.lock").open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    jobs = read_json(OUTPUT / "manifest.json")["jobs"]
    (OUTPUT / "launcher.pid").write_text(str(os.getpid()), encoding="ascii")
    active = {}
    summarize()
    while True:
        for index, (process, stdout, stderr) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stdout.close()
            stderr.close()
            jobs[index].update(state="complete" if code == 0 else "failed", exit_code=code, finished_at=now())
            del active[index]
            summarize()
        for index, job in enumerate(jobs):
            if len(active) >= 2:
                break
            if job["state"] != "queued":
                continue
            stdout = open(job["stdout"], "x", encoding="utf-8")
            stderr = open(job["stderr"], "x", encoding="utf-8")
            process = subprocess.Popen(job["command"], cwd=ROOT, stdout=stdout, stderr=stderr,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            active[index] = process, stdout, stderr
            job.update(state="running", pid=process.pid, started_at=now())
        for job in jobs:
            job["progress"] = log_best(job["output"])
        done = not active and all(job["state"] != "queued" for job in jobs)
        state = ("failed" if any(job["state"] == "failed" for job in jobs) else "complete") if done else "running"
        write_json(OUTPUT / "status.json", {"state": state, "coordinator_pid": os.getpid(), "updated_at": now(), "jobs": jobs})
        if done:
            summarize()
            return
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", choices=DATASETS)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        worker(args.worker)
    elif args.summary_only:
        summarize()
    else:
        coordinate()
