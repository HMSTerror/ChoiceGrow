"""Run the user-selected all-position method on three datasets, then the Toys factorial.

The variant is fixed by the user; no metric-based version selection occurs.
Main-dataset fits must all finish training, testing, and long-tail analysis before the factorial starts.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import protocol as p

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs"
PREVIOUS = ROOT.parent / "neighbor_window_ablation_20261103/runs"
DATASETS = ("Toys", "Sports", "Beauty")
SEEDS = (100, 101, 102)
ARMS = ("neighbor_community", "neighbor_root", "shared_community", "shared_root")
NEIGHBORS = "all_positions"
BACKGROUND = "prefix"


def snapshot_hashes():
    """Bind every new fit to the same saved numerical implementation."""
    return {path.name: p.sha256(path) for path in sorted((ROOT / "code").iterdir()) if path.is_file()}


def ready_for_factorial(jobs):
    """The complete main phase includes inference and long-tail outputs, not only fitted weights."""
    main = [job for job in jobs if job["phase"] == "main"]
    return len(main) == 9 and all(job["state"] == "complete" for job in main)


def launchable_jobs(manifest, control):
    """Enforce the phase boundary and any later user hold before scheduling a process."""
    if control.get("hold_new_jobs", False):
        return []
    return [job for job in manifest["jobs"] if job["phase"] == manifest["phase"]
            and job["state"] == "queued"]


def save_manifest(manifest):
    manifest["updated_at"] = p.now()
    p.write_json(OUTPUT / "manifest.json", manifest)


def create_job(phase, dataset, seed, arm, source=None, dependency=None):
    """Freeze one run's inputs; accept reuse only after checking data/configuration/checkpoint identity."""
    relative = Path("main") / dataset / f"seed_{seed}" if phase == "main" else Path("factorial") / arm / f"seed_{seed}"
    output = OUTPUT / relative
    output.mkdir(parents=True, exist_ok=False)
    config = p.make_config(dataset, seed, arm, NEIGHBORS, BACKGROUND)
    artifact = p.prepare_artifact(output, config)
    job = {"id": f"{phase}/{dataset}/{arm}/seed_{seed}", "phase": phase, "dataset": dataset,
           "seed": seed, "arm": arm, "neighbors": NEIGHBORS, "background": BACKGROUND,
           "output": str(output), "reuse_source": None,
           "reuse_dependency": str(dependency) if dependency else None}
    state = "waiting_dependency" if dependency else "queued"
    if source:
        p.reuse_training(source, output, config, artifact)
        job["reuse_source"] = str(source)
        state = "complete" if (output / "long_tail.json").exists() else "queued"
    spec = output / "job.json"
    p.write_json(spec, job)
    return job | {"state": state, "spec": str(spec), "pid": None, "progress": p.progress(output)}


def popularity_report(directory):
    """Save each dataset's own tail boundary instead of copying the Toys threshold elsewhere."""
    training = p.load_split(directory, "train")
    n = p.metadata(directory)["item_count"]
    values = p.np.concatenate((training["seq"].numpy().ravel(), training["target"].numpy()))
    counts = p.np.bincount(values[(values >= 0) & (values < n)], minlength=n)
    boundary = int(p.np.sort(counts)[::-1][p.math.ceil(.2 * n) - 1])
    tail = counts < boundary
    test = p.load_split(directory, "test")
    artifact = p.torch.load(Path(directory) / "structure.pt", map_location="cpu", weights_only=True)
    return {"catalog_items": n, "tail_frequency_strictly_below": boundary,
            "tail_items": int(tail.sum()), "tail_test_targets": int(tail[test["target"].numpy()].sum()),
            "test_examples": len(test["target"]), "partition_seed": 100,
            "community_count": int(artifact["blocks"][0].unique().numel()),
            "split_sha256": p.split_hashes(directory)}


def prepare():
    OUTPUT.mkdir(exist_ok=False)
    jobs = []
    # The current Toys seed100 selected method has completed and is reused exactly once here.
    source = PREVIOUS / "screening/Toys/all_positions__prefix"
    for seed in SEEDS:
        for dataset in DATASETS:
            reuse = source if dataset == "Toys" and seed == 100 else None
            jobs.append(create_job("main", dataset, seed, "neighbor_community", source=reuse))
    for seed in SEEDS:
        for arm in ARMS:
            dependency = OUTPUT / "main/Toys" / f"seed_{seed}" if arm == "neighbor_community" else None
            jobs.append(create_job("factorial", "Toys", seed, arm, dependency=dependency))
    control = {"authorized_at": p.now(), "hold_new_jobs": False,
               "user_fixed_neighbors": NEIGHBORS, "user_fixed_background": BACKGROUND,
               "automatic_selection": False, "automatic_factorial_after_main_complete": True,
               "datasets": list(DATASETS), "seeds": list(SEEDS),
               "reason": "User selected new neighbors plus new b, authorized three datasets first, then Toys factorial."}
    p.write_json(OUTPUT / "execution_control.json", control)
    manifest = {"created_at": p.now(), "phase": "main", "selected_by": "user",
                "neighbors": NEIGHBORS, "background": BACKGROUND, "datasets": list(DATASETS),
                "seeds": list(SEEDS), "parallel_workers": 2, "automatic_selection": False,
                "main_training_runs": 9, "factorial_entries": 12, "unique_training_runs": 18,
                "existing_training_reused": 1, "new_training_runs": 17,
                "main_runs_reused_in_factorial": 3, "training_sample_expansion": False,
                "partition_seed": 100, "initial_shares": [.2165, .7835],
                "learning_rate": .0006, "patience": 150, "validation_interval": 10,
                "validation_streams": 1, "test_streams": 3, "tf32": False,
                "dataset_protocols": {dataset: popularity_report(OUTPUT / "main" / dataset / "seed_100")
                                      for dataset in DATASETS},
                "snapshot_sha256": snapshot_hashes(), "jobs": jobs}
    save_manifest(manifest)
    summarize(manifest)
    p.write_json(OUTPUT / "status.json", {"state": "prepared", "phase": "main", "updated_at": p.now(),
                 "automatic_selection": False, "jobs": jobs})
    print(json.dumps({key: manifest[key] for key in ("phase", "main_training_runs", "factorial_entries",
        "unique_training_runs", "existing_training_reused", "new_training_runs", "dataset_protocols")}), flush=True)


def complete_dependency_reuse(job):
    """Reuse the same seed's Toys main-method training in its factorial entry, without a second fit."""
    source = Path(job["reuse_dependency"])
    if p.read_json(source / "worker_status.json")["state"] != "complete":
        raise RuntimeError(f"Main training/evaluation is not complete: {source}")
    output = Path(job["output"])
    config = p.read_json(output / "resolved_config.json")
    artifact = p.torch.load(output / "structure.pt", map_location="cpu", weights_only=True)
    p.reuse_training(source, output, config, artifact)
    job.update(reuse_source=str(source), state="complete", finished_at=p.now(), progress=p.progress(output))
    p.write_json(job["spec"], {key: value for key, value in job.items() if key not in ("state", "progress")})


def advance_phase(manifest):
    if manifest["phase"] != "main" or not ready_for_factorial(manifest["jobs"]):
        raise RuntimeError("All nine main runs must complete before the Toys factorial")
    for job in manifest["jobs"]:
        if job["phase"] == "factorial" and job["state"] == "waiting_dependency":
            complete_dependency_reuse(job)
    manifest.update(phase="factorial", main_completed_at=p.now(), factorial_started_at=p.now())
    save_manifest(manifest)
    summarize(manifest)
    print(json.dumps({"phase": "factorial", "main_complete": 9,
                      "factorial_main_training_reused": 3, "new_factorial_fits": 9}), flush=True)


def summarize(manifest):
    """Keep main-dataset aggregates and the Toys factorial separate, including long-tail decreases."""
    rows = []
    for job in manifest["jobs"]:
        output = Path(job["output"])
        if not (output / "long_tail.json").exists():
            continue
        tail, report, training = (p.read_json(output / filename) for filename in
                                  ("long_tail.json", "report.json", "training.json"))
        assert report["best_epoch"] == training["best_epoch"] and report["test"]["streams"] == 3
        rows.append({key: job[key] for key in ("id", "phase", "dataset", "seed", "arm", "reuse_source")} | {
            "output": str(output), "metrics": p.flatten(tail), "best_epoch": training["best_epoch"],
            "best_validation": training["best_validation"], "epochs_completed": training["epochs_completed"],
            "training_elapsed_seconds": training["elapsed_seconds"], "test_rate_shares": report.get("test_rate_shares")})

    def aggregate(selected):
        return {"completed_seeds": sorted(row["seed"] for row in selected),
                "metrics": {key: p.moments([row["metrics"][key] for row in selected])
                            for key in selected[0]["metrics"]} if selected else {}}

    main = {dataset: aggregate([row for row in rows if row["phase"] == "main" and row["dataset"] == dataset])
            for dataset in DATASETS}
    factorial = {arm: aggregate([row for row in rows if row["phase"] == "factorial" and row["arm"] == arm])
                 for arm in ARMS}
    effects = {}
    for name, arm, baseline in (
        ("community_with_neighbors", "neighbor_community", "neighbor_root"),
        ("community_without_neighbors", "shared_community", "shared_root"),
        ("neighbors_with_community", "neighbor_community", "shared_community"),
        ("neighbors_at_root", "neighbor_root", "shared_root")):
        left = {row["seed"]: row["metrics"] for row in rows if row["phase"] == "factorial" and row["arm"] == arm}
        right = {row["seed"]: row["metrics"] for row in rows if row["phase"] == "factorial" and row["arm"] == baseline}
        paired = sorted(left.keys() & right.keys())
        effects[name] = {"paired_seeds": paired, "differences": {
            key: p.moments([left[seed][key] - right[seed][key] for seed in paired])
            for key in left[paired[0]]} if paired else {}}
    result = {"updated_at": p.now(), "phase": manifest["phase"], "selected_by": "user",
              "neighbors": NEIGHBORS, "background": BACKGROUND, "completed_entries": len(rows),
              "total_entries": len(manifest["jobs"]), "rows": rows, "main": main,
              "factorial": factorial, "effects": effects,
              "aggregation": "three sampling-stream means per trained model; then training-seed mean and sample std"}
    p.write_json(OUTPUT / "summary.json", result)
    lines = ["# All-position neighbors + prefix background", "",
             "Variant fixed by the user. Seeds100/101/102 are training seeds; three streams share each model.", "",
             "## Completed individual runs", "",
             "| Phase | Dataset | Arm | Seed | Best val NDCG@10 | Best epoch | Test NDCG@10 | Tail NDCG@10 | Tail exposure@10 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['phase']} | {row['dataset']} | {row['arm']} | {row['seed']} | "
                     f"{row['best_validation']['ndcg@10']:.6f} | {row['best_epoch']} | {m['ndcg@10']:.6f} | "
                     f"{m['tail_ndcg@10']:.6f} | {m['tail_exposure@10']:.2%} |")
    metrics = (*p.SCORES, *("tail_" + key for key in p.SCORES),
               "tail_exposure@10", "tail_catalog_coverage@10")
    for title, groups in (("Three-dataset main method", main), ("Toys factorial", factorial)):
        lines += ["", "## " + title, "", "| Group | Seeds | " + " | ".join(metrics) + " |",
                  "| --- | --- | " + " | ".join("---:" for _ in metrics) + " |"]
        for label, group in groups.items():
            if not group["metrics"]:
                continue
            values = []
            for key in metrics:
                value = group["metrics"][key]
                values.append(f"{value['mean']:.6f} +/- {value['std']:.6f}" if value["std"] is not None
                              else f"{value['mean']:.6f} (one seed)")
            lines.append(f"| {label} | {group['completed_seeds']} | " + " | ".join(values) + " |")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_jobs(manifest):
    """Inspect existing workers before a coordinator restart; never duplicate a live fit."""
    for job in manifest["jobs"]:
        output = Path(job["output"])
        worker = output / "worker_status.json"
        status = p.read_json(worker) if worker.exists() else {}
        if job["state"] == "running" and p.pid_alive(job.get("pid")):
            raise RuntimeError(f"Existing worker {job['pid']} is still active; wait before restarting coordinator")
        if status.get("state") in ("training", "evaluating") and p.pid_alive(status.get("pid")):
            raise RuntimeError(f"Worker {status['pid']} is active; do not duplicate it")
        if status.get("state") == "complete":
            if not all((output / name).exists() for name in ("training.json", "report.json", "long_tail.json", "recommendations.npz")):
                raise RuntimeError(f"Incomplete output despite complete status: {output}")
            job["state"] = "complete"
        elif status.get("state") == "cancelled" or (output / "cancellation.json").exists():
            job["state"] = "cancelled"
        elif job["state"] == "waiting_dependency":
            continue
        elif status.get("state") in ("training", "evaluating", "failed") or job["state"] == "running":
            # Only inference after a completed fit may resume automatically.
            job["state"] = "queued" if (output / "training.json").exists() else "failed"


def write_status(manifest, state):
    for job in manifest["jobs"]:
        job["progress"] = p.progress(job["output"])
    save_manifest(manifest)
    p.write_json(OUTPUT / "status.json", {"state": state, "phase": manifest["phase"],
                 "updated_at": p.now(), "coordinator_pid": os.getpid(), "automatic_selection": False,
                 "main_completed": sum(j["phase"] == "main" and j["state"] == "complete" for j in manifest["jobs"]),
                 "factorial_completed": sum(j["phase"] == "factorial" and j["state"] == "complete" for j in manifest["jobs"]),
                 "jobs": manifest["jobs"]})


def coordinate():
    manifest = p.read_json(OUTPUT / "manifest.json")
    if snapshot_hashes() != manifest["snapshot_sha256"]:
        raise RuntimeError("Frozen numerical code changed; do not mix implementations inside the suite")
    lock = OUTPUT / "launcher.lock"
    if lock.exists():
        old = int(lock.read_text())
        if p.pid_alive(old):
            raise RuntimeError(f"Coordinator is already active: {old}")
        lock.unlink()
    refresh_jobs(manifest)
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    (OUTPUT / "launcher.pid").write_text(str(os.getpid()), encoding="ascii")
    active = {}
    try:
        while True:
            control = p.read_json(OUTPUT / "execution_control.json")
            for index, (process, stdout, stderr) in list(active.items()):
                job = manifest["jobs"][index]
                output = Path(job["output"])
                cancelled = (output / "cancellation.json").exists()
                if cancelled and process.poll() is None:
                    process.terminate()
                code = process.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                complete = all((output / filename).exists() for filename in
                               ("training.json", "report.json", "recommendations.npz", "long_tail.json"))
                job.update(state="cancelled" if cancelled else "complete" if code == 0 and complete else "failed",
                           exit_code=code, finished_at=p.now())
                del active[index]
                summarize(manifest)
            if manifest["phase"] == "main" and ready_for_factorial(manifest["jobs"]):
                if control.get("automatic_factorial_after_main_complete", False) and not control.get("hold_new_jobs", False):
                    advance_phase(manifest)
            eligible = {job["id"] for job in launchable_jobs(manifest, control)}
            for index, job in enumerate(manifest["jobs"]):
                if len(active) >= 2:
                    break
                if job["id"] not in eligible:
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
                job.update(state="running", pid=process.pid, command=command, started_at=p.now())
                active[index] = process, stdout, stderr
            phase_jobs = [job for job in manifest["jobs"] if job["phase"] == manifest["phase"]]
            phase_done = not active and all(job["state"] in ("complete", "failed", "cancelled") for job in phase_jobs)
            failed = any(job["state"] in ("failed", "cancelled") for job in phase_jobs)
            held = not active and (control.get("hold_new_jobs", False) or
                (manifest["phase"] == "main" and ready_for_factorial(manifest["jobs"])
                 and not control.get("automatic_factorial_after_main_complete", False)))
            state = "held" if held else "failed" if phase_done and failed else (
                "complete" if phase_done and manifest["phase"] == "factorial" else "running")
            write_status(manifest, state)
            if held or phase_done:
                summarize(manifest)
                print(json.dumps({"state": state, "phase": manifest["phase"]}), flush=True)
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
        p.worker(args.worker_job)
    elif args.summary_only:
        summarize(p.read_json(OUTPUT / "manifest.json"))
    else:
        coordinate()
