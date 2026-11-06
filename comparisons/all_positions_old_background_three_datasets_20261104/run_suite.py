"""Finish the current nine new-background runs before testing all-position neighbors with old b.

Use three datasets and three training seeds, with learned community/catalog rates.
The previously planned Toys factorial is cancelled; this suite performs no factorial or selection.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import protocol as p

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs"
AFTER = ROOT.parent / "all_positions_three_datasets_20261104/runs"
REUSE = ROOT.parent / "neighbor_window_ablation_20261103/runs/screening/Toys/all_positions__target"
DATASETS = ("Toys", "Sports", "Beauty")
SEEDS = (100, 101, 102)
REQUIRED = ("training.json", "report.json", "recommendations.npz", "long_tail.json")


def snapshot_hashes():
    return {path.name: p.sha256(path) for path in sorted((ROOT / "code").iterdir()) if path.is_file()}


def dependency_state(directory=AFTER):
    """Require nine complete training/test/tail outputs and release of the previous coordinator."""
    directory = Path(directory)
    control = p.read_json(directory / "execution_control.json")
    if control.get("automatic_factorial_after_main_complete", False):
        raise RuntimeError("The cancelled factorial must remain disabled")
    complete, failed = [], []
    for seed in SEEDS:
        for dataset in DATASETS:
            folder = directory / "main" / dataset / f"seed_{seed}"
            status_file = folder / "worker_status.json"
            status = p.read_json(status_file) if status_file.exists() else {}
            if status.get("state") in ("failed", "cancelled"):
                failed.append(f"{dataset}/seed_{seed}")
            elif status.get("state") == "complete" and all((folder / name).exists() for name in REQUIRED):
                complete.append(f"{dataset}/seed_{seed}")
    status = p.read_json(directory / "status.json")
    coordinator_alive = p.pid_alive(status.get("coordinator_pid"))
    if failed:
        raise RuntimeError(f"Predecessor jobs failed; no follow-up training will start: {failed}")
    return {"ready": len(complete) == 9 and not coordinator_alive,
            "completed": len(complete), "required": 9, "coordinator_alive": coordinator_alive}


def finalize_cancelled_factorial():
    """Update the old queue only after its writer exits; keep all historical files."""
    if not dependency_state()["ready"]:
        raise RuntimeError("Do not edit the predecessor manifest while its coordinator is active")
    manifest = p.read_json(AFTER / "manifest.json")
    for job in manifest["jobs"]:
        if job["phase"] == "factorial":
            if not (Path(job["output"]) / "cancellation.json").exists():
                raise RuntimeError("A factorial entry lacks its user cancellation marker")
            if p.pid_alive(job.get("pid")):
                raise RuntimeError("A cancelled factorial worker is still active")
            job.update(state="cancelled", cancelled_at=p.now())
    manifest.update(updated_at=p.now(), factorial_cancelled=True,
                    replacement_suite=str(ROOT), factorial_entries=0, cancelled_factorial_entries=12,
                    unique_training_runs=9, new_training_runs=8, main_runs_reused_in_factorial=0)
    p.write_json(AFTER / "manifest.json", manifest)
    status = p.read_json(AFTER / "status.json")
    status.update(state="complete", phase="main", updated_at=p.now(), main_completed=9,
                  factorial_cancelled=12, replacement_suite=str(ROOT), jobs=manifest["jobs"])
    p.write_json(AFTER / "status.json", status)
    summary_path = AFTER / "summary.json"
    if summary_path.exists():
        summary = p.read_json(summary_path)
        summary.update(factorial_cancelled=True, cancelled_entries=12, planned_active_entries=9,
                       replacement_suite=str(ROOT))
        p.write_json(summary_path, summary)


def summarize(manifest):
    rows = []
    for job in manifest["jobs"]:
        folder = Path(job["output"])
        if job["state"] != "complete" or not all((folder / name).exists() for name in REQUIRED):
            continue
        tail, report, training = (p.read_json(folder / name) for name in
                                  ("long_tail.json", "report.json", "training.json"))
        assert report["best_epoch"] == training["best_epoch"] and report["test"]["streams"] == 3
        rows.append({"dataset": job["dataset"], "seed": job["seed"], "output": str(folder),
                     "reuse_source": job["reuse_source"], "best_epoch": report["best_epoch"],
                     "metrics": p.flatten(tail), "test_rate_shares": report.get("test_rate_shares")})
    groups = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        groups[dataset] = {"completed_seeds": sorted(row["seed"] for row in selected),
                           "metrics": {key: p.moments([row["metrics"][key] for row in selected])
                                       for key in selected[0]["metrics"]} if selected else {}}
    p.write_json(OUTPUT / "summary.json", {"updated_at": p.now(), "neighbors": "all_positions",
                 "background": "target", "completed": len(rows), "total": 9, "rows": rows,
                 "datasets": groups, "aggregation": "three stream means per model, then training-seed mean/sample std"})
    lines = ["# All-position neighbors + original target-count background", "",
             "The current new-background suite must finish all nine jobs first. No factorial or automatic selection.", "",
             "| Dataset | Seed | Checkpoint epoch | Test NDCG@10 | Tail NDCG@10 | Tail exposure@10 | Community share |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        share = row["test_rate_shares"]["community"]["mean"]
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['best_epoch']} | {m['ndcg@10']:.6f} | "
                     f"{m['tail_ndcg@10']:.6f} | {m['tail_exposure@10']:.2%} | {share:.2%} |")
    lines += ["", "## Training-seed aggregates", "",
              "| Dataset | Completed seeds | Test NDCG@10 | Tail NDCG@10 |",
              "| --- | --- | ---: | ---: |"]
    for dataset, group in groups.items():
        if not group["metrics"]:
            continue
        values = []
        for key in ("ndcg@10", "tail_ndcg@10"):
            value = group["metrics"][key]
            values.append(f"{value['mean']:.6f} +/- {value['std']:.6f}" if value["std"] is not None
                          else f"{value['mean']:.6f} (one seed)")
        lines.append(f"| {dataset} | {group['completed_seeds']} | " + " | ".join(values) + " |")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_status(manifest, state, dependency=None):
    for job in manifest["jobs"]:
        job["progress"] = p.progress(job["output"])
    manifest.update(updated_at=p.now(), state=state)
    p.write_json(OUTPUT / "manifest.json", manifest)
    p.write_json(OUTPUT / "status.json", {"state": state, "updated_at": p.now(),
                 "coordinator_pid": os.getpid(), "completed": sum(j["state"] == "complete" for j in manifest["jobs"]),
                 "total": 9, "after": str(AFTER), "dependency": dependency, "jobs": manifest["jobs"]})


def prepare():
    OUTPUT.mkdir(exist_ok=False)
    expected = p.read_json(AFTER / "manifest.json")["snapshot_sha256"]
    if snapshot_hashes() != expected:
        raise RuntimeError("The numerical code must match the current three-dataset suite")
    jobs = []
    for seed in SEEDS:
        for dataset in DATASETS:
            folder = OUTPUT / "main" / dataset / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=False)
            config = p.make_config(dataset, seed, "neighbor_community", "all_positions", "target")
            artifact = p.prepare_artifact(folder, config)
            job = {"id": f"{dataset}/seed_{seed}", "dataset": dataset, "seed": seed,
                   "arm": "neighbor_community", "neighbors": "all_positions", "background": "target",
                   "output": str(folder), "reuse_source": None, "state": "queued", "pid": None}
            if dataset == "Toys" and seed == 100:
                p.reuse_training(REUSE, folder, config, artifact)
                job.update(reuse_source=str(REUSE), state="complete")
            job["spec"] = str(folder / "job.json")
            p.write_json(folder / "job.json", job)
            jobs.append(job)
    control = {"authorized_at": p.now(), "hold_new_jobs": False, "automatic_selection": False,
               "factorial_enabled": False, "after": str(AFTER), "datasets": list(DATASETS), "seeds": list(SEEDS),
               "reason": "User replaced the cancelled Toys factorial with new-neighbor/old-b runs after the current nine jobs."}
    p.write_json(OUTPUT / "execution_control.json", control)
    manifest = {"created_at": p.now(), "state": "prepared", "after": str(AFTER),
                "neighbors": "all_positions", "background": "target", "datasets": list(DATASETS),
                "seeds": list(SEEDS), "training_runs": 9, "existing_training_reused": 1, "new_training_runs": 8,
                "parallel_workers": 2, "factorial_enabled": False, "automatic_selection": False,
                "snapshot_sha256": expected, "jobs": jobs}
    save_status(manifest, "prepared")
    summarize(manifest)
    print(json.dumps({"state": "prepared", "training_runs": 9, "reused": 1, "new_fits": 8}), flush=True)


def refresh_jobs(manifest):
    """Do not duplicate active fits or silently restart training without optimizer state."""
    for job in manifest["jobs"]:
        folder = Path(job["output"])
        status_file = folder / "worker_status.json"
        worker = p.read_json(status_file) if status_file.exists() else {}
        if p.pid_alive(job.get("pid")) or (worker.get("state") in ("training", "evaluating") and p.pid_alive(worker.get("pid"))):
            raise RuntimeError(f"An existing worker is still active: {job['id']}")
        if worker.get("state") == "complete" and all((folder / name).exists() for name in REQUIRED):
            job["state"] = "complete"
        elif (folder / "cancellation.json").exists():
            job["state"] = "cancelled"
        elif job["state"] == "running" or worker.get("state") in ("training", "evaluating", "failed"):
            job["state"] = "queued" if (folder / "training.json").exists() else "failed"


def coordinate():
    manifest = p.read_json(OUTPUT / "manifest.json")
    if snapshot_hashes() != manifest["snapshot_sha256"]:
        raise RuntimeError("Frozen numerical code changed")
    lock = OUTPUT / "launcher.lock"
    if lock.exists():
        if p.pid_alive(int(lock.read_text())):
            raise RuntimeError("This follow-up coordinator is already active")
        lock.unlink()
    refresh_jobs(manifest)
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    (OUTPUT / "launcher.pid").write_text(str(os.getpid()), encoding="ascii")
    active = {}
    try:
        while True:
            control = p.read_json(OUTPUT / "execution_control.json")
            if control.get("hold_new_jobs", False):
                save_status(manifest, "held")
                return
            dependency = dependency_state()
            save_status(manifest, "waiting_for_previous", dependency)
            if dependency["ready"]:
                finalize_cancelled_factorial()
                break
            time.sleep(10)
        while True:
            control = p.read_json(OUTPUT / "execution_control.json")
            for index, (process, stdout, stderr) in list(active.items()):
                job = manifest["jobs"][index]
                folder = Path(job["output"])
                cancelled = (folder / "cancellation.json").exists()
                if cancelled and process.poll() is None:
                    process.terminate()
                code = process.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                complete = all((folder / name).exists() for name in REQUIRED)
                job.update(state="cancelled" if cancelled else "complete" if code == 0 and complete else "failed",
                           exit_code=code, finished_at=p.now())
                del active[index]
                summarize(manifest)
            if not control.get("hold_new_jobs", False):
                for index, job in enumerate(manifest["jobs"]):
                    if len(active) >= manifest["parallel_workers"]:
                        break
                    if job["state"] != "queued":
                        continue
                    folder = Path(job["output"])
                    if (folder / "cancellation.json").exists():
                        job["state"] = "cancelled"
                        continue
                    stdout = (folder / "worker.log").open("a", encoding="utf-8")
                    stderr = (folder / "worker.err.log").open("a", encoding="utf-8")
                    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker-job", job["spec"]]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    job.update(state="running", pid=process.pid, command=command, started_at=p.now())
                    active[index] = process, stdout, stderr
            done = not active and all(j["state"] in ("complete", "failed", "cancelled") for j in manifest["jobs"])
            failed = any(j["state"] in ("failed", "cancelled") for j in manifest["jobs"])
            held = not active and control.get("hold_new_jobs", False)
            state = "held" if held else "failed" if done and failed else "complete" if done else "running"
            save_status(manifest, state)
            if held or done:
                summarize(manifest)
                return
            time.sleep(5)
    except BaseException:
        save_status(manifest, "failed")
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--worker-job", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker_job:
        p.worker(args.worker_job)
    elif args.summary_only:
        summarize(p.read_json(OUTPUT / "manifest.json"))
    else:
        coordinate()
