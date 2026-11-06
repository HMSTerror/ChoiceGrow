"""Compare reused native community/catalog runs with original random64/random1024/catalog nesting.

Use the current all-position neighbors, prefix-count b and Toys training seeds100/101/102.
Keep the paused old-b suite suspended. No other factorials or automatic method selection run here.
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
SOURCE = ROOT.parent / "all_positions_three_datasets_20261104/runs/main/Toys"
PAUSED = ROOT.parent / "all_positions_old_background_three_datasets_20261104/runs"
SEEDS = (100, 101, 102)
ARMS = ("native_two", "random_three")
INITIAL_SHARES = [.1920355, .0244645, .7835]
REQUIRED = ("training.json", "report.json", "recommendations.npz", "long_tail.json")


def random_partition(item_count, fine_size=64, coarse_size=1024, seed=100):
    """Original permutation/rank construction from Choice/experiments/diffusion.py:36."""
    if coarse_size % fine_size:
        raise ValueError("coarse_size must be a multiple of fine_size for nested blocks")
    order = p.torch.randperm(item_count, generator=p.torch.Generator().manual_seed(seed))
    rank = p.torch.empty_like(order)
    rank[order] = p.torch.arange(item_count)
    return p.torch.stack((rank // fine_size, rank // coarse_size, p.torch.zeros_like(rank)))


def make_config(seed, arm):
    config = p.make_config("Toys", seed, "neighbor_community", "all_positions", "prefix")
    if arm == "random_three":
        config.update(experiment="neighbor_random_three", partition="original_random_64_1024_catalog")
        settings = config["diffusion"]
        settings.update(initial_shares=list(INITIAL_SHARES), fine_size=64, coarse_size=1024,
                        level_rates=[settings["total_budget"] * share for share in INITIAL_SHARES])
    elif arm != "native_two":
        raise ValueError(arm)
    return config


def snapshot_hashes():
    return {path.name: p.sha256(path) for path in sorted((ROOT / "code").iterdir()) if path.is_file()}


def paused_suite_is_stopped():
    """Verify that the old-b workers remain suspended before adding GPU work."""
    import psutil
    pause = p.read_json(PAUSED / "pause.json")
    control = p.read_json(PAUSED / "execution_control.json")
    if not control.get("hold_new_jobs", False) or pause["state"] != "paused":
        raise RuntimeError("The old-background suite is no longer paused")
    for record in pause["processes"]:
        if not psutil.pid_exists(record["pid"]):
            continue
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["create_time"]) < .01 and process.status() != psutil.STATUS_STOPPED:
            raise RuntimeError(f"Previously paused process is active: {process.pid}")


def prepare():
    paused_suite_is_stopped()
    OUTPUT.mkdir(exist_ok=False)
    jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            source = SOURCE / f"seed_{seed}"
            folder = OUTPUT / arm / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=False)
            config = make_config(seed, arm)
            artifact = p.torch.load(source / "structure.pt", map_location="cpu", weights_only=True)
            if arm == "random_three":
                artifact["blocks"] = random_partition(len(artifact["base"]))
            p.torch.save(artifact, folder / "structure.pt")
            p.write_json(folder / "data_source.json", {"path": str(p.prepared_root(source))})
            p.write_json(folder / "resolved_config.json", config)
            p.write_json(folder / "config.json", config)
            p.write_json(folder / "artifacts.json", {
                "statistics_split": "train", "reference": str(source), "background_statistics": "prefix",
                "neighbor_statistics": "all_positions", "partition": config["partition"],
                "partition_seed": 100, "block_counts": [int(level.unique().numel()) for level in artifact["blocks"]],
                "prepared_split_sha256": p.split_hashes(source),
                "random_partition_source": "Choice/experiments/diffusion.py:36" if arm == "random_three" else None})
            job = {"id": f"{arm}/seed_{seed}", "arm": arm, "dataset": "Toys", "seed": seed,
                   "output": str(folder), "state": "queued", "pid": None, "reuse_source": None}
            if arm == "native_two":
                p.reuse_training(source, folder, config, artifact)
                job.update(state="complete", reuse_source=str(source))
            job["spec"] = str(folder / "job.json")
            p.write_json(folder / "job.json", job)
            jobs.append(job)
    p.write_json(OUTPUT / "execution_control.json", {"hold_new_jobs": False, "automatic_selection": False,
                 "authorized_at": p.now(), "resume_old_background_automatically": False})
    manifest = {"created_at": p.now(), "dataset": "Toys", "seeds": list(SEEDS), "arms": list(ARMS),
                "neighbors": "all_positions", "background": "prefix", "random_partition_seed": 100,
                "random_sizes": [64, 1024], "random_initial_shares": INITIAL_SHARES,
                "dynamic_rate_gate": True, "parallel_workers": 2, "entries": 6,
                "reused_training_runs": 3, "new_training_runs": 3,
                "snapshot_sha256": snapshot_hashes(), "jobs": jobs}
    save_status(manifest, "prepared")
    summarize(manifest)
    print(json.dumps({"state": "prepared", "reused": 3, "new_fits": 3,
                      "random_initial_shares": INITIAL_SHARES}), flush=True)


def summarize(manifest):
    rows = []
    for job in manifest["jobs"]:
        folder = Path(job["output"])
        if job["state"] != "complete" or not all((folder / name).exists() for name in REQUIRED):
            continue
        tail, report, training = (p.read_json(folder / name) for name in
                                  ("long_tail.json", "report.json", "training.json"))
        assert report["best_epoch"] == training["best_epoch"] and report["test"]["streams"] == 3
        rows.append({"arm": job["arm"], "seed": job["seed"], "metrics": p.flatten(tail),
                     "checkpoint_epoch": report["best_epoch"], "test_rate_shares": report.get("test_rate_shares"),
                     "reuse_source": job["reuse_source"], "output": str(folder)})
    groups = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        groups[arm] = {"completed_seeds": sorted(row["seed"] for row in selected),
                       "metrics": {key: p.moments([row["metrics"][key] for row in selected])
                                   for key in selected[0]["metrics"]} if selected else {}}
    by_arm = {arm: {row["seed"]: row for row in rows if row["arm"] == arm} for arm in ARMS}
    paired = sorted(by_arm["native_two"].keys() & by_arm["random_three"].keys())
    differences = {key: p.moments([by_arm["random_three"][seed]["metrics"][key] -
                                  by_arm["native_two"][seed]["metrics"][key] for seed in paired])
                   for key in rows[0]["metrics"]} if paired else {}
    p.write_json(OUTPUT / "summary.json", {"updated_at": p.now(), "rows": rows, "groups": groups,
                 "paired_seeds": paired, "random_minus_native": differences,
                 "aggregation": "three sampling-stream means per model; three training-seed means/sample std"})
    lines = ["# Toys all-position neighbors + prefix b: native two levels vs random three levels", "",
             "The comparison changes both the partition and its depth. It does not isolate partition quality from depth.", "",
             "| Arm | Seed | Checkpoint epoch | Test NDCG@10 | Tail NDCG@10 | Tail exposure@10 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['arm']} | {row['seed']} | {row['checkpoint_epoch']} | {m['ndcg@10']:.6f} | "
                     f"{m['tail_ndcg@10']:.6f} | {m['tail_exposure@10']:.2%} |")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_status(manifest, state):
    for job in manifest["jobs"]:
        job["progress"] = p.progress(job["output"])
    manifest.update(state=state, updated_at=p.now())
    p.write_json(OUTPUT / "manifest.json", manifest)
    p.write_json(OUTPUT / "status.json", {"state": state, "coordinator_pid": os.getpid(), "updated_at": p.now(),
                 "completed": sum(job["state"] == "complete" for job in manifest["jobs"]),
                 "total": 6, "jobs": manifest["jobs"]})


def coordinate():
    paused_suite_is_stopped()
    manifest = p.read_json(OUTPUT / "manifest.json")
    if snapshot_hashes() != manifest["snapshot_sha256"]:
        raise RuntimeError("Frozen numerical code changed")
    lock = OUTPUT / "launcher.lock"
    if lock.exists():
        if p.pid_alive(int(lock.read_text())):
            raise RuntimeError("This coordinator is already active")
        lock.unlink()
    for job in manifest["jobs"]:
        if p.pid_alive(job.get("pid")):
            raise RuntimeError(f"An existing worker is active: {job['id']}")
        folder = Path(job["output"])
        worker_file = folder / "worker_status.json"
        worker = p.read_json(worker_file) if worker_file.exists() else {}
        if worker.get("state") in ("training", "evaluating") and p.pid_alive(worker.get("pid")):
            raise RuntimeError(f"An existing worker is active: {job['id']}")
        if worker.get("state") == "complete":
            job["state"] = "complete"
        elif job["state"] == "running" or worker.get("state") in ("training", "evaluating", "failed"):
            job["state"] = "queued" if (folder / "training.json").exists() else "failed"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    (OUTPUT / "launcher.pid").write_text(str(os.getpid()), encoding="ascii")
    active = {}
    try:
        while True:
            control = p.read_json(OUTPUT / "execution_control.json")
            for index, (process, stdout, stderr) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                job = manifest["jobs"][index]
                complete = all((Path(job["output"]) / name).exists() for name in REQUIRED)
                job.update(state="complete" if code == 0 and complete else "failed", exit_code=code,
                           finished_at=p.now())
                del active[index]
                summarize(manifest)
            if not control.get("hold_new_jobs", False):
                for index, job in enumerate(manifest["jobs"]):
                    if len(active) >= 2:
                        break
                    if job["state"] != "queued":
                        continue
                    folder = Path(job["output"])
                    stdout = (folder / "worker.log").open("a", encoding="utf-8")
                    stderr = (folder / "worker.err.log").open("a", encoding="utf-8")
                    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker-job", job["spec"]]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    job.update(state="running", pid=process.pid, started_at=p.now(), command=command)
                    active[index] = process, stdout, stderr
            done = not active and all(j["state"] in ("complete", "failed") for j in manifest["jobs"])
            held = not active and control.get("hold_new_jobs", False)
            state = "held" if held else "failed" if done and any(j["state"] == "failed" for j in manifest["jobs"]) else "complete" if done else "running"
            save_status(manifest, state)
            if done or held:
                summarize(manifest)
                return
            time.sleep(5)
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
