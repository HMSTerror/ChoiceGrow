"""Three Toys fits with prefix b, no neighbor weighting, and the catalog Root only."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil
import protocol as p
from diffusion import ChoiceDiffusion

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs"
SOURCE_SUITE = ROOT.parent / "all_positions_three_datasets_20261104"
SOURCE = SOURCE_SUITE / "runs/main/Toys"
RANDOM = ROOT.parent / "toys_random_three_levels_20261104"
NEIGHBOR_ROOT = ROOT.parent / "toys_root_only_20261104"
PAUSED = ROOT.parent / "all_positions_old_background_three_datasets_20261104/runs"
SEEDS = (100, 101, 102)
PYTHON = Path("E:/anaco/python.exe")
REQUIRED = ("training.json", "choice_best.pt", "report.json", "recommendations.npz", "long_tail.json")
METRICS = (*p.SCORES, *("tail_" + key for key in p.SCORES), "tail_exposure@10")


def norm(value):
    return os.path.normcase(os.path.abspath(str(value)))


def process_record(process):
    return {"pid": process.pid, "create_time": process.create_time(),
            "command": process.cmdline(), "status": process.status()}


def same_process(record):
    try:
        actual = process_record(psutil.Process(record["pid"]))
        return (abs(actual["create_time"] - record["create_time"]) < .01
                and [norm(x) for x in actual["command"]] == [norm(x) for x in record["command"]])
    except psutil.NoSuchProcess:
        return False


def python_processes():
    records = []
    for process in psutil.process_iter(["name"]):
        try:
            if "python" in (process.info["name"] or "").lower():
                records.append(process_record(process))
        except psutil.NoSuchProcess:
            continue
    return records


def paused_records():
    pause = p.read_json(PAUSED / "pause.json")
    control = p.read_json(PAUSED / "execution_control.json")
    if pause["state"] != "paused" or not control.get("hold_new_jobs"):
        raise RuntimeError("Old-background suite is no longer held and paused; waiting for resource review")
    result = []
    for saved in pause["processes"]:
        if same_process(saved):
            current = process_record(psutil.Process(saved["pid"]))
            if current["status"] != psutil.STATUS_STOPPED:
                raise RuntimeError(f"Old-background process {saved['pid']} is no longer suspended")
            result.append(current)
    return result


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    values = [int(x.strip()) for x in result.stdout.splitlines()[0].split(",")]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True)
    return {"index": values[0], "used_mib": values[1], "free_mib": values[2], "utilization": values[3],
            "process_ids": [int(line.strip()) for line in apps.stdout.splitlines() if line.strip().isdigit()]}


def resource_snapshot():
    paused = paused_records()
    processes = python_processes()
    gpu = gpu_snapshot()
    active = [record for record in processes if record["status"] != psutil.STATUS_STOPPED
              and record["pid"] != os.getpid()]
    random_processes = [r for r in active if any(norm(x) == norm(RANDOM / "run_suite.py") for x in r["command"])]
    other_coordinators = [r for r in active if "--worker-job" not in r["command"]
                          and any(Path(x).name == "run_suite.py" for x in r["command"])
                          and not any(norm(x) == norm(__file__) for x in r["command"])]
    workloads = [r for r in active if "--worker-job" in r["command"] or r["pid"] in gpu["process_ids"]]
    random_status = p.read_json(RANDOM / "runs/status.json")
    random_pending = [j["id"] for j in random_status["jobs"] if j["state"] not in ("complete", "failed")]
    neighbor_status = p.read_json(NEIGHBOR_ROOT / "runs/status.json")
    neighbor_pending = [j["id"] for j in neighbor_status["jobs"] if j["state"] not in ("complete", "failed")]
    neighbor_processes = [r for r in active if any(norm(x) == norm(NEIGHBOR_ROOT / "run_suite.py")
                                                 for x in r["command"])]
    # Its coordinator does not share our admission lock. Wait for its entire queue to drain
    # so that both coordinators cannot observe one free slot and launch simultaneously.
    reason = ("waiting_for_random_three_queue_to_finish" if random_processes or random_pending else
              "waiting_for_neighbor_root_queue_to_finish" if neighbor_processes or neighbor_pending else
              "waiting_for_other_coordinator" if other_coordinators else
              "waiting_for_global_two_workload_limit" if len(workloads) >= 2 else
              "waiting_for_gpu_memory" if gpu["free_mib"] < 1200 else None)
    return {"checked_at": p.now(), "waiting_reason": reason, "gpu": gpu,
            "active_workloads": workloads, "global_active_workloads": len(workloads),
            "random_pending": random_pending, "random_processes": random_processes,
            "neighbor_root_pending": neighbor_pending, "neighbor_root_processes": neighbor_processes,
            "other_coordinators": other_coordinators, "paused_processes": paused}


def code_hashes():
    return {str(path.relative_to(ROOT)): p.sha256(path) for path in
            [ROOT / "protocol.py", *sorted((ROOT / "code").glob("*.py")), ROOT / "code/config.json"]}


def verify_tail(tail):
    actual = (tail["protocol"]["tail_rule"], tail["catalog_items"], tail["tail_catalog_items"],
              tail["target_groups"]["tail"]["examples"], tail["examples"], tail["streams"])
    expected = ("frequency < 12", 11924, 9367, 1035, 1942, 3)
    if actual != expected:
        raise ValueError(f"Toys tail/data mismatch: {actual}; expected {expected}. Do not alter the threshold.")


def verify_completed(folder, root=False):
    folder = Path(folder)
    if not all((folder / name).exists() for name in REQUIRED):
        raise ValueError(f"Incomplete result: {folder}")
    training, report, tail = [p.read_json(folder / name) for name in ("training.json", "report.json", "long_tail.json")]
    verify_tail(tail)
    if report["best_epoch"] != training["best_epoch"] or report["test"]["streams"] != 3:
        raise ValueError("Training checkpoint/test stream mismatch")
    if p.progress(folder)["best_validation"] != {"epoch": training["best_epoch"],
                                                "ndcg@10": training["best_validation"]["ndcg@10"]}:
        raise ValueError("Full validation log does not select the reported checkpoint")
    if root and (report["dynamic_rate_gate"] or report.get("test_rate_shares") is not None):
        raise ValueError("Root must have no learned community shares")
    for key in p.SCORES:
        if abs(tail[key] - report["test"][key]) >= 1e-7:
            raise ValueError("Long-tail recomputation disagrees with the full-catalog test")
    return {"seed": report["seed"], "checkpoint_epoch": report["best_epoch"],
            "metrics": {key: p.flatten(tail)[key] for key in METRICS}, "output": str(folder)}


def prepare():
    if norm(sys.executable) != norm(PYTHON):
        raise RuntimeError("Use E:/anaco/python.exe")
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("This Root queue has already been prepared")
    OUTPUT.mkdir(exist_ok=True)
    snapshot = resource_snapshot()
    checks, jobs = [], []
    for seed in SEEDS:
        source = SOURCE / f"seed_{seed}"
        original_config = p.read_json(source / "resolved_config.json")
        original = p.torch.load(source / "structure.pt", map_location="cpu", weights_only=True)
        p.verify_reuse(source, source, original_config, original)
        verify_completed(source)
        source_tail = p.tail_metrics(source)
        verify_tail(source_tail)
        assert original_config["seed"] == seed and original_config["neighbor_statistics"] == "all_positions"
        assert original_config["background_statistics"] == "prefix" and original_config["device"] == "cuda:0"
        assert original_config["destination"]["neighbor_lags"] == 3
        assert original_config["training"]["learning_rate"] == .0006
        assert original_config["training"]["patience"] == 150
        assert original_config["training"]["validation_interval"] == 10
        assert original_config["training"]["test_streams"] == 3
        assert math.isclose(original_config["diffusion"]["total_budget"], -math.log(.1), rel_tol=0., abs_tol=1e-14)
        assert original_config["diffusion"]["reset_rate"] == .035
        config = deepcopy(original_config)
        config.update(experiment="shared_root", partition="catalog_root_only")
        config["destination"]["neighbor_mix"] = 0.
        config["diffusion"].update(level_rates=[config["diffusion"]["total_budget"]], dynamic_rate_gate=False)
        del config["diffusion"]["initial_shares"]
        expected = deepcopy(config)
        expected.update(experiment=original_config["experiment"], partition=original_config["partition"])
        expected["destination"]["neighbor_mix"] = original_config["destination"]["neighbor_mix"]
        for key in ("level_rates", "dynamic_rate_gate", "initial_shares"):
            expected["diffusion"][key] = original_config["diffusion"][key]
        assert expected == original_config
        folder = OUTPUT / "shared_root" / f"seed_{seed}"
        if any((folder / name).exists() for name in ("choice_best.pt", "choice_newst.pt", "choice_log.jsonl", "training.json")):
            raise FileExistsError(f"Refusing to overwrite a started fit: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        p.write_json(folder / "data_source.json", {"path": str(p.prepared_root(source).resolve())})
        training = p.load_split(folder, "train")
        n = p.metadata(folder)["item_count"]
        counts = p.prefix_counts(training, n)
        base = (counts.float() + 1).sqrt()
        base = base / base.sum()
        assert p.torch.equal(base, original["base"])
        ids, weights = p.build_neighbors(training["seq"], training["target"], n,
            config["destination"]["neighbor_count"], 3, config["destination"]["neighbor_self_prior"], "all_positions")
        assert p.torch.equal(ids, original["neighbor_ids"])
        assert p.torch.equal(weights, original["neighbor_weights"])
        p.np.save(folder / "prefix_target_counts.npy", counts.numpy())
        artifact = {key: value.clone() for key, value in original.items()}
        artifact["blocks"] = p.torch.zeros((1, n), dtype=p.torch.long)
        p.torch.save(artifact, folder / "structure.pt")
        p.write_json(folder / "config.json", config)
        p.write_json(folder / "resolved_config.json", config)
        resolved = p.load_config(folder / "resolved_config.json")
        assert config == resolved
        assert p.split_hashes(folder) == p.split_hashes(source)
        diffusion = ChoiceDiffusion.from_directory(folder, resolved, p.torch.device("cpu"))
        generator = diffusion.generator
        effective = (1 - .035) * config["diffusion"]["total_budget"]
        assert generator.level_count == 1 and not generator.rate_policy.is_learned
        assert generator.rate_policy.initial_rates == (effective,) and generator.reset_rate == .035
        assert not list(generator.rate_policy.parameters())
        context = generator.context(training["seq"][:3])
        assert p.torch.equal(context.rho, original["base"].expand(3, -1))
        current = training["target"][:3]
        incoming = generator.incoming_rates(current, context)
        expected_incoming = (effective * context.rho.gather(1, current[:, None]) /
            context.block_mass[0].gather(1, current[:, None]) + .035 * generator.base[current, None]).expand_as(incoming).clone()
        expected_incoming.scatter_(1, current[:, None], 0.)
        p.torch.testing.assert_close(incoming, expected_incoming)
        probabilities = diffusion.kernel.probability(current, p.torch.tensor([.001, 1., 10.]), context)
        # The unchanged CPU float32 scatter_add sums 11924 values sequentially;
        # its block-mass rounding is about 3.2e-5 for this b. Check the resulting
        # mass error separately from the much smaller per-item kernel error.
        p.torch.testing.assert_close(probabilities.sum(-1), p.torch.ones(3), atol=5e-5, rtol=0.)
        # With rho=b and one catalog block, Q=((1-alpha)*Lambda+alpha)*(G-I).
        sigma = p.torch.tensor([.001, 1., 10.])
        survival = p.torch.exp(-(effective + .035) * sigma.double())[:, None]
        exact = (1 - survival) * original["base"].double().expand(3, -1)
        exact.scatter_add_(1, current[:, None], survival)
        p.torch.testing.assert_close(probabilities.double(), exact, atol=1e-7, rtol=0.)
        prepared = p.split_hashes(folder)
        check = {"seed": seed, "reference": str(source), "prepared_root": str(p.prepared_root(folder)),
            "prepared_split_sha256": prepared, "base_equal": True, "neighbor_ids_equal": True,
            "neighbor_weights_equal": True, "neighbor_statistics": "all_positions", "background_statistics": "prefix",
            "neighbor_mix": 0., "rho_equals_prefix_b": True, "analytic_root_kernel_verified": True,
            "cpu_float32_kernel_max_absolute_error": float((probabilities.double() - exact).abs().max()),
            "cpu_float32_probability_mass_max_error": float((probabilities.sum(-1) - 1).abs().max()),
            "prefix_count_total": int(counts.sum()), "training_samples": len(training["target"]),
            "training_sample_expansion": False, "blocks_shape": list(artifact["blocks"].shape), "block_counts": [1],
            "total_budget": config["diffusion"]["total_budget"], "effective_root_rate": effective,
            "reset_rate": generator.reset_rate, "identity_coefficient": -effective - .035,
            "formula": "Q=((1-reset_rate)*Lambda+reset_rate)*(G-I); G rows are prefix b; rho=b",
            "rate_policy_parameters": 0, "tail_rule": source_tail["protocol"]["tail_rule"],
            "tail_catalog_items": 9367, "tail_test_targets": 1035, "test_examples": 1942}
        p.write_json(folder / "artifacts.json", check)
        p.write_json(folder / "worker_status.json", {"state": "queued", "updated_at": p.now(),
            "waiting_reason": snapshot["waiting_reason"], "pid": None})
        job = {"id": f"shared_root/seed_{seed}", "dataset": "Toys", "arm": "shared_root",
               "neighbors": "disabled", "background": "prefix", "seed": seed, "output": str(folder), "state": "queued",
               "spec": str(folder / "job.json"), "attempts": 0, "pid": None, "reference": str(source)}
        p.write_json(folder / "job.json", job)
        checks.append(check)
        jobs.append(job)
    manifest = {"created_at": p.now(), "dataset": "Toys", "seeds": list(SEEDS), "new_training_runs": 3,
        "max_global_active_workloads": 2,
        "admission": "drain random-three and neighbor-root queues before shared-root starts",
        "experiment": "prefix b; neighbor_mix=0; one catalog block; no rate gate",
        "python": str(PYTHON), "device": "cuda:0", "snapshot_sha256": code_hashes(), "jobs": jobs}
    p.write_json(OUTPUT / "verification.json", {"verified_at": p.now(), "checks": checks, "resources": snapshot})
    save_status(manifest, "queued", snapshot)
    summarize(manifest)
    print(json.dumps({"state": "prepared", "fits": 3, "waiting_reason": snapshot["waiting_reason"]}), flush=True)


def summarize(manifest):
    arms = {"neighbor_native_two": {seed: verify_completed(SOURCE / f"seed_{seed}") for seed in SEEDS},
            "neighbor_root": {}, "shared_root": {}}
    for seed in SEEDS:
        folder = NEIGHBOR_ROOT / "runs/root" / f"seed_{seed}"
        if all((folder / name).exists() for name in REQUIRED):
            arms["neighbor_root"][seed] = verify_completed(folder, root=True)
    for job in manifest["jobs"]:
        if job["state"] == "complete":
            arms["shared_root"][job["seed"]] = verify_completed(job["output"], root=True)
    groups = {arm: {"completed_seeds": sorted(rows), "metrics": {
        key: p.moments([rows[seed]["metrics"][key] for seed in sorted(rows)])
        for key in METRICS} if rows else {}} for arm, rows in arms.items()}
    comparisons = {}
    for arm in ("neighbor_native_two", "neighbor_root"):
        paired = sorted(set(arms[arm]) & set(arms["shared_root"]))
        comparisons[arm + "_versus_shared_root"] = {"paired_seeds": paired, "metrics": {}}
        for key in METRICS if paired else ():
            reference = [arms["shared_root"][seed]["metrics"][key] for seed in paired]
            values = [arms[arm][seed]["metrics"][key] for seed in paired]
            comparisons[arm + "_versus_shared_root"]["metrics"][key] = {
                "paired_difference": p.moments([a - b for a, b in zip(values, reference)]),
                "ratio_of_paired_means_percent": float(100 * (p.np.mean(values) / p.np.mean(reference) - 1))
                    if p.np.mean(reference) != 0 else None}
    result = {"updated_at": p.now(), "state": "complete" if len(arms["shared_root"]) == 3 else "incomplete",
        "aggregation": "3 sampling streams per fitted model; mean and sample std (ddof=1) across training seeds",
        "rows": [{"arm": arm, **row} for arm, rows in arms.items() for row in rows.values()],
        "groups": groups, "comparisons": comparisons,
        "interpretation": "neighbor_root vs shared_root isolates neighbor weighting; neighbor_native_two vs shared_root changes both neighbor weighting and hierarchy"}
    p.write_json(OUTPUT / "summary.json", result)
    lines = ["# Toys: prefix b, no neighbors, catalog Root ablation", "",
        f"State: {result['state']}; shared_root completed {len(arms['shared_root'])}/3 training seeds.", "",
        "Average the three sampling streams first, then report training-seed means and sample standard deviations (ddof=1).", "",
        "Tail: training histories plus targets, frequency < 12; 9367 items and 1035/1942 test targets.", "",
        "shared_root: rho=b, Q=((1-0.035)*Lambda+0.035)*(G-I), Lambda=-log(0.1); no rate gate.", "",
        result["interpretation"], "",
        "| Arm | Seed | Best epoch | " + " | ".join(METRICS) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in METRICS) + " |"]
    for row in result["rows"]:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['checkpoint_epoch']} | " +
                     " | ".join(f"{row['metrics'][key]:.6f}" for key in METRICS) + " |")
    lines += ["", "| Arm | Metric | Mean | Sample std | Training seeds |",
              "| --- | --- | ---: | ---: | ---: |"]
    for arm, group in groups.items():
        for key, value in group["metrics"].items():
            std = f"{value['std']:.6f}" if value["std"] is not None else "pending"
            lines.append(f"| {arm} | {key} | {value['mean']:.6f} | {std} | {value['n']} |")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_status(manifest, state, resources=None):
    for job in manifest["jobs"]:
        job["progress"] = p.progress(job["output"])
        p.write_json(job["spec"], job)
    manifest.update(state=state, updated_at=p.now())
    p.write_json(OUTPUT / "manifest.json", manifest)
    p.write_json(OUTPUT / "status.json", {"state": state, "updated_at": p.now(),
        "coordinator": process_record(psutil.Process()), "completed": sum(j["state"] == "complete" for j in manifest["jobs"]),
        "total": 3, "waiting_reason": resources.get("waiting_reason") if resources else None,
        "resources": resources, "jobs": manifest["jobs"]})


def matching_workers(spec):
    return [record for record in python_processes() if record["pid"] != os.getpid()
            and "--worker-job" in record["command"] and any(norm(x) == norm(spec) for x in record["command"])]


def worker(spec):
    if matching_workers(spec):
        raise RuntimeError("A worker for this exact job is already alive")
    job = p.read_json(spec)
    try:
        p.worker(spec)
        verify_completed(job["output"], root=True)
    except BaseException:
        p.write_json(Path(job["output"]) / "worker_status.json", {"state": "failed", "pid": os.getpid(),
                     "updated_at": p.now(), "traceback": traceback.format_exc()})
        raise


def coordinate():
    manifest = p.read_json(OUTPUT / "manifest.json")
    if code_hashes() != manifest["snapshot_sha256"]:
        raise RuntimeError("Frozen numerical code changed")
    lock = OUTPUT / "launcher.lock"
    if lock.exists():
        if same_process(p.read_json(lock)):
            raise RuntimeError("Root coordinator is already alive")
        lock.unlink()
    for job in manifest["jobs"]:
        if matching_workers(job["spec"]):
            raise RuntimeError(f"Existing live worker: {job['id']}")
        folder = Path(job["output"])
        if all((folder / name).exists() for name in REQUIRED):
            verify_completed(folder, root=True)
            job["state"] = "complete"
        elif (folder / "training.json").exists():
            job["state"] = "queued"  # Evaluation only, never an optimizer resume.
        elif any((folder / name).exists() for name in ("choice_best.pt", "choice_newst.pt", "choice_log.jsonl")):
            job.update(state="failed", reason="incomplete_training_has_no_optimizer_state")
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(process_record(psutil.Process()), stream)
    p.write_json(OUTPUT / "launcher.json", process_record(psutil.Process()))
    active, previous_state = {}, None
    try:
        while True:
            for index, (process, stdout, stderr) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                job = manifest["jobs"][index]
                folder = Path(job["output"])
                job.update(exit_code=code, finished_at=p.now())
                try:
                    if code:
                        raise RuntimeError(f"Worker exited {code}")
                    verify_completed(folder, root=True)
                    job["state"] = "complete"
                except Exception:
                    # A fit with training.json can retry only inference, after the process has exited.
                    can_retry = ((folder / "training.json").exists() and job["attempts"] < 3
                                 and not matching_workers(job["spec"]))
                    job.update(state="queued" if can_retry else "failed", error=traceback.format_exc(),
                               retry_mode="evaluation_only" if can_retry else None)
                del active[index]
                summarize(manifest)
            try:
                resources = resource_snapshot()
                admission = resources["waiting_reason"] is None
            except Exception:
                resources = {"checked_at": p.now(), "waiting_reason": "resource_verification_failed",
                             "error": traceback.format_exc()}
                admission = False
            if admission:
                for index, job in enumerate(manifest["jobs"]):
                    if job["state"] != "queued":
                        continue
                    # Re-query after each launch; count workers before their CUDA initialization too.
                    resources = resource_snapshot()
                    if resources["waiting_reason"] is not None:
                        break
                    if matching_workers(job["spec"]):
                        raise RuntimeError(f"Unexpected duplicate worker: {job['id']}")
                    folder = Path(job["output"])
                    stdout = (folder / "worker.log").open("a", encoding="utf-8")
                    stderr = (folder / "worker.err.log").open("a", encoding="utf-8")
                    command = [str(PYTHON), "-B", "-u", str(Path(__file__).resolve()), "--worker-job", job["spec"]]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    record = process_record(psutil.Process(process.pid))
                    job.update(state="running", pid=process.pid, process=record, started_at=p.now(),
                               attempts=job["attempts"] + 1, command=command)
                    active[index] = process, stdout, stderr
                    p.write_json(folder / "launch_resources.json", resources)
                    p.write_json(job["spec"], job)
                    print(json.dumps({"event": "worker_started", "job": job["id"], **record}), flush=True)
            done = not active and all(job["state"] in ("complete", "failed") for job in manifest["jobs"])
            state = ("failed" if any(job["state"] == "failed" for job in manifest["jobs"]) else "complete") if done else "running" if active else "queued"
            for job in manifest["jobs"]:
                if job["state"] == "queued":
                    p.write_json(Path(job["output"]) / "worker_status.json", {"state": "queued", "pid": None,
                        "updated_at": p.now(), "waiting_reason": resources["waiting_reason"]})
            save_status(manifest, state, resources)
            message = (state, resources["waiting_reason"], tuple((j["id"], j["state"]) for j in manifest["jobs"]))
            if message != previous_state:
                print(json.dumps({"state": state, "at": p.now(), "waiting_reason": resources["waiting_reason"],
                    "jobs": [{"id": j["id"], "state": j["state"]} for j in manifest["jobs"]]}), flush=True)
                previous_state = message
            if done:
                summarize(manifest)
                return
            time.sleep(15)
    except BaseException:
        p.write_json(OUTPUT / "coordinator_error.json", {"at": p.now(), "traceback": traceback.format_exc()})
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
        worker(args.worker_job)
    elif args.summary_only:
        summarize(p.read_json(OUTPUT / "manifest.json"))
    else:
        coordinate()
