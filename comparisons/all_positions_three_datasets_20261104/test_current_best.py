"""Test immutable Sports/Beauty seed100 best checkpoints without modifying active runs."""
import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import protocol as p
import torch

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs"


def freeze(path):
    """Read a complete checkpoint version, retrying if a training save overlaps capture."""
    for attempt in range(5):
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            time.sleep(.2)
            continue
        try:
            state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        except (OSError, EOFError, RuntimeError):
            if attempt == 4:
                raise
            time.sleep(.2)
            continue
        return content, state
    raise RuntimeError(f"Checkpoint changed during every capture attempt: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("Sports", "Beauty"),
                        default=("Sports", "Beauty"))
    args = parser.parse_args()
    started = datetime.now().astimezone()
    root = OUTPUT / "snapshot_tests" / started.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    p.write_json(root / "status.json", {"state": "capturing", "pid": os.getpid(), "started_at": started.isoformat()})
    try:
        jobs = []
        for dataset in args.datasets:
            source = OUTPUT / "main" / dataset / "seed_100"
            content, state = freeze(source / "choice_best.pt")
            config = p.read_json(source / "resolved_config.json")
            assert p.signature(config) == p.signature(state["config"])
            destination = root / dataset
            destination.mkdir()
            (destination / "choice_best.pt").write_bytes(content)
            shutil.copy2(source / "structure.pt", destination / "structure.pt")
            p.write_json(destination / "resolved_config.json", config)
            p.write_json(destination / "data_source.json", {"path": str(p.prepared_root(source))})
            captured = {"dataset": dataset, "seed": config["seed"], "source": str(source),
                        "checkpoint_epoch": state["epoch"], "checkpoint_validation": state["validation"],
                        "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
                        "captured_at": datetime.now().astimezone().isoformat(),
                        "selection_split": "val", "evaluation_split": "test", "streams": 3,
                        "training_settings_changed": False}
            p.write_json(destination / "snapshot.json", captured)
            jobs.append((source, destination, config, captured))
            print(json.dumps({"captured": captured, "output": str(destination)}), flush=True)
            del content, state
        results = {}
        for source, destination, config, captured in jobs:
            dataset = captured["dataset"]
            p.write_json(root / "status.json", {"state": "evaluating", "dataset": dataset, "pid": os.getpid()})
            torch.set_num_threads(config["threads"])
            if config["tf32"]:
                torch.set_float32_matmul_precision("high")
                torch.backends.cudnn.allow_tf32 = True
            report_path, predictions = source / "report.json", source / "recommendations.npz"
            worker_path = source / "worker_status.json"
            completed = worker_path.exists() and p.read_json(worker_path)["state"] == "complete"
            report = p.read_json(report_path) if report_path.exists() else None
            reuse = (completed and report is not None and predictions.exists()
                     and report["best_epoch"] == captured["checkpoint_epoch"]
                     and report["test"]["streams"] == 3
                     and p.sha256(source / "choice_best.pt") == captured["checkpoint_sha256"])
            if reuse:
                shutil.copy2(report_path, destination / "report.json")
                shutil.copy2(predictions, destination / "recommendations.npz")
                p.write_json(destination / "reuse.json", {"source": str(source),
                             "checkpoint_sha256": captured["checkpoint_sha256"],
                             "reason": "Completed three-stream test of the same best checkpoint"})
            else:
                report = p.Evaluator(config, torch.device("cuda:0")).test_best(destination)
                torch.cuda.empty_cache()
            assert report["best_epoch"] == captured["checkpoint_epoch"] and report["test"]["streams"] == 3
            tail = p.tail_metrics(destination)
            p.write_json(destination / "long_tail.json", tail)
            baseline = p.REFERENCE / dataset
            original = p.read_json(baseline / "report.json")
            original_tail = p.tail_metrics(baseline)
            results[dataset] = {**captured, "reused_completed_test": reuse, "test": report["test"],
                                "long_tail": tail, "rate_shares": report.get("test_rate_shares"),
                                "baseline_test": original["test"], "baseline_long_tail": original_tail,
                                "test_ndcg10_relative_percent": 100*(report["test"]["ndcg@10"]/original["test"]["ndcg@10"]-1),
                                "tail_ndcg10_relative_percent": 100*(tail["target_groups"]["tail"]["ndcg@10"]/
                                    original_tail["target_groups"]["tail"]["ndcg@10"]-1)}
            p.write_json(root / "comparison.json", {"datasets": results, "output": str(root),
                         "protocol": "seed100; validation-selected immutable checkpoints; full-catalog test; three streams"})
            print(json.dumps({"dataset": dataset, "epoch": captured["checkpoint_epoch"], "test": report["test"],
                              "long_tail": tail, "reused_completed_test": reuse}), flush=True)
        p.write_json(root / "status.json", {"state": "complete", "pid": os.getpid(),
                     "finished_at": datetime.now().astimezone().isoformat(), "datasets": list(results)})
        print(json.dumps({"state": "complete", "output": str(root)}), flush=True)
    except BaseException:
        p.write_json(root / "status.json", {"state": "failed", "pid": os.getpid(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
