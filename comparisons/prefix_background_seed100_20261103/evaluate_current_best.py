"""Evaluate immutable copies of selected best checkpoints without touching training outputs."""
import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import torch

from run_suite import CODE, OUTPUT, REFERENCE, read_json, tail_metrics
from data import prepared_root, write_json
from evaluate import Evaluator


def frozen_checkpoint(path):
    """Copy one complete checkpoint version, retrying if a training write overlaps the read."""
    for attempt in range(5):
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            time.sleep(.2)
            continue
        try:
            state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError):
            if attempt == 4:
                raise
            time.sleep(.2)
            continue
        return content, state
    raise RuntimeError(f"Checkpoint kept changing during capture: {path}")


def main(datasets=("Toys", "Beauty")):
    started = datetime.now().astimezone()
    root = OUTPUT / "snapshot_tests" / started.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "status.json", {"state": "capturing", "pid": os.getpid(), "started_at": started.isoformat()})
    jobs = []
    for dataset in datasets:
        source = OUTPUT / dataset
        checkpoint, state = frozen_checkpoint(source / "choice_best.pt")
        config = read_json(source / "resolved_config.json")
        destination = root / dataset
        destination.mkdir()
        (destination / "choice_best.pt").write_bytes(checkpoint)
        shutil.copyfile(source / "structure.pt", destination / "structure.pt")
        write_json(destination / "resolved_config.json", config)
        write_json(destination / "data_source.json", {"path": str(prepared_root(source))})
        captured = {"dataset": dataset, "checkpoint_epoch": state["epoch"],
                    "checkpoint_validation": state["validation"], "source": str(source),
                    "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
                    "captured_at": datetime.now().astimezone().isoformat(),
                    "training_settings_changed": False, "selection_split": "val"}
        write_json(destination / "snapshot.json", captured)
        jobs.append((source, destination, config, captured))
        print(json.dumps({"captured": captured, "output": str(destination)}), flush=True)
        del state, checkpoint
    results = {}
    for source, destination, config, captured in jobs:
        dataset = config["dataset"]
        write_json(root / "status.json", {"state": "evaluating", "dataset": dataset, "pid": os.getpid()})
        report_path, predictions = source / "report.json", source / "recommendations.npz"
        report = read_json(report_path) if report_path.exists() else None
        worker_path = source / "worker_status.json"
        completed = worker_path.exists() and read_json(worker_path)["state"] == "complete"
        reuse = (completed and report is not None and predictions.exists()
                 and report["best_epoch"] == captured["checkpoint_epoch"]
                 and report["test"]["streams"] == config["training"]["test_streams"]
                 and hashlib.sha256((source / "choice_best.pt").read_bytes()).hexdigest() == captured["checkpoint_sha256"])
        if reuse:
            shutil.copyfile(report_path, destination / "report.json")
            shutil.copyfile(predictions, destination / "recommendations.npz")
            write_json(destination / "reuse.json", {"source_report": str(report_path),
                       "reason": "Completed three-stream test of the exact same immutable best checkpoint",
                       "checkpoint_sha256": captured["checkpoint_sha256"]})
        else:
            torch.set_num_threads(config["threads"])
            if config["tf32"]:
                torch.set_float32_matmul_precision("high")
                torch.backends.cudnn.allow_tf32 = True
            report = Evaluator(config, torch.device("cuda:0")).test_best(destination)
            torch.cuda.empty_cache()
        tail = tail_metrics(destination, REFERENCE / dataset)
        write_json(destination / "long_tail.json", tail)
        original = read_json(REFERENCE / dataset / "report.json")
        original_tail = tail_metrics(REFERENCE / dataset, REFERENCE / dataset)
        results[dataset] = {**captured, "reused_completed_test": reuse, "test": report["test"],
                            "long_tail": tail, "baseline_test": original["test"],
                            "baseline_long_tail": original_tail,
                            "test_ndcg10_relative_percent": 100*(report["test"]["ndcg@10"]/original["test"]["ndcg@10"]-1)}
        write_json(root / "comparison.json", {"datasets": results, "output": str(root)})
        print(json.dumps({"dataset": dataset, "reused_completed_test": reuse,
                          "checkpoint_epoch": captured["checkpoint_epoch"], "test": report["test"],
                          "long_tail": tail}), flush=True)
    write_json(root / "status.json", {"state": "complete", "pid": os.getpid(),
               "finished_at": datetime.now().astimezone().isoformat(), "datasets": list(results)})
    print(json.dumps({"state": "complete", "output": str(root)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("Toys", "Beauty"), default=("Toys", "Beauty"))
    main(parser.parse_args().datasets)
