"""Test immutable current-best random-three-level checkpoints while training continues."""
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

import torch
import protocol as p

ROOT = Path(__file__).resolve().parent / "runs"


def freeze(path):
    """Capture a valid, stable save without locking or changing the training checkpoint."""
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
    parser.add_argument("--seeds", nargs="+", type=int, choices=(100, 101, 102), default=(100, 101))
    args = parser.parse_args()
    started = datetime.now().astimezone()
    output = ROOT / "snapshot_tests" / started.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    p.write_json(output / "status.json", {"state": "capturing", "pid": os.getpid(), "started_at": started.isoformat()})
    try:
        jobs = []
        # Freeze all requested seeds before running inference on any of them.
        for seed in args.seeds:
            source = ROOT / "random_three" / f"seed_{seed}"
            content, state = freeze(source / "choice_best.pt")
            config = p.read_json(source / "resolved_config.json")
            assert p.signature(config) == p.signature(state["config"])
            assert config["seed"] == seed and config["training"]["test_streams"] == 3
            destination = output / f"seed_{seed}"
            destination.mkdir()
            (destination / "choice_best.pt").write_bytes(content)
            shutil.copy2(source / "structure.pt", destination / "structure.pt")
            p.write_json(destination / "resolved_config.json", config)
            p.write_json(destination / "data_source.json", {"path": str(p.prepared_root(source))})
            captured = {"dataset": "Toys", "seed": seed, "source": str(source),
                        "checkpoint_epoch": state["epoch"], "checkpoint_validation": state["validation"],
                        "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
                        "captured_at": datetime.now().astimezone().isoformat(),
                        "selection_split": "val", "evaluation_split": "test", "streams": 3,
                        "training_settings_changed": False}
            p.write_json(destination / "snapshot.json", captured)
            jobs.append((destination, config, captured))
            print(json.dumps({"captured": captured, "output": str(destination)}), flush=True)
            del content, state
        results = {}
        for destination, config, captured in jobs:
            seed = captured["seed"]
            p.write_json(output / "status.json", {"state": "evaluating", "seed": seed, "pid": os.getpid()})
            torch.set_num_threads(config["threads"])
            if config["tf32"]:
                torch.set_float32_matmul_precision("high")
                torch.backends.cudnn.allow_tf32 = True
            report = p.Evaluator(config, torch.device("cuda:0")).test_best(destination)
            torch.cuda.empty_cache()
            assert report["best_epoch"] == captured["checkpoint_epoch"] and report["test"]["streams"] == 3
            tail = p.tail_metrics(destination)
            p.write_json(destination / "long_tail.json", tail)
            baseline = ROOT / "native_two" / f"seed_{seed}"
            original = p.read_json(baseline / "report.json")
            original_tail = p.read_json(baseline / "long_tail.json")
            results[str(seed)] = {**captured, "test": report["test"], "long_tail": tail,
                "rate_shares": report["test_rate_shares"], "baseline_source": str(baseline),
                "baseline_epoch": original["best_epoch"], "baseline_test": original["test"],
                "baseline_long_tail": original_tail,
                "test_ndcg10_relative_percent": 100 * (report["test"]["ndcg@10"] / original["test"]["ndcg@10"] - 1),
                "tail_ndcg10_relative_percent": 100 * (tail["target_groups"]["tail"]["ndcg@10"] /
                                                       original_tail["target_groups"]["tail"]["ndcg@10"] - 1)}
            p.write_json(output / "comparison.json", {"seeds": results, "output": str(output),
                "protocol": "Validation-selected immutable checkpoints; full-catalog test; three sampling streams per seed"})
            print(json.dumps({"seed": seed, "epoch": captured["checkpoint_epoch"], "test": report["test"],
                              "long_tail": tail, "rate_shares": report["test_rate_shares"]}), flush=True)
        p.write_json(output / "status.json", {"state": "complete", "pid": os.getpid(),
            "finished_at": datetime.now().astimezone().isoformat(), "seeds": list(results)})
        print(json.dumps({"state": "complete", "output": str(output)}), flush=True)
    except BaseException:
        p.write_json(output / "status.json", {"state": "failed", "pid": os.getpid(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
