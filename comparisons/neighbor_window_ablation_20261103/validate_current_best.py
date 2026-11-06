"""Validate an immutable snapshot of the current Toys all-position/original-b best checkpoint."""
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import run_suite as suite
import torch
from checkpoint import CheckpointStore, make_denoiser
from data import load_split, prepared_root, write_json
from diffusion import ChoiceDiffusion
from evaluate import Evaluator


def freeze(path):
    """Read one complete checkpoint version without blocking or changing the training worker."""
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
    raise RuntimeError("Checkpoint changed during all capture attempts")


def main():
    started = datetime.now().astimezone()
    output = suite.OUTPUT / "snapshot_validation" / started.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    source = suite.OUTPUT / "screening/Toys/all_positions__target"
    content, state = freeze(source / "choice_best.pt")
    config = suite.read_json(source / "resolved_config.json")
    assert suite.signature(config) == suite.signature(state["config"])
    (output / "choice_best.pt").write_bytes(content)
    shutil.copy2(source / "structure.pt", output / "structure.pt")
    write_json(output / "resolved_config.json", config)
    write_json(output / "data_source.json", {"path": str(prepared_root(source))})
    captured = {"source": str(source), "captured_at": started.isoformat(),
                "checkpoint_sha256": hashlib.sha256(content).hexdigest(), "epoch": state["epoch"],
                "stored_validation": state["validation"], "split": "val", "streams": 1,
                "seed": config["seed"], "training_settings_changed": False}
    write_json(output / "snapshot.json", captured)
    write_json(output / "status.json", {"state": "evaluating", "pid": os.getpid()})
    print(json.dumps({"captured": captured, "output": str(output)}), flush=True)
    del content, state

    torch.set_num_threads(config["threads"])
    if config["tf32"]:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    process = ChoiceDiffusion.from_directory(output, config, device)
    loaded = CheckpointStore(output, config).load_best(process, device)
    model = make_denoiser(output, config, process).to(device)
    model.load_state_dict(loaded["model"])
    result = Evaluator(config, device).evaluate(model, process, load_split(output, "val"), "val", 1)
    differences = {key: result[key] - captured["stored_validation"][key] for key in suite.SCORES}
    if max(abs(value) for value in differences.values()) > 1e-7:
        raise RuntimeError(f"Fresh validation differs from stored checkpoint validation: {differences}")
    write_json(output / "validation.json", {"epoch": captured["epoch"], "split": "val",
               "metrics": result, "difference_from_stored": differences})

    comparisons = []
    for neighbors, background in suite.VARIANTS:
        variant = suite.variant_name(neighbors, background)
        directory = suite.OUTPUT / "screening/Toys" / variant
        if variant == "all_positions__target":
            comparisons.append({"variant": variant, "state": "snapshot_validation",
                                "epoch": captured["epoch"], "validation": result, "source": str(output)})
            continue
        worker = suite.read_json(directory / "worker_status.json")
        if worker["state"] != "complete":
            continue
        training = suite.read_json(directory / "training.json")
        best = suite.progress(directory)["best_validation"]
        assert best == {"epoch": training["best_epoch"], "ndcg@10": training["best_validation"]["ndcg@10"]}
        comparisons.append({"variant": variant, "state": "completed_training",
                            "epoch": training["best_epoch"], "validation": training["best_validation"],
                            "source": str(directory), "evaluation": "stored validation, verified against complete log"})
    baseline = next(row for row in comparisons if row["variant"] == "last_target__target")
    for row in comparisons:
        row["relative_ndcg10_percent_vs_original"] = 100 * (
            row["validation"]["ndcg@10"] / baseline["validation"]["ndcg@10"] - 1)
    write_json(output / "comparison.json", {"dataset": "Toys", "seed": 100, "split": "val", "streams": 1,
               "rows": comparisons, "fresh_evaluation": "all_positions__target", "snapshot": captured})
    write_json(output / "status.json", {"state": "complete", "pid": os.getpid(),
               "finished_at": datetime.now().astimezone().isoformat()})
    print(json.dumps({"state": "complete", "output": str(output), "rows": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
