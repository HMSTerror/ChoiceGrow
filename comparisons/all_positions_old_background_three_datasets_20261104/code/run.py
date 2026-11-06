"""Grow entry point: prepare data, construct neighbors and scopes, train, and test the best checkpoint."""
# Note: docs/decisions/implemented/architecture/2026-11-03-paper-modules.md
import argparse
import json
from pathlib import Path

import torch

from data import ALIASES, prepare, write_json
from artifacts import build_artifacts
from train import Trainer
from evaluate import Evaluator
from configuration import load_config


ROOT = Path(__file__).resolve().parent
STAGES = ("prepare", "artifacts", "train", "evaluate")


def local_path(value):
    """Resolve relative paths against the project root, regardless of the caller's working directory."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    """Merge explicit CLI overrides and dispatch stages; mathematical operations live in their own modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",
                        choices=(*ALIASES, "Toys", "Sports", "Beauty", "Steam", "ML1M", "ATG", "ASO"))
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", help="Output directory; relative paths are resolved against the Grow directory")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--device", help="auto/cpu/cuda:0; omit to use the configuration file")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, help="Epoch limit; 0 uses validation early stopping only")
    parser.add_argument("--initial-shares", type=float, nargs=2, metavar=("COMMUNITY", "ROOT"),
                        help="Override initial community/catalog shares; they must sum to 1; omit to use the configuration file")
    parser.add_argument("--learnable-rates", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable learned rates; --no-learnable-rates disables them; omit to use the configuration file")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None,
                        help="Request CUDA TF32 acceleration; omit to use the configuration file (default: false)")
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None,
                        help="Run a one-epoch pipeline check on a small subset; omit to use the configuration file")
    args = parser.parse_args()
    # Apply only explicit overrides, so argparse's False cannot overwrite a configured True.
    try:
        config = load_config(local_path(args.config), initial_shares=args.initial_shares,
                             learnable_rates=args.learnable_rates)
    except ValueError as error:
        parser.error(str(error))
    config["dataset"] = ALIASES[(args.dataset or config.get("dataset", "Toys")).lower()]
    config["device"] = args.device or config.get("device", "auto")
    config["smoke"] = config.get("smoke", False) if args.smoke is None else args.smoke
    config["method"] = "choicegrow"
    config["experiment"] = "neighbor_community_root"
    config["partition"] = "native_infomap_leaf_communities_and_catalog"
    config["tf32"] = config.get("tf32", False) if args.tf32 is None else args.tf32
    if "sasrec" in config or any(key in config["destination"] for key in
                                  ("candidate_gate", "candidate_mass", "use_sasrec", "mode", "weighting")):
        raise ValueError("Use the main-method config.json; historical configurations belong in experiments/")
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if config["smoke"]:
        config["training"].update(epochs=1, batch_size=8, eval_batch_size=8, validation_interval=1,
                                  validation_streams=1, test_streams=1)
    if config["training"]["epochs"] < 0:
        raise ValueError("Epoch limits must be nonnegative; 0 means early stopping only")
    if config["training"]["validation_interval"] < 1 or config["training"]["patience"] < 1:
        raise ValueError("Validation interval and patience must be positive")
    torch.set_num_threads(config["threads"])
    if config["tf32"]:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    name = f"{'smoke' if config['smoke'] else 'neighbor_community'}_{config['dataset'].lower()}_s{config['seed']}"
    output = local_path(args.output or f"runs/{name}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "resolved_config.json"
    # Bind each output directory to one complete configuration to protect existing experiment results.
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory belongs to different settings; choose a new --output")
    write_json(config_path, config)
    # Dispatch each stage through a stable module interface; --stage supports separate execution.
    actions = {
        "prepare": lambda: prepare(local_path(config.get("data_root", "dataset")), output, config["dataset"], config),
        "artifacts": lambda: build_artifacts(output, config, device),
        "train": lambda: Trainer(output, config, device).fit(),
        "evaluate": lambda: Evaluator(config, device).test_best(output),
    }
    for stage in STAGES if args.stage == "all" else (args.stage,):
        print(f"[{stage}] dataset={config['dataset']} device={device} output={output}", flush=True)
        actions[stage]()


if __name__ == "__main__":
    main()
