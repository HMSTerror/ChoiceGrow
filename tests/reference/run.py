"""Train ChoiceGrow with history neighbors and native-community/catalog scopes."""
# Note: docs/decisions/implemented/architecture/2026-11-02-neighbor-real-main.md
import argparse
import json
from pathlib import Path

import torch

from data import ALIASES, prepare, write_json
from train import build_artifacts, final_evaluation, train_diffusion


ROOT = Path(__file__).resolve().parent
STAGES = ("prepare", "artifacts", "train", "evaluate")


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Toys",
                        choices=(*ALIASES, "Toys", "Sports", "Beauty", "Steam", "ML1M", "ATG", "ASO"))
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", help="A run directory, relative to this project")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, help="Diffusion epoch limit; 0 means validation early stopping only")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 on supported CUDA GPUs")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(local_path(args.config).read_text(encoding="utf-8"))
    config["dataset"] = ALIASES[args.dataset.lower()]
    config["smoke"] = args.smoke
    config["method"] = "choicegrow"
    config["experiment"] = "neighbor_community_root"
    config["partition"] = "native_infomap_leaf_communities_and_catalog"
    config["tf32"] = args.tf32
    if "sasrec" in config or any(key in config["destination"] for key in
                                  ("candidate_gate", "candidate_mass", "use_sasrec", "mode", "weighting")):
        raise ValueError("Use the main-method config.json; historical configurations belong in experiments/")
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.smoke:
        config["training"].update(epochs=1, batch_size=8, eval_batch_size=8, validation_interval=1,
                                  validation_streams=1, test_streams=1)
    if config["training"]["epochs"] < 0:
        raise ValueError("Epoch limits must be nonnegative; 0 means early stopping only")
    if config["training"]["validation_interval"] < 1 or config["training"]["patience"] < 1:
        raise ValueError("Validation interval and patience must be positive")
    torch.set_num_threads(config["threads"])
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    name = f"{'smoke' if args.smoke else 'neighbor_community'}_{config['dataset'].lower()}_s{config['seed']}"
    output = local_path(args.output or f"runs/{name}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "resolved_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory belongs to different settings; choose a new --output")
    write_json(config_path, config)
    actions = {
        "prepare": lambda: prepare(ROOT / "dataset", output, config["dataset"], config),
        "artifacts": lambda: build_artifacts(output, config, device),
        "train": lambda: train_diffusion(output, config, device),
        "evaluate": lambda: final_evaluation(output, config, device),
    }
    for stage in STAGES if args.stage == "all" else (args.stage,):
        print(f"[{stage}] dataset={config['dataset']} device={device} output={output}", flush=True)
        actions[stage]()


if __name__ == "__main__":
    main()
