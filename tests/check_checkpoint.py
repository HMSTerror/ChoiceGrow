"""Read-only legacy-checkpoint comparison of forward distributions, denoiser scores, and reverse sampling.

Example: python -m tests.check_checkpoint --run-dir <existing-run-directory> --device cpu
Write a verification summary only under the requested new output; do not modify old runs or start training.
"""
import argparse
import json
from pathlib import Path

import torch

from checkpoint import CheckpointStore, make_denoiser
from data import load_split, write_json
from diffusion import ChoiceDiffusion
from tests.reference_loader import load_reference


def main():
    """Compare complete inference paths with the same weights, history batch, and random seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("runs/refactor_checks/checkpoint.json"))
    args = parser.parse_args()
    old = load_reference()
    config = json.loads((args.run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    torch.set_num_threads(config["threads"])
    previous = old.train.load_process(args.run_dir, config, device)
    current = ChoiceDiffusion.from_directory(args.run_dir, config, device)
    state = CheckpointStore(args.run_dir, config).load_best(current, device)
    old.train.restore_process(previous, state)
    old_model = old.train.make_denoiser(args.run_dir, config, previous).to(device).eval()
    new_model = make_denoiser(args.run_dir, config, current).to(device).eval()
    old_model.load_state_dict(state["model"], strict=True)
    new_model.load_state_dict(state["model"], strict=True)
    values = load_split(args.run_dir, "test")
    sequence = values["seq"][:args.samples].to(device)
    target = values["target"][:args.samples].to(device)
    sigma = torch.full((len(sequence),), .5, device=device)
    checks = []

    def same(name, a, b):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        checks.append(name)

    with torch.no_grad():
        a, b = previous.context(sequence), current.generator.context(sequence)
        same("rho", a.rho, b.rho)
        same("learned_rates", a.level_rates, b.level_rates)
        same("forward_probability", previous.forward_probability(target, sigma, a), current.kernel.probability(target, sigma, b))
        same("incoming_rates", previous.incoming_rates(target, a), current.generator.incoming_rates(target, b))
        same("terminal_reference", previous.terminal_reference(10., a), current.kernel.terminal_reference(10., b))
        same("denoiser", old_model(sequence, target, sigma), new_model(sequence, target, sigma))
        torch.manual_seed(10291)
        left, left_stats = previous.sample_scores(old_model, sequence, config["training"]["guidance_strength"])
        torch.manual_seed(10291)
        right, right_stats = current.sample_scores(new_model, sequence, config["training"]["guidance_strength"])
        same("reverse_sample_scores", left, right)
        same("top10", left.topk(10, -1).indices, right.topk(10, -1).indices)
        if left_stats != right_stats:
            raise AssertionError("Reverse sampler telemetry differs")
    report = {"status": "passed", "source": str(args.run_dir.resolve()), "checkpoint_epoch": state["epoch"],
              "dataset": config["dataset"], "seed": config["seed"], "device": str(device),
              "samples": len(sequence), "checks": checks, "atol": 0, "rtol": 0,
              "telemetry": right_stats, "torch": torch.__version__}
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
