"""Validation and testing with isolated random streams, full-catalog ranking, and validation-selected weights.

Evaluate a trained model with multiple sampling streams, then average their individual metrics.
Streams share one trained model and must not be reported as independent training seeds.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import loader, load_split, write_json
from diffusion import ChoiceDiffusion
from checkpoint import CheckpointStore, make_denoiser


def ranking_totals(scores, target):
    """Sum batch HR/NDCG@5/10 from scores [B,N] and one ground-truth target [B] per sample.

    Each sample has one relevant target; a hit at rank r contributes 1/log2(r+1) to NDCG.
    A missing target has found=False and contributes zero, even if argmax returns position zero.
    """
    ranked = scores.topk(min(10, scores.shape[1]), -1).indices
    matches = ranked == target[:, None]
    rank = matches.to(torch.int64).argmax(-1) + 1
    found = matches.any(-1)
    result = {}
    for cutoff in (5, 10):
        hit = found & (rank <= cutoff)
        result[f"hr@{cutoff}"] = float(hit.sum())
        result[f"ndcg@{cutoff}"] = float((hit / torch.log2(rank.float() + 1)).sum())
    return result


@contextmanager
def evaluation_rng(seed, split, stream, device):
    """Derive a seed from training seed, split, and stream; restore the training RNG state on exit."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        payload = f"{seed}:{split}:{stream}".encode("utf-8")
        torch.manual_seed(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1))
        yield


class Evaluator:
    """Share the same evaluation protocol between training-time validation and final testing."""
    def __init__(self, config, device):
        """Store configuration and device; callers explicitly supply the model and data."""
        self.config, self.device = config, device

    @torch.no_grad()
    def evaluate(self, model, process, values, split, streams, predictions_path=None):
        """Run reverse sampling and full-catalog ranking, returning stream means and individual stream metrics.

        Optionally save top10 [streams,samples,10] and targets for later analyses such as long-tail evaluation.
        Rank only actual catalog items, exclude padding, and preserve the existing history-item handling protocol.
        """
        config, device = self.config, self.device
        model.eval()
        results, predictions = [], []
        for stream in range(streams):
            totals, substeps, cap_count, batches = {}, 0., 0, []
            with evaluation_rng(config["seed"], split, stream, device):
                for sequence, _, target in loader(values, config["training"]["eval_batch_size"]):
                    scores, telemetry = process.sample_scores(model, sequence.to(device),
                                                             config["training"]["guidance_strength"])
                    if scores.shape[1] != process.item_count or not torch.isfinite(scores).all():
                        raise FloatingPointError("Expected finite full-catalog ranking scores")
                    scores = scores[:, :model.catalog_count]
                    # Rank the full catalog; do not mix sampled-candidate evaluation with the main method's results.
                    if predictions_path is not None:
                        batches.append(scores.topk(min(10, model.catalog_count), -1).indices.cpu().numpy())
                    for key, value in ranking_totals(scores, target.to(device)).items():
                        totals[key] = totals.get(key, 0.) + value
                    substeps += telemetry["mean_reverse_substeps"] * len(target)
                    cap_count += telemetry["exit_cap_activations"]
            if predictions_path is not None:
                predictions.append(np.concatenate(batches))
            results.append({**{key: value / len(values["target"]) for key, value in totals.items()},
                            "mean_reverse_substeps": substeps / len(values["target"]),
                            "exit_cap_activations": cap_count})
        if predictions_path is not None:
            np.savez_compressed(predictions_path, top10=np.stack(predictions), target=values["target"].numpy())
        metrics = {key: sum(row[key] for row in results) / streams for key in results[0]}
        return {**metrics, "examples": len(values["target"]), "streams": streams, "per_stream": results}

    @torch.no_grad()
    def rate_statistics(self, process, values):
        """Summarize user-specific community shares and catalog means; return None for fixed rates.

        Divide effective rates by each user's total rate; the common (1-alpha) factor cancels.
        The standard deviation measures differences between users, not independent training runs.
        """
        config, device = self.config, self.device
        if not process.generator.rate_policy.is_learned:
            return None
        shares = []
        for sequence, _, _ in loader(values, config["training"]["eval_batch_size"]):
            rates = process.generator.context(sequence.to(device)).level_rates
            shares.append((rates / rates.sum(-1, keepdim=True)).cpu())
        shares = torch.cat(shares).numpy()
        if shares.shape[1] == 3:
            result = {name: {"mean": float(shares[:, level].mean()), "std": float(shares[:, level].std()),
                             "min": float(shares[:, level].min()), "max": float(shares[:, level].max())}
                      for level, name in enumerate(("fine", "coarse", "catalog"))}
            result["root_mean"] = result["catalog"]["mean"]
            return result
        return {"community": {"mean": float(shares[:, 0].mean()), "std": float(shares[:, 0].std()),
                               "min": float(shares[:, 0].min()), "max": float(shares[:, 0].max())},
                "root_mean": float(shares[:, 1].mean())}

    def test_best(self, output):
        """Load the EMA denoiser and gate from choice_best.pt, then write test results and recommendation caches.

        Do not select epochs on the test set or retrain; data_source.json may reference shared prepared data.
        """
        config, device = self.config, self.device
        process = ChoiceDiffusion.from_directory(output, config, device)
        state = CheckpointStore(output, config).load_best(process, device)
        model = make_denoiser(output, config, process).to(device)
        model.load_state_dict(state["model"])
        testing = load_split(output, "test")
        result = self.evaluate(model, process, testing, "test", config["training"]["test_streams"],
                                    predictions_path=Path(output) / "recommendations.npz")
        report = {"dataset": config["dataset"], "seed": config["seed"], "smoke": config["smoke"],
                  "method": "choicegrow", "experiment": config.get("experiment", "neighbor_community_root"),
                  "partition": config.get("partition", "native_infomap_leaf_communities_and_catalog"),
                  "evaluation": "full_catalog", "best_epoch": state["epoch"], "selection_split": "val",
                  "objective": state["objective"], "neighbor_mix": config["destination"]["neighbor_mix"], "test": result,
                  "dynamic_rate_gate": process.generator.rate_policy.is_learned,
                  "test_rate_shares": self.rate_statistics(process, testing)}
        write_json(Path(output) / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report
