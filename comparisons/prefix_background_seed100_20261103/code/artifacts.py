"""Comparison-only background construction; reuse all other baseline artifacts exactly."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from data import load_split, metadata, write_json


def prefix_target_counts(sequence, targets, item_count):
    """Count each next-item label once when expanding a retained sequence into prefixes.

    History [A,B,C,D] and target E yield labels B,C,D,E. The first valid history
    item and padding contribute nothing; repeated labels count once per occurrence.
    Prefixes are used only to estimate the background, not as new training samples.
    """
    valid = (sequence >= 0) & (sequence < item_count)
    first = valid & (valid.long().cumsum(dim=1) == 1)
    labels = torch.cat((sequence[valid & ~first], targets))
    return torch.bincount(labels, minlength=item_count)


def build_artifacts(output, config, device):
    """Replace only base in an existing baseline's structural artifacts.

    Reusing neighbors and partition labels guarantees the same graph structure.
    The caller references exactly the baseline's prepared train/validation/test.
    """
    output, reference = Path(output), Path(config["comparison_reference"])
    n = metadata(output)["item_count"]
    training = load_split(output, "train")
    counts = prefix_target_counts(training["seq"], training["target"], n)
    original = torch.load(reference / "structure.pt", map_location="cpu", weights_only=True)
    base = (counts.float() + 1).sqrt()
    artifact = {**original, "base": base / base.sum()}
    for name in ("neighbor_ids", "neighbor_weights", "blocks"):
        assert torch.equal(artifact[name], original[name]), name
    torch.save(artifact, output / "structure.pt")
    np.save(output / "prefix_target_counts.npy", counts.numpy())
    shutil.copyfile(reference / "native_memberships.json", output / "native_memberships.json")
    report = json.loads((reference / "artifacts.json").read_text(encoding="utf-8"))
    report.update(base_count_definition="next-item labels from all nonempty prefixes of retained training sequences",
                  base_first_item="excluded", base_training_examples="unchanged; no prefix augmentation of training",
                  reference_artifacts=str(reference / "structure.pt"),
                  total_prefix_labels=int(counts.sum()),
                  base_l1_difference=float((artifact["base"] - original["base"]).abs().sum()),
                  unchanged_artifact_tensors=["neighbor_ids", "neighbor_weights", "blocks"])
    write_json(output / "artifacts.json", report)
    return report
