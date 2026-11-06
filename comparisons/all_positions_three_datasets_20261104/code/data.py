"""Read and prepare the supplied train/validation/test splits for model input.

Training reads pandas-pickle .df files only, with item IDs 0..N-1 and padding ID N.
TXT conversion and official-source processing live in dataset/prepare.py, outside the training loader.
Existing split membership, targets, and valid histories are preserved.
"""
# Note: docs/decisions/implemented/feature/2026-11-03-source-datasets.md
from pathlib import Path
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DATASETS = {"Toys": "Toys", "Sports": "Sports", "Beauty": "Beauty",
            "Steam": "Steam", "ML1M": "ML1M"}
ALIASES = {name.lower(): name for name in DATASETS}
ALIASES.update(atg="Toys", aso="Sports")
SPLITS = ("train", "val", "test")


def write_json(path, value):
    """Atomically replace JSON after a temporary write; retry brief Windows file locks a bounded number of times."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Training and summary processes may access the same file; readers must never see partial JSON.
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                # Windows readers or scanners may briefly lock the destination; keep the previous complete file meanwhile.
                if getattr(error, "winerror", None) not in (5, 32) or attempt == 7:
                    raise
                time.sleep(min(.05 * 2 ** attempt, .5))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_source(directory, split, item_count, length):
    """Read a .df split, validate IDs and lengths, and normalize histories to left-padded inputs.

    Return seq:[samples,H], length:[samples], target:[samples], and the source filename.
    Check ID ranges, history lengths, and empty splits so padding cannot be treated as a catalog item.
    """
    source = {"train": "train_data.df", "val": "val_data.df", "test": "test_data.df"}[split]
    frame = pd.read_pickle(Path(directory) / source)
    histories, targets = [], []
    for sequence, declared_length, target in frame[["seq", "len_seq", "next"]].itertuples(index=False, name=None):
        values = [int(value) for value in sequence]
        if any(value < 0 or value > item_count for value in values):
            raise ValueError(f"Item ID outside catalog in {source}")
        valid = [value for value in values if value != item_count]
        if len(valid) != int(declared_length) or not 1 <= len(valid) <= length:
            raise ValueError(f"History length mismatch in {source}")
        histories.append(valid)
        targets.append(int(target))
    if not histories or any(not 0 <= value < item_count for value in targets):
        raise ValueError(f"Empty split or invalid target in {source}")
    sequence = np.full((len(histories), length), item_count, dtype=np.int64)
    lengths = np.asarray([len(row) for row in histories], dtype=np.int64)
    for index, row in enumerate(histories):
        sequence[index, -len(row):] = row
    return {"seq": sequence, "length": lengths,
            "target": np.asarray(targets, dtype=np.int64)}, source


def prepare(source_root, output, dataset, config):
    """Save the three existing DF splits under prepared/, with metadata and training-target counts.

    train_target_counts includes training targets only, for the shared background b proportional to sqrt(count+1).
    Smoke mode takes a small subset for a pipeline check; ordinary experiments read the complete supplied splits.
    """
    directory = Path(source_root) / DATASETS[dataset]
    statistics = pd.read_pickle(directory / "data_statis.df")
    item_count = int(statistics["item_num"].iloc[0])
    length = int(config["history_length"])
    if int(statistics["seq_size"].iloc[0]) != length:
        raise ValueError("Configured history length disagrees with supplied dataset")
    prepared = Path(output) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    report = {"dataset": dataset, "item_count": item_count, "padding_id": item_count,
              "history_length": length, "smoke": config["smoke"], "splits": {}}
    counts = None
    for split in SPLITS:
        values, source = read_source(directory, split, item_count, length)
        original_count = len(values["target"])
        if config["smoke"]:
            limit = 32 if split == "train" else 8
            values = {key: value[:limit] for key, value in values.items()}
        if split == "train":
            counts = np.bincount(values["target"], minlength=item_count)
        np.savez(prepared / f"{split}.npz", **values)
        report["splits"][split] = {"source": source, "original_rows": original_count,
                                    "used_rows": len(values["target"])}
    np.save(prepared / "train_target_counts.npy", counts)
    write_json(prepared / "metadata.json", report)
    return report


def prepared_root(output):
    """Resolve data_source.json references to shared prepared data, rejecting reference cycles."""
    current, seen = Path(output), set()
    while (current / "data_source.json").exists():
        current = current.resolve()
        if current in seen:
            raise ValueError(f"Circular prepared-data reference at {current}")
        seen.add(current)
        current = Path(json.loads((current / "data_source.json").read_text(encoding="utf-8"))["path"])
    return current


def load_split(output, split):
    """Copy .npz columns into CPU tensors so no tensor depends on a closed archive."""
    with np.load(prepared_root(output) / "prepared" / f"{split}.npz") as values:
        return {key: torch.from_numpy(values[key].copy()) for key in values.files}


def metadata(output):
    """Read source metadata, including catalog size, history length, and split sample counts."""
    return json.loads((prepared_root(output) / "prepared" / "metadata.json").read_text(encoding="utf-8"))


class BatchedColumns(Dataset):
    """Store columns with vectorized batch indexing while preserving DataLoader sampling semantics."""
    def __init__(self, columns):
        """Store history, history length, and target columns with equal sample counts."""
        self.columns = columns

    def __len__(self):
        """Use the first column to determine the number of samples."""
        return len(self.columns[0])

    def __getitem__(self, index):
        """Read every column for one sample."""
        return tuple(column[index] for column in self.columns)

    def __getitems__(self, indices):
        """Read a whole batch as three tensors, reducing per-sample Python overhead."""
        index = torch.as_tensor(indices, dtype=torch.long)
        return tuple(column.index_select(0, index) for column in self.columns)


def batched_identity(values):
    """Return the already assembled batch without stacking its tensors again."""
    return values


def loader(values, batch_size, shuffle=False, seed=100):
    """Create a loader with an independent shuffle generator; validation and test retain source order."""
    columns = [values[key] for key in ("seq", "length", "target")]
    return DataLoader(BatchedColumns(columns), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=batched_identity,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)
