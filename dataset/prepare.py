"""Convert existing TXT to DF, or build explicitly defined DF splits from official interaction sources.

Run python -m dataset.prepare --help for subcommands. Raw processing preserves original item mappings,
user split membership, processing settings, and source summaries without overwriting existing experiments.
"""
# Note: docs/decisions/implemented/feature/2026-11-03-source-datasets.md
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd

from .download import download_datasets, download_file
from .sources import (DEFAULT_CONFIG, DF_NAMES, MAIN_DATASETS, TXT_NAMES,
                      file_record, load_settings, records, source_file, write_json)


def pack_sample(history, target, item_count, history_length):
    """Keep the latest H history items, left-pad with N, and return seq, len_seq, and next fields.

    History and target IDs are already zero-based; padding N is distinct from catalog IDs 0..N-1.
    Each history produces one sample, avoiding the inconsistent long-sequence branch of the legacy notebook.
    """
    history = [int(item) for item in history][-history_length:]
    target = int(target)
    if not history or any(item < 0 or item >= item_count for item in (*history, target)):
        raise ValueError("Sample history must be nonempty and all item IDs must be within the catalog")
    return {"seq": [item_count] * (history_length - len(history)) + history,
            "len_seq": len(history), "next": target}


def frame_from_text(path, item_count, history_length):
    """Convert legacy one-based TXT sequences to DF compatible with the current training loader.

    Example: 4 8 12 19 becomes history [3,7,11] and target 18, then history is padded to the requested length.
    Reject rows with fewer than two items instead of silently discarding samples; preserve existing splits.
    """
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            values = [int(item) - 1 for item in line.split()]
            if len(values) < 2:
                raise ValueError(f"{path}:{line_number} requires at least a history item and a target")
            rows.append(pack_sample(values[:-1], values[-1], item_count, history_length))
    if not rows:
        raise ValueError(f"Empty sample file: {path}")
    return pd.DataFrame(rows, columns=["seq", "len_seq", "next"])


def canonical_frame(frame, item_count, history_length):
    """Validate existing DF and normalize valid histories to check conversion equivalence."""
    rows = []
    for seq, length, target in frame[["seq", "len_seq", "next"]].itertuples(index=False, name=None):
        values = [int(item) for item in seq]
        if any(item < 0 or item > item_count for item in values):
            raise ValueError("DF item ID is outside the catalog")
        history = [item for item in values if item != item_count]
        if len(history) != int(length) or not 1 <= len(history) <= history_length:
            raise ValueError("DF len_seq does not match the valid history")
        rows.append(pack_sample(history, target, item_count, history_length))
    if not rows:
        raise ValueError("DF split is empty")
    return pd.DataFrame(rows, columns=["seq", "len_seq", "next"])


def convert_existing(source_root, output_root, config, datasets):
    """Apply the valid legacy conversion rules; compare existing DF rows and reject differences.

    If TXT is absent, read and validate existing DF, also supporting Steam and ML1M.
    Preserve samples and split membership; convert storage only, without inventing an original ASIN mapping.
    """
    report = {}
    for dataset in datasets:
        settings = config["datasets"][dataset]
        directory = settings["directory"]
        # Legacy source directories may still be named ATG/ASO; new output uses Toys/Sports consistently.
        source = Path(source_root) / settings.get("legacy_directory", directory)
        output = Path(output_root) / directory
        statistics = pd.read_pickle(source / "data_statis.df")
        n, h = int(statistics["item_num"].iloc[0]), int(statistics["seq_size"].iloc[0])
        frames, rows = {}, {}
        # Validate all three splits before writing to avoid partially overwriting the destination.
        for split, name in DF_NAMES.items():
            text = source / TXT_NAMES[split]
            frame = (frame_from_text(text, n, h) if text.exists()
                     else canonical_frame(pd.read_pickle(source / name), n, h))
            if (source / name).exists():
                pd.testing.assert_frame_equal(frame, canonical_frame(pd.read_pickle(source / name), n, h))
            if (output / name).exists():
                pd.testing.assert_frame_equal(frame, canonical_frame(pd.read_pickle(output / name), n, h))
            frames[split] = frame
            rows[split] = {"samples": len(frame), "input": str(text if text.exists() else source / name)}
        if (output / "data_statis.df").exists():
            pd.testing.assert_frame_equal(pd.read_pickle(output / "data_statis.df"), statistics)
        output.mkdir(parents=True, exist_ok=True)
        for split, frame in frames.items():
            # Keep byte-identical existing files rather than reserializing equivalent frames.
            if not (output / DF_NAMES[split]).exists():
                frame.to_pickle(output / DF_NAMES[split])
        if not (output / "data_statis.df").exists():
            statistics.to_pickle(output / "data_statis.df")
        report[dataset] = {"item_count": n, "history_length": h, "splits": rows, "equivalent": True}
    write_json(Path(output_root) / "conversion_report.json", report)
    return report


def read_interactions(path, source_format):
    """Extract original user IDs, item IDs, timestamps, and line positions, excluding review text as a feature.

    Use unixReviewTime for Amazon, posting dates for Steam, and rating timestamps for MovieLens.
    Break equal-time ties by original file position; reject missing/invalid times rather than inventing order.
    """
    rows = []
    if source_format == "movielens1m":
        with zipfile.ZipFile(path) as archive:
            with archive.open("ml-1m/ratings.dat") as handle:
                for position, line in enumerate(handle):
                    user, item, _, timestamp = line.decode("latin-1").strip().split("::")
                    rows.append((user, item, int(timestamp), position))
    else:
        for position, row in enumerate(records(path)):
            if source_format == "amazon2014":
                user, item, timestamp = row["reviewerID"], row["asin"], int(row["unixReviewTime"])
            elif source_format == "steam_reviews":
                user, item = row["username"], row["product_id"]
                timestamp = int(datetime.fromisoformat(row["date"]).replace(tzinfo=timezone.utc).timestamp())
            else:
                raise ValueError(f"Unknown interaction format: {source_format}")
            rows.append((str(user), str(item), timestamp, position))
    if not rows:
        raise ValueError(f"Raw interaction file is empty: {path}")
    frame = pd.DataFrame(rows, columns=["user", "item", "timestamp", "position"])
    if not np.isfinite(frame["timestamp"].to_numpy()).all() or (frame["timestamp"] < 0).any():
        raise ValueError("Timestamps must be finite and nonnegative")
    return frame


def filter_core(frame, settings):
    """Iteratively filter user/item interaction counts until both thresholds hold; retain duplicates by default.

    Count interaction records. If deduplication is enabled, sort by time and keep the last user/item occurrence.
    Return filtered events and per-round sizes to explain changes in dataset size.
    """
    frame = frame.sort_values(["user", "timestamp", "position"], kind="stable")
    if settings["deduplicate_user_item"]:
        frame = frame.drop_duplicates(["user", "item"], keep="last")
    minimum_user = int(settings["min_user_interactions"])
    minimum_item = int(settings["min_item_interactions"])
    if minimum_user < 2 or minimum_item < 1:
        raise ValueError("Users require at least 2 interactions and the item threshold must be at least 1")
    rounds = []
    while True:
        before = len(frame)
        users = frame["user"].value_counts()
        frame = frame[frame["user"].isin(users[users >= minimum_user].index)]
        items = frame["item"].value_counts()
        frame = frame[frame["item"].isin(items[items >= minimum_item].index)]
        rounds.append({"interactions": len(frame), "users": int(frame["user"].nunique()),
                       "items": int(frame["item"].nunique())})
        if frame.empty:
            raise ValueError("No interactions remain after filtering; check source data and threshold settings")
        if len(frame) == before:
            return frame, rounds


def make_splits(frame, settings):
    """Use each user's final interaction as target, up to H preceding items as history, then split users 80%/10%/remainder.

    This matches the one-sequence-per-user shape of legacy TXT, but its user order, ID mapping, and seed are unknown.
    This is an explicitly defined new protocol, not a row-for-row legacy reproduction; all training seeds share the splits.
    """
    if settings["sample_protocol"] != "one_last_target_per_user":
        raise ValueError("Only the one_last_target_per_user protocol is implemented")
    h = int(settings["history_length"])
    train, val = float(settings["train_fraction"]), float(settings["validation_fraction"])
    if h < 1 or not all(math.isfinite(x) and x > 0 for x in (train, val)) or train + val >= 1:
        raise ValueError("History length and train/validation shares must be positive, with shares summing to less than 1")
    item_map = {item: index for index, item in enumerate(sorted(frame["item"].unique()))}
    users = np.asarray(sorted(frame["user"].unique()))
    users = np.random.default_rng(int(settings["split_seed"])).permutation(users).tolist()
    n_train, n_val = int(len(users) * train), int(len(users) * val)
    split_users = {"train": users[:n_train], "val": users[n_train:n_train + n_val],
                   "test": users[n_train + n_val:]}
    if any(not values for values in split_users.values()):
        raise ValueError("Too few users to create nonempty training, validation, and test splits")
    samples = {}
    for user, group in frame.groupby("user", sort=False):
        ordered = group.sort_values(["timestamp", "position"], kind="stable")
        sequence = [item_map[item] for item in ordered["item"]]
        samples[user] = pack_sample(sequence[:-1], sequence[-1], len(item_map), h)
    splits = {name: pd.DataFrame([samples[user] for user in values], columns=["seq", "len_seq", "next"])
              for name, values in split_users.items()}
    return splits, item_map, split_users


def align_metadata(path, source_format, item_map, output):
    """Align attributes by original item ID and save item_metadata.jsonl without an external embedding service.

    Keep available titles, categories, and descriptions; missing attributes do not change items, interactions, or splits.
    Read MovieLens attributes from movies.dat in the same ZIP, without an additional download.
    """
    found = set()
    with Path(output).open("w", encoding="utf-8") as handle:
        if source_format == "movielens1m":
            with zipfile.ZipFile(path) as archive:
                with archive.open("ml-1m/movies.dat") as movies:
                    for line in movies:
                        item, title, genres = line.decode("latin-1").strip().split("::")
                        if item in item_map:
                            record = {"item_id": item_map[item], "original_item_id": item,
                                      "title": title, "genres": genres.split("|")}
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            found.add(item)
        else:
            key = "asin" if source_format == "amazon2014" else "id"
            fields = ("title", "app_name", "categories", "genres", "tags", "description", "brand", "price")
            for row in records(path):
                item = str(row.get(key, ""))
                if item not in item_map or item in found:
                    continue
                record = {"item_id": item_map[item], "original_item_id": item,
                          **{field: row[field] for field in fields if field in row}}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                found.add(item)
    return {"aligned_items": len(found), "missing_items": len(item_map) - len(found),
            "used_by_main_model": False}


def build_raw_dataset(config, dataset):
    """Build an independent DF directory from one official source, preserving mappings, protocol, and provenance."""
    source = config["datasets"][dataset]
    output = Path(config["output_root"]) / source["directory"]
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new output_root in sources.json: {output}")
    path = source_file(config["raw_root"], dataset, source["interactions"])
    frame = read_interactions(path, source["format"])
    input_rows = len(frame)
    filtered, rounds = filter_core(frame, config["processing"])
    splits, item_map, split_users = make_splits(filtered, config["processing"])
    n, h = len(item_map), int(config["processing"]["history_length"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build the complete dataset in a sibling temporary directory, then rename it to avoid incomplete final outputs.
    with tempfile.TemporaryDirectory(prefix=".dataset_build_", dir=output.parent) as temporary:
        directory = Path(temporary) / output.name
        directory.mkdir()
        for name, values in splits.items():
            canonical_frame(values, n, h)
            values.to_pickle(directory / DF_NAMES[name])
        pd.DataFrame([{"seq_size": h, "item_num": n}]).to_pickle(directory / "data_statis.df")
        write_json(directory / "item_mapping.json", item_map)
        write_json(directory / "split_users.json", split_users)
        sources = {"interactions": file_record(path, source["interactions"])}
        metadata_path = (path if source["format"] == "movielens1m" else
                         source_file(config["raw_root"], dataset, source["metadata"]))
        if metadata_path.exists():
            metadata = align_metadata(metadata_path, source["format"], item_map, directory / "item_metadata.jsonl")
            sources["metadata"] = file_record(metadata_path, source.get("metadata", source["interactions"]))
        else:
            metadata = {"status": "not_downloaded", "used_by_main_model": False}
        report = {"dataset": dataset, "origin": "official_raw_new_protocol",
                  "matches_legacy_split": False, "settings": config["processing"], "sources": sources,
                  "input_interactions": input_rows, "retained_interactions": len(filtered),
                  "retained_users": sum(len(users) for users in split_users.values()), "item_count": n,
                  "filter_rounds": rounds, "samples": {name: len(values) for name, values in splits.items()},
                  "ordering": "user timestamp then original file position", "item_ids": "lexicographic original ID",
                  "metadata": metadata}
        write_json(directory / "provenance.json", report)
        directory.rename(output)
    print(json.dumps({"dataset_complete": str(output), **report}, ensure_ascii=False), flush=True)
    return report


def attach_metadata(config, dataset, download=False):
    """Add item attributes to an existing generated dataset without rebuilding interactions, mappings, or splits.

    Raw-generated directories contain traceable item_mapping.json; do not infer ASINs from legacy integer IDs.
    Write attributes through a temporary file and record their actual source and coverage in provenance.
    """
    source = config["datasets"][dataset]
    output = Path(config["output_root"]) / source["directory"]
    item_map = json.loads((output / "item_mapping.json").read_text(encoding="utf-8"))
    report = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    url = source["interactions"] if source["format"] == "movielens1m" else source["metadata"]
    path = source_file(config["raw_root"], dataset, url)
    if download:
        download_file(path, url)
    if not path.exists():
        raise FileNotFoundError(f"Missing item metadata source; add --download to fetch it: {path}")
    with tempfile.TemporaryDirectory(prefix=".metadata_", dir=output) as temporary:
        target = Path(temporary) / "item_metadata.jsonl"
        report["metadata"] = align_metadata(path, source["format"], item_map, target)
        target.replace(output / "item_metadata.jsonl")
    report["sources"]["metadata"] = file_record(path, url)
    write_json(output / "provenance.json", report)
    print(json.dumps({"metadata_complete": dataset, **report["metadata"]}, ensure_ascii=False), flush=True)
    return report["metadata"]


def main():
    """Convert legacy splits, build explicitly defined raw splits, or attach metadata to generated data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Convert existing TXT to DF while preserving splits and checking existing DF")
    convert.add_argument("--source-root", type=Path, required=True)
    convert.add_argument("--output-root", type=Path, required=True)
    convert.add_argument("--datasets", nargs="+", default=list(MAIN_DATASETS))
    raw = commands.add_parser("raw", help="Build DF from official sources using the explicit protocol without overwriting data")
    raw.add_argument("--datasets", nargs="+", default=list(MAIN_DATASETS))
    raw.add_argument("--download", action="store_true")
    raw.add_argument("--include-metadata", action="store_true")
    raw.add_argument("--workers", type=int, default=3)
    metadata = commands.add_parser("metadata", help="Attach item attributes to generated data while preserving interaction splits")
    metadata.add_argument("--datasets", nargs="+", default=list(MAIN_DATASETS))
    metadata.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = load_settings(args.config)
    if any(name not in config["datasets"] for name in args.datasets):
        parser.error("Unknown dataset; use Beauty/Toys/Sports/Steam/ML1M or a configured name")
    if args.command == "convert":
        report = convert_existing(args.source_root, args.output_root, config, args.datasets)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "metadata":
        for dataset in args.datasets:
            attach_metadata(config, dataset, args.download)
    else:
        if args.include_metadata and not args.download:
            parser.error("--include-metadata requires --download; existing local metadata files are aligned automatically")
        if args.download:
            download_datasets(config, args.datasets, args.include_metadata, args.workers)
        for dataset in args.datasets:
            build_raw_dataset(config, dataset)


if __name__ == "__main__":
    main()
