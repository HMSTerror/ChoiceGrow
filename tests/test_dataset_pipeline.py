"""Dataset regressions: conversion semantics, chronology, split isolation, filtering, metadata, and overwrite rejection."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io
import zipfile
from unittest.mock import patch

import pandas as pd

from dataset.prepare import (align_metadata, build_raw_dataset, canonical_frame, convert_existing,
                             filter_core, frame_from_text, make_splits, pack_sample, read_interactions)
from dataset.sources import load_settings, records, source_file, validate_archive
from dataset.download import download_file


class DatasetPipelineTests(unittest.TestCase):
    def test_downloader_resumes_correct_range_or_restarts_if_ignored(self):
        """Simulate HTTP responses to verify resume/restart writes and run actual gzip CRC validation."""
        class Response(io.BytesIO):
            """In-memory response body with HTTP status and headers."""
            def __init__(self, data, status, headers):
                super().__init__(data)
                self.status, self.headers = status, headers
        payload = gzip.compress(b"source-data" * 100)
        for status in (200, 206):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                target = Path(temp) / "source.gz"
                target.with_name("source.gz.partial").write_bytes(payload[:10])
                data = payload[10:] if status == 206 else payload
                headers = {"Content-Length": str(len(data)), "Content-Range": f"bytes 10-{len(payload)-1}/{len(payload)}"}
                with patch("dataset.download.urlopen", return_value=Response(data, status, headers)) as open_url, redirect_stdout(io.StringIO()):
                    download_file(target, "https://example.com/source.gz")
                self.assertEqual(open_url.call_args.args[0].get_header("Range"), "bytes=10-")
                self.assertEqual(target.read_bytes(), payload)

    def test_downloader_rejects_misaligned_partial_response(self):
        """Reject a server's incorrect resume offset without appending to the existing file."""
        response = io.BytesIO(b"wrong")
        response.status = 206
        response.headers = {"Content-Range": "bytes 99-103/104", "Content-Length": "5"}
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "source.gz"
            partial = target.with_name("source.gz.partial")
            partial.write_bytes(b"original")
            with patch("dataset.download.urlopen", return_value=response), redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                download_file(target, "https://example.com/source.gz")
            self.assertFalse(target.exists())
            self.assertEqual(partial.read_bytes(), b"original")

    def test_example_and_long_history_keep_correct_target(self):
        expected = {"seq": [20, 20, 3, 7, 11], "len_seq": 3, "next": 18}
        self.assertEqual(pack_sample([3, 7, 11], 18, 20, 5), expected)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.txt"
            path.write_text("4 8 12 19\n1 2 3 4 5 6 7 8\n", encoding="utf-8")
            frame = frame_from_text(path, 20, 5)
            self.assertEqual(frame.iloc[0].to_dict(), expected)
            self.assertEqual(frame.iloc[1].to_dict(), {"seq": [2, 3, 4, 5, 6], "len_seq": 5, "next": 7})

    def test_existing_conversion_keeps_split_and_rejects_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "output"
            directory = source / "ATG"
            directory.mkdir(parents=True)
            pd.DataFrame([{"item_num": 20, "seq_size": 5}]).to_pickle(directory / "data_statis.df")
            for name in ("train_data.txt", "valid_data.txt", "test_data.txt"):
                (directory / name).write_text("4 8 12 19\n", encoding="utf-8")
            report = convert_existing(source, output, load_settings(), ["Toys"])
            self.assertEqual(report["Toys"]["splits"]["test"]["samples"], 1)
            pd.testing.assert_frame_equal(pd.read_pickle(output / "Toys/test_data.df"),
                                          frame_from_text(directory / "test_data.txt", 20, 5))
            (directory / "test_data.txt").write_text("4 8 12 18\n", encoding="utf-8")
            with self.assertRaises(AssertionError):
                convert_existing(source, output, load_settings(), ["Toys"])
            self.assertEqual(pd.read_pickle(output / "Toys/test_data.df")["next"].iloc[0], 18)

    def test_iterative_core_requires_multiple_rounds(self):
        frame = pd.DataFrame([("a", "x", 1, 0), ("a", "y", 2, 1), ("b", "x", 1, 2),
                              ("b", "y", 2, 3), ("c", "y", 1, 4), ("c", "z", 2, 5)],
                             columns=["user", "item", "timestamp", "position"])
        config = deepcopy(load_settings()["processing"])
        config.update(min_user_interactions=2, min_item_interactions=2)
        filtered, rounds = filter_core(frame, config)
        self.assertEqual(len(filtered), 4)
        self.assertEqual(set(filtered["user"]), {"a", "b"})
        self.assertGreaterEqual(len(rounds), 3)

    def test_raw_pipeline_deterministic_split_and_metadata_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_settings()
            config.update(raw_root=str(Path(temp) / "raw"), output_root=str(Path(temp) / "generated"))
            settings = config["datasets"]["Beauty"]
            source = source_file(config["raw_root"], "Beauty", settings["interactions"])
            source.parent.mkdir(parents=True)
            # Source timestamps are deliberately reversed; processing must select the final target chronologically.
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                for user in range(10):
                    for timestamp in reversed(range(6)):
                        row = {"reviewerID": f"u{user}", "asin": f"i{timestamp % 5}", "unixReviewTime": timestamp}
                        handle.write(json.dumps(row) + "\n")
            meta = source_file(config["raw_root"], "Beauty", settings["metadata"])
            with gzip.open(meta, "wt", encoding="utf-8") as handle:
                handle.write(repr({"asin": "i0", "title": "example", "categories": [["Toys"]]}) + "\n")
                handle.write(repr({"asin": "unrelated", "title": "excluded"}) + "\n")
            with redirect_stdout(io.StringIO()):
                report = build_raw_dataset(config, "Beauty")
            self.assertEqual(report["samples"], {"train": 8, "val": 1, "test": 1})
            self.assertEqual(report["metadata"]["aligned_items"], 1)
            self.assertEqual(report["metadata"]["missing_items"], 4)
            self.assertFalse(report["matches_legacy_split"])
            output = Path(config["output_root"]) / "Beauty"
            users = json.loads((output / "split_users.json").read_text())
            self.assertEqual(sum(map(len, users.values())), len(set(sum(users.values(), []))))
            for name in ("train", "val", "test"):
                frame = pd.read_pickle(output / f"{name}_data.df")
                self.assertTrue((frame["next"] == 0).all())
                canonical_frame(frame, 5, 10)
            with self.assertRaises(FileExistsError):
                build_raw_dataset(config, "Beauty")
            again, mapping, split_users = make_splits(read_interactions(source, "amazon2014"), config["processing"])
            self.assertEqual(users, split_users)
            pd.testing.assert_frame_equal(again["train"], pd.read_pickle(output / "train_data.df"))

    def test_steam_and_movielens_source_adapters(self):
        with tempfile.TemporaryDirectory() as temp:
            steam = Path(temp) / "steam.gz"
            with gzip.open(steam, "wt", encoding="utf-8") as handle:
                handle.write(repr({"username": "user", "product_id": "17", "date": "2020-01-01"}) + "\n")
            frame = read_interactions(steam, "steam_reviews")
            self.assertEqual((frame.iloc[0]["user"], frame.iloc[0]["item"]), ("user", "17"))
            self.assertEqual(frame.iloc[0]["timestamp"], 1577836800)
            movies = Path(temp) / "ml.zip"
            with zipfile.ZipFile(movies, "w") as archive:
                archive.writestr("ml-1m/ratings.dat", "1::7::5::100\n1::9::3::101\n")
                archive.writestr("ml-1m/movies.dat", "7::Example (2000)::Drama|Comedy\n")
            frame = read_interactions(movies, "movielens1m")
            self.assertEqual(frame["item"].tolist(), ["7", "9"])
            info = align_metadata(movies, "movielens1m", {"7": 0, "9": 1}, Path(temp) / "meta.jsonl")
            self.assertEqual(info["missing_items"], 1)

    def test_literal_parser_never_executes_code_and_corrupt_archive_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.txt"
            path.write_text("{'field': __import__('os')}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(records(path))
            bad = Path(temp) / "bad.gz"
            bad.write_bytes(b"not gzip")
            with self.assertRaises(OSError):
                validate_archive(bad)


if __name__ == "__main__":
    unittest.main()
