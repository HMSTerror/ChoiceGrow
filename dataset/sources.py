"""Shared dataset settings, paths, and safe text parsing, independent of models and PyTorch."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlparse
import os
import zipfile


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "sources.json"
MAIN_DATASETS = ("Beauty", "Toys", "Sports")
DF_NAMES = {"train": "train_data.df", "val": "val_data.df", "test": "test_data.df"}
TXT_NAMES = {"train": "train_data.txt", "val": "valid_data.txt", "test": "test_data.txt"}


def load_settings(path=DEFAULT_CONFIG):
    """Read dataset settings; resolve raw/output paths relative to the configuration file."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for name in ("raw_root", "output_root"):
        value = Path(config[name])
        config[name] = str(value if value.is_absolute() else path.parent / value)
    return config


def source_file(raw_root, dataset, url):
    """Preserve official filenames and keep downloads in separate dataset directories."""
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"URL has no filename: {url}")
    return Path(raw_root) / dataset / name


def write_json(path, value):
    """Write JSON through a same-directory temporary file for settings, mappings, and processing manifests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def file_record(path, url=None):
    """Record actual size and SHA256 to associate generated data with the exact downloaded source."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": str(path.resolve()), "url": url, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def validate_archive(path):
    """Fully read gzip or validate ZIP member CRCs, rejecting truncated source downloads."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"ZIP member validation failed: {bad}")
    else:
        raise ValueError(f"Unsupported source file format: {path}")


def records(path):
    """Stream JSON records, using literal_eval for legacy Amazon/Steam dictionary literals, never eval.

    Do not load all review text or metadata into memory. Require dictionaries and report line numbers on errors.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = ast.literal_eval(line)
                if not isinstance(value, dict):
                    raise ValueError("Each record must be a dictionary")
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"Cannot parse {path}:{line_number}: {error}") from error
            yield value
