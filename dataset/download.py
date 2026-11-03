"""Download official interaction files and optional item metadata with resumption and source manifests.

Run: python -m dataset.download --datasets Beauty Toys Sports
Add --include-metadata to download item attributes; downloading does not create model weights.
"""
# Note: docs/decisions/implemented/feature/2026-11-03-source-datasets.md
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import ssl
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from .sources import DEFAULT_CONFIG, MAIN_DATASETS, file_record, load_settings, source_file, validate_archive, write_json


def download_file(path, url):
    """Resume downloads and rename .partial only after the archive passes CRC validation.

    Validate existing files before reuse; verify TLS certificates normally without an insecure bypass.
    Use HTTP Range to resume from downloaded bytes, or rewrite the temporary file if Range is ignored.
    Mirrors are configurable in sources.json; use certifi or system CAs for normal TLS verification.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_name(path.name + ".partial")
        print(json.dumps({"download": url, "target": str(path)}, ensure_ascii=False), flush=True)
        # certifi supplies public root CAs across platforms; otherwise use Python's system trust configuration.
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "Grow-dataset-preparation/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with urlopen(Request(url, headers=headers), context=context, timeout=30) as response:
                    resumed = response.status == 206 and offset > 0
                    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned a mismatched resume range; refusing to append bytes at the wrong offset")
                    expected = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
                    received = 0
                    with partial.open("ab" if resumed else "wb") as handle:
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                            handle.flush()
                            received += len(chunk)
                    if expected is not None and received != expected:
                        raise OSError(f"Truncated download: received {received}, expected {expected} bytes")
                break
            except (OSError, URLError):
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 4))
        # .partial lacks the final archive suffix, so select the archive reader from the intended filename.
        # Validate the completed temporary archive before renaming it to the final filename.
        if path.suffix == ".gz":
            import gzip
            with gzip.open(partial, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass
        else:
            import zipfile
            with zipfile.ZipFile(partial) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"ZIP validation failed: {partial}")
        partial.rename(path)
    else:
        validate_archive(path)
    result = file_record(path, url)
    print(json.dumps({"download_complete": result}, ensure_ascii=False), flush=True)
    return result


def download_datasets(config, datasets, include_metadata=False, workers=3):
    """Download selected datasets concurrently; interactions are required and metadata is opt-in."""
    tasks = []
    for dataset in datasets:
        settings = config["datasets"][dataset]
        for role in ("interactions", "metadata") if include_metadata else ("interactions",):
            if role in settings:
                url = settings[role]
                tasks.append((dataset, role, source_file(config["raw_root"], dataset, url), url))
    if workers < 1:
        raise ValueError("workers must be greater than zero")
    results = {name: {} for name in datasets}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [(dataset, role, executor.submit(download_file, path, url)) for dataset, role, path, url in tasks]
        for dataset, role, future in futures:
            results[dataset][role] = future.result()
    for dataset, files in results.items():
        write_json(Path(config["raw_root"]) / dataset / "download_manifest.json", {"dataset": dataset, "files": files})
    return results


def main():
    """Download-only CLI; --datasets selects the three main datasets or Steam/ML1M."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=list(MAIN_DATASETS))
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    config = load_settings(args.config)
    if any(name not in config["datasets"] for name in args.datasets):
        parser.error("Use Beauty/Toys/Sports/Steam/ML1M or a dataset name defined in the configuration")
    download_datasets(config, args.datasets, args.include_metadata, args.workers)


if __name__ == "__main__":
    main()
