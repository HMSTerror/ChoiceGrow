# Dataset sources and preprocessing

The default training inputs are Toys, Beauty, and Sports, generated from official interaction sources using the scripts in this directory:

```text
dataset/
  raw/                   Official downloads and download manifests
  processed/
    Toys/
    Beauty/
    Sports/
  sources.json           Source URLs, paths, and processing settings
  download.py            Downloads, HTTP resumption, and archive validation
  prepare.py             DF generation, metadata alignment, and legacy conversion
```

Each processed directory contains `train_data.df`, `val_data.df`, `test_data.df`, `data_statis.df`, `item_mapping.json`, `split_users.json`, and `provenance.json`. Raw and processed data are not distributed in this Git repository. Once generated and verified, the same datasets can be reused without rebuilding them for each training run. The main configuration uses `data_root=dataset/processed` and learning rate `0.0006`.

## Build from official sources

From the repository root, using your configured Python environment:

```bash
python -m dataset.prepare raw --datasets Toys Beauty Sports --download
```

On Linux, use `.venv/bin/python` if you installed with `setup.sh`. Omit `--download` when source files already exist. Existing processed outputs are not overwritten. To use a different protocol, copy `sources.json`, change its `output_root`, and select it with the global `--config` option:

```bash
python -m dataset.prepare --config dataset/my_sources.json raw --datasets Toys Beauty Sports --download
```

Download without preprocessing:

```bash
python -m dataset.download --datasets Toys Beauty Sports
```

Downloads use temporary `.partial` files, HTTP Range resumption where supported, normal TLS certificate verification, and gzip/ZIP integrity checks before creating the final filename. Existing downloads are validated before reuse.

## Sample construction

1. Extract user IDs, ASINs, and interaction times from the corresponding Amazon 2014 category's 5-core reviews.
2. Sort chronologically with stable file-position tie breaking. Retain repeated interactions and apply no rating filter by default. Iteratively retain users and items with at least five interactions.
3. Sort ASINs lexicographically, assign IDs `0..N-1`, and save the mapping.
4. Use each user's final interaction as the target and up to ten preceding interactions as history, producing one sample per user.
5. Split users with fixed seed 100 into 80% training, 10% validation, and the remainder for testing. Save explicit user membership for every split.
6. Left-pad histories with ID `N` and save DF files. Record source URLs, SHA256, processing parameters, and counts in provenance.

For example, with catalog size `N=20` and history length `H=5`, the legacy text sequence `4 8 12 19`, using one-based IDs, converts to:

```python
{"seq": [20, 20, 3, 7, 11], "len_seq": 3, "next": 18}
```

DF files are pandas DataFrames stored in pickle format. They are generated before training and do not indicate that a model has already been trained. The three sample files contain histories, valid lengths, and targets; `data_statis.df` contains catalog size and history length.

The default source files and protocol produce:

| Dataset | Source interactions | Items | Train / validation / test samples |
| --- | ---: | ---: | ---: |
| Toys | 167597 | 11924 | 15529 / 1941 / 1942 |
| Beauty | 198502 | 12101 | 17890 / 2236 / 2237 |
| Sports | 296337 | 18357 | 28478 / 3559 / 3561 |

The original ID mapping and split order of legacy TXT files were not preserved. This explicit protocol therefore does not claim row-for-row equivalence to those older splits. Use new experiment output directories and do not mix metrics from models evaluated on different splits into one direct comparison.

## Optional item metadata

```bash
python -m dataset.prepare metadata --datasets Beauty --download
```

Align available titles, categories, descriptions, and related attributes by ASIN into `item_metadata.jsonl`, without calling an embedding service. The verified local Beauty export covers all 12101 items; Toys and Sports metadata can be downloaded on demand. Generated files are excluded from Git.

The current method uses interactions only. Its communities come from the training interaction graph, not item metadata. Attaching metadata preserves interactions, mappings, and split membership.

## Other datasets and legacy conversion

The tools also support Steam and ML1M, but neither is included in the default processed inputs:

```bash
python -m dataset.prepare raw --datasets ML1M --download
python -m dataset.prepare raw --datasets Steam --download --include-metadata
```

The complete Steam review archive is approximately 1.34 GB. These datasets also use the configured explicit protocol, which is not guaranteed to reproduce legacy samples.

To convert existing TXT/DF inputs while preserving their split membership, use a separate output directory rather than overwriting current processed data:

```bash
python -m dataset.prepare convert --source-root /path/to/legacy/dataset --output-root runs/legacy_conversion --datasets Toys Beauty Sports
```

Legacy `ATG` and `ASO` directory names are accepted for Toys and Sports. When TXT is absent, conversion validates and reads the corresponding existing DF. This mode preserves existing samples and does not reconstruct an unknown original ASIN mapping.
