# ChoiceGrow

The original configurable entry point combines **history-neighbor weighting, native community/catalog nesting, and learned diffusion rates**. The current frozen research protocol is provided by the [comparison suites](comparisons/README.md): full-sequence background `bn`, history-only neighbors `nn_clean`, and fixed Root or stratified community/catalog shares. Source comments, docstrings, and CLI help are in English. Dataset sources and preprocessing are described in [dataset/README.md](dataset/README.md).

The frozen comparisons use `gamma=0.02`, `patience=300`, validation-selected EMA weights, and the original `eta=0` guided-score ranking. Their runners prepare the reviewed structural artifacts explicitly. The repository-root `config.json` retains the original learned-rate defaults; it does not select the frozen comparison protocol. The no-neighbor factor control explicitly uses `gamma=0`.

## Environment

The Linux setup script supports Python 3.10-3.12 and creates a project-local virtual environment:

```bash
bash setup.sh          # PyTorch with CUDA 12.4
# bash setup.sh --cpu  # CPU-only PyTorch
```

On Windows, use a compatible Python environment with the dependencies in `requirements.txt`. Install the appropriate PyTorch build for your hardware before the remaining requirements if necessary. The current local Python 3.13 environment has also passed the regression suite.

`run.cmd` selects `CHOICE_PYTHON` when set, then the project virtual environment, then the existing local Anaconda interpreter if present, and finally `python` from `PATH`. `run.sh` selects `CHOICE_PYTHON`, the project virtual environment, or system `python3`, in that order.

## Prepare datasets

From the repository root, using the Python environment prepared above (on Linux, use `.venv/bin/python` instead of `python`):

```bash
python -m dataset.prepare raw --datasets Toys Beauty Sports --download
```

The scripts download official sources and create traceable DF splits under `dataset/processed/`. Raw downloads, generated datasets, model weights, and run outputs are excluded from Git. Existing processed datasets are not overwritten. See the [dataset guide](dataset/README.md) for the exact protocol and options.

## Run

Windows PowerShell:

```powershell
./run.cmd --config config.json
```

Linux:

```bash
bash run.sh --config config.json
```

The default experiment uses Toys, training seed 100, and learning rate `0.0006`. To change the dataset or seed:

```powershell
./run.cmd --dataset Beauty --seed 101 --output runs/beauty_s101
./run.cmd --dataset Sports --seed 102 --output runs/sports_s102
```

Linux accepts the same arguments with `bash run.sh` in place of `./run.cmd`.

## Configuration and overrides

Edit `config.json`. Omitted CLI options retain file values; explicitly supplied options override them. Rate initialization is configured under `diffusion`:

```json
"initial_shares": [0.2165, 0.7835],
"total_budget": 2.302585092994046,
"dynamic_rate_gate": true
```

The share order is **community, catalog**. Both values must be strictly positive and sum to one, otherwise validation fails. With learned rates, these shares initialize the gate; training adjusts the allocation while keeping the total budget fixed. There is no separate fixed-rate training warm-up stage.

```powershell
# Override initialization and keep learning the allocation.
./run.cmd --initial-shares 0.3 0.7 --learnable-rates --output runs/learned_30_70

# Disable learning and keep both shares fixed throughout the run.
./run.cmd --no-learnable-rates --output runs/fixed
```

## Stages and checks

To run stages separately, keep the configuration, dataset, seed, and output directory the same:

```powershell
./run.cmd --stage prepare --output runs/my_run
./run.cmd --stage artifacts --output runs/my_run
./run.cmd --stage train --output runs/my_run
./run.cmd --stage evaluate --output runs/my_run
```

Run a one-epoch check on a small subset, or execute the regression suite:

```powershell
./run.cmd --smoke --output runs/my_smoke
python -m unittest discover -s tests -v
```

Use a new output directory for a new training run. An existing best checkpoint is protected from accidental retraining.

New training runs save a complete `training_state.pt` after every finished epoch. To continue a run, keep its output directory and use:

```powershell
./run.cmd --resume --output runs/my_run --epochs 2000 --patience 300
```

`--epochs 2000` means total epoch2000, rather than2000 additional epochs; `--epochs 0` keeps validation early stopping without a hard epoch limit. Resume reads the saved run's configuration by default and skips data/artifact preparation. Only the epoch limit and patience may change; model, rate policy, dataset, seed, batches, and other settings must match. Use the same device and numerical environment. The optimizer, live and EMA weights, gate, random generators, shuffle sequence, and validation-best/early-stop history are restored before the next epoch. Outputs left beyond the saved epoch are reconciled with the snapshot; old test reports and prediction caches move to `resume_history/` and are recomputed after training. Add `--stage train` to continue training without automatically testing afterward.

Historical EMA-only checkpoints cannot supply missing optimizer or live-model state. Existing background processes continue using their already-loaded code; this feature applies to new main-method training and explicitly updated comparison jobs.

## Code map

| File | Responsibility |
| --- | --- |
| `run.py`, `configuration.py` | CLI entry point, defaults, explicit overrides, and validation |
| `dataset/` | Official downloads, raw preprocessing, metadata alignment, and legacy conversion |
| `data.py`, `artifacts.py` | Model inputs and training-only structural artifacts |
| `neighbors.py` | Neighbor tables and user-specific destination distributions |
| `scope.py` | Native Infomap communities and nested weighted projections |
| `transition.py` | User-conditioned generator `Q_u` and fixed/learned rate policies |
| `kernel.py` | Exact forward probabilities using the closed nested-partition algebra |
| `models.py` | ChoiceGrow denoiser and bounded log-score ratios |
| `diffusion.py` | Training objective and adaptive reverse sampling |
| `train.py`, `evaluate.py`, `checkpoint.py`, `training_state.py` | Optimization, evaluation weights, and complete training snapshots/resume |
| `tests/` | Configuration, dataset, and mathematical regression tests |

Tests use the current implementation and independent mathematical checks; no pre-refactor source snapshot is bundled. Local design notes under `docs/` are excluded from Git.

## Outputs

Each experiment writes to its directory under `runs/`:

| File | Contents |
| --- | --- |
| `choice_log.jsonl` | Per-epoch training and validation records |
| `training.json` | Best validation metrics and training summary |
| `report.json` | Final test metrics from the validation-selected checkpoint |
| `recommendations.npz` | Recommendation cache for later analyses, including long-tail evaluation |
| `resolved_config.json` | Effective configuration and derived `level_rates` |
| `choice_best.pt`, `choice_newst.pt` | Best and latest saved weights |
| `training_state.pt` | Latest complete epoch: live/EMA weights, AdamW, learned gate/process, RNGs, shuffle and selection history |
| `resume_history/` | Previous reports and any log tail superseded by an explicit resume |

`choice_best.pt` and `choice_newst.pt` remain lightweight EMA evaluation checkpoints. Resume uses `training_state.pt`, written through a temporary file so an interrupted write retains the preceding complete snapshot. Sampling streams evaluate the same trained checkpoint; they are not independent training runs.
