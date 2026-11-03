# ChoiceGrow

The main method combines **history-neighbor weighting, native community/catalog nesting, and learned diffusion rates**. Source comments, docstrings, and CLI help are in English. Dataset sources and preprocessing are described in [dataset/README.md](dataset/README.md).

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
| `train.py`, `evaluate.py`, `checkpoint.py` | Optimization, evaluation, and compatible checkpoint I/O |
| `tests/` | Configuration, dataset, mathematical, and differential regression tests |

`tests/reference/` is a frozen pre-refactor snapshot used only by differential tests. Local design notes under `docs/` are excluded from Git.

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

Checkpoints do not contain optimizer state and are not complete training-resume snapshots. Sampling streams evaluate the same trained checkpoint; they are not independent training runs.
