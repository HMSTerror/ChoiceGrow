# Neighbor-window comparison and Toys factorial

This experiment uses a frozen copy of the current main method in `code/`.
The main project and earlier background comparison remain unchanged.

## Current user override (2026-11-04)

The user has held all subsequent execution until reviewing the Toys results.
The coordinator was stopped while the two existing Toys workers continue unchanged.
Beauty jobs remain held, and automatic version selection/factorial execution is disabled.
`runs/execution_control.json` blocks coordinator relaunches until the user authorizes the next step.
The historical stage plan below describes the original authorization, not the current execution policy.

The Toys runs subsequently completed. The user selected all-position neighbors plus
prefix background and authorized a new three-dataset/three-seed suite followed by
the Toys factorial. That work lives in `../all_positions_three_datasets_20261104/`.
This old coordinator stays held to prevent duplicate or obsolete jobs.

## Statistical change

For a retained training sequence A,B,C,D,E, all-position neighbors count each target
position against at most its three preceding positions. The contribution is 1/distance:

```
B: A->B (1)
C: B->C (1), A->C (1/2)
D: C->D (1), B->D (1/2), A->D (1/3)
E: D->E (1), C->E (1/2), B->E (1/3)
```

Each ordered position pair is counted once, rather than repeatedly counting earlier
edges inside overlapping prefixes. Repeated real occurrences still accumulate.
Self prior 1, top16 truncation, tie-breaking, normalization, history decay 0.2,
and neighbor mixture 0.02 are unchanged. The model's training examples are not expanded.

Background alternatives are the original final-target counts and the earlier
prefix-target counts (B,C,D,E once, excluding A and padding). Both normalize sqrt(count+1).
Only training interactions contribute to either statistic.

## Stage 1: screening

Toys and Beauty use seed100. Four variants per dataset include two reused controls:

| Neighbor counts | Background counts | Training |
| --- | --- | --- |
| Final target only | Final target only | Reuse current baseline |
| Final target only | All prefix targets | Reuse completed background comparison |
| All positions, lag3 | Final target only | New |
| All positions, lag3 | All prefix targets | New |

All retain native community/catalog two-level learned rates initialized at
0.2165/0.7835, fixed total Lambda=-log(0.1), and reset 0.035. Data splits and community
labels are copied from the current raw-data baseline, not rebuilt.

Selection was agreed before the new fits: maximize the average, across Toys and Beauty,
of best validation NDCG@10 divided by the original variant's best validation NDCG@10,
minus one. Ties favor the declared order in `VARIANTS`. Test accuracy, long-tail accuracy,
and exposure are reported, but never used for selection.

## Stage 2: selected Toys factorial

The coordinator automatically starts this stage after all screening runs finish:

* No neighbors + catalog Root
* Neighbors + catalog Root
* No neighbors + native community/catalog
* Neighbors + native community/catalog

Each arm uses seeds100/101/102. The selected background is identical across all four arms.
The selected neighbor construction applies to both neighbor arms. Root receives the
same full Lambda budget in one level and has no share gate; two-level arms learn their shares.
No-neighbor arms use neighbor_mix=0, with the shared reset and background otherwise unchanged.

Completed matching training is reused. In particular, the selected Toys screening
seed100 main method is reused. The older Choice factorial uses different splits,
learning rate, patience, and TF32, so its results are excluded (`runs/reuse_audit.json`).
Seed100 participated in version selection; the three-seed results are exploratory,
and the reused seed100 is not independent confirmation.

## Shared protocol and outputs

* Python: `E:/anaco/python.exe`; CUDA device 0; at most two workers.
* Learning rate 0.0006, patience150 measured in epochs, no hard epoch limit.
* Validate every10 epochs with one stream; select checkpoints only on validation NDCG@10.
* Test the selected EMA denoiser and its gate with three sampling streams.
* Long-tail popularity counts all retained training history and target occurrences.
  Items strictly below the top20%-frequency boundary are tail; boundary ties are head.
  The same masks and full-catalog protocol apply to every variant of a dataset.
* Long-tail HR/NDCG@5/10 condition on a tail ground-truth target. Exposure measures
  the fraction of recommended slots occupied by tail items. Coverage measures the
  fraction of the tail catalog appearing, averaged over streams.
* Average streams inside each training run, then report mean and sample standard
  deviation over the three training seeds. Three streams are not three training runs.

`runs/status.json` records live phase/jobs/PIDs and complete-log best validation values.
`runs/manifest.json` records frozen code hashes, exact commands, and reuse decisions.
`runs/selection.json` records the validation-only ranking after screening.
`runs/summary.json` and `summary.md` contain per-run results, factorial aggregates,
and paired effects of communities/neighbors. Each job has `worker.log`,
`worker.err.log`, `training.json`, `report.json`, and `long_tail.json`.

## Commands

From `E:/Grow`:

```
E:/anaco/python.exe -B comparisons/neighbor_window_ablation_20261103/test_protocol.py
E:/anaco/python.exe -B -u comparisons/neighbor_window_ablation_20261103/run_suite.py --prepare
E:/anaco/python.exe -B -u comparisons/neighbor_window_ablation_20261103/run_suite.py
```

Run the coordinator once. Never start duplicate workers. It does not retry failed
training: checkpoints omit optimizer state, so an interrupted fit cannot be called
an exact resume. A completed fit whose evaluation failed can finish evaluation using
`--worker-job <job.json>` after verifying that the original worker has exited.
Per-job `cancellation.json` prevents a queued job from starting and causes the
coordinator to stop its active worker. No Sports jobs are included.

No code, datasets, runs, or weights are uploaded by this experiment.
