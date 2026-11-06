# User-selected all-position neighbors and prefix background

The user fixed this method on 2026-11-04 after inspecting the Toys comparison.
There is no automated candidate selection in this suite.

## Fixed method

For a retained training chain A,B,C,D,E:

* Background b counts B,C,D,E once each, excludes the first item/padding, and normalizes sqrt(count+1).
* Each target position uses its previous three positions as neighbors, weighted by 1/distance.
  Each ordered position pair is counted once; earlier edges are not repeatedly counted for later prefixes.
* Neighbor self prior1, top16, neighbor mixture0.02, and history decay0.2 remain unchanged.
* Native Infomap leaf communities plus the catalog root are reused from the current raw-data baseline.
* Community/catalog shares are learned, initially0.2165/0.7835. Lambda=-log(0.1), reset0.035.
* Training samples are unchanged. Prefix expansion applies to statistics only.

The independent frozen numerical implementation is in `code/`. `protocol.py` contains
the shared experiment utilities copied from the completed comparison, with the old
selection/coordinator functions removed. The old comparison remains held and is not restarted.

## Stage 1: three datasets, three training seeds

Toys, Sports, and Beauty each use training seeds100/101/102, for nine main-method entries.
The completed Toys seed100 new-neighbor/new-b fit is reused after checking its
checkpoint, full validation log, complete configuration, data split hashes, and structural tensors.
The other eight main-method fits start fresh. Sports training cancelled in the earlier
background-only experiment is not resumed or reused; this is a new authorized method/run.

At most two workers train concurrently, starting with Sports100 and Beauty100.
Each worker finishes training, tests its validation-selected best checkpoint with three
sampling streams, and calculates long-tail results. All nine must finish these stages
before Stage2 begins.

## Stage 2: Toys factorial

Each of the following four arms uses seeds100/101/102:

| Arm | Neighbor mix | Structure | Rates |
| --- | ---: | --- | --- |
| neighbor_community | 0.02 | Native community/catalog | Learned two-level shares |
| neighbor_root | 0.02 | Catalog root | Same full Lambda, one level |
| shared_community | 0 | Native community/catalog | Learned two-level shares |
| shared_root | 0 | Catalog root | Same full Lambda, one level |

All arms use the same new background b. Neighbor tables, where used, use all-position
lag3 statistics. Root has no share gate. Reset and total Lambda remain identical.
The three Toys main-method fits from Stage1 are reused in neighbor_community, so only
nine additional factorial fits are needed.

Overall there are eighteen distinct training runs: one already complete and seventeen new.
The nine main entries plus twelve factorial entries total twenty-one table entries;
three entries reuse the same main-method training and must not be counted as extra runs.
Older Choice factorial results use different splits, learning rate, patience, and TF32
and cannot be mixed into this experiment.

## Shared training and evaluation

* Python `E:/anaco/python.exe`, CUDA0, four CPU threads per worker, TF32 disabled.
* Learning rate0.0006, no epoch cap, patience150 epochs, validation every10 epochs.
* Checkpoint selection uses validation NDCG@10 with one sampling stream.
* Full-catalog test HR/NDCG@5/10 averages three sampling streams from one trained model.
* Long-tail popularity counts retained training history and target occurrences.
  The frequency at the top20% item boundary is calculated separately for each dataset;
  strictly lower counts are tail, boundary ties are head.
* Report tail-target HR/NDCG@5/10, tail exposure, catalog coverage, and list incidence.
* Average streams inside each run, then report mean and sample standard deviation
  over the three training seeds. The reused seed100 participated in method exploration.

Datasets, mappings, splits, and each dataset's partition_seed100 are fixed across seeds.
`runs/manifest.json` records exact tail thresholds, catalog/community counts, code hashes,
run dependencies, configurations, and reuse provenance. No runs or weights are uploaded.

## Running and inspection

From `E:/Grow`:

```
E:/anaco/python.exe -B comparisons/all_positions_three_datasets_20261104/test_schedule.py
E:/anaco/python.exe -B -u comparisons/all_positions_three_datasets_20261104/run_suite.py --prepare
E:/anaco/python.exe -B -u comparisons/all_positions_three_datasets_20261104/run_suite.py
```

Start only one coordinator. `runs/launcher.pid`, `status.json`, `launcher.log`,
and `launcher.err.log` describe its state. Each job has worker logs, full epoch history,
training/test/long-tail reports, and optional `reuse.json`.
`runs/summary.json` and `summary.md` separate main-dataset results from the Toys factorial,
including three-seed means/sample deviations and paired effects of hierarchy/neighbors.
Progress always derives best validation NDCG@10 and its epoch from the full JSONL log.

`runs/execution_control.json` can hold new jobs or disable the automatic transition.
Per-job `cancellation.json` prevents launch and terminates an already active worker.
The coordinator never retries incomplete training: saved weights lack optimizer state.
After a completed fit, failed inference may be completed with `--worker-job <job.json>`
only after verifying that its earlier worker is no longer active.
# Execution change on 2026-11-04

The user cancelled the Toys factorial. Its twelve queued entries have cancellation
markers and `automatic_factorial_after_main_complete` is false. The nine current
main jobs continue unchanged. After all nine finish training, testing and long-tail
analysis, `../all_positions_old_background_three_datasets_20261104/` runs new
neighbors with the original background on all three datasets and seeds100/101/102.
The historical plan below is superseded for its factorial phase.
