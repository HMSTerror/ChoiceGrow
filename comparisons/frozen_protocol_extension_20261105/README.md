# Frozen protocol extension

The suite runs fourteen new fits in three ordered batches. Two GPU slots fill continuously inside each batch; a worker remains active through selected-checkpoint test and long-tail evaluation. Later batches wait for all preceding workers to complete evaluation.

| Batch | Dataset / training seeds | Runs |
|---|---|---|
| 1 | Beauty100 (K12), Sports100 (K16) | bn + nn_clean, Root / hr.10 / hs.10 |
| 2 | Toys100 | bn + n0 + Root; bo + nn_clean + Root |
| 3 | Toys101/102 (K12) | bn + nn_clean, Root / hr.10 / hs.10 |

The protocol uses gamma0.02 except the explicitly requested n0 control at gamma0, patience300, no epoch cap, and unchanged eta0 guided-score ranking. Other numerical settings are copied from the completed matched Toys Root patience300 fit. Training/validation/test splits and the reviewed Step0 hs/hr labels remain fixed, including across training seeds. Complete live/EMA/AdamW/process/RNG/shuffle state is saved after every epoch.

`runs/input_audit.json` records reconstructed bn and clean-neighbor checks against Step0. `runs/manifest.json` records all jobs and three reused Toys100 fits. `runs/execution_control.json` owns scheduling controls. Batch reports contain overall/tail HR/NDCG@5/10 in JSON, plus a compact Markdown comparison. Toys three-seed reports average sampling streams within each fit and then use sample SD across training seeds100/101/102.

```powershell
# Read-only result refresh; never starts training.
E:/anaco/python.exe -B E:/Grow/comparisons/frozen_protocol_extension_20261105/run_suite.py --summary-only
```

The coordinator is already launched; do not launch duplicate workers. After an interruption, inspect `worker_status.json`, actual stderr, process identity and the complete `training_state.pt` before using the explicit worker `--resume` option. An EMA-only checkpoint is never a full-state continuation.

`efficiency.jsonl` separates train compute time from full-epoch time, which includes scheduled validation and checkpoint/state serialization. Training timings disclose two simultaneous GPU fits. After all fits finish, the coordinator runs an isolated evaluation-only benchmark of retained Toys100 Root and hs.10 checkpoints (full-split warmup, then three repetitions). The benchmark retains validation1/test3 sampling streams, excludes loading, and adds no training epochs. Reports are written to `runs/efficiency_comparison.json` and `.md`.

Numerical modules in `code/` are snapshots of the reviewed fixed-share suite. The new runner adds orchestration and timers without modifying training arithmetic, validation selection, reverse sampling, final scoring, or the old results.
