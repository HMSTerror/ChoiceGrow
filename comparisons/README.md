# Frozen research protocol and diagnostic source

The current protocol uses full-sequence background `bn`, history-only neighbors `nn_clean`, `gamma=0.02`, fixed layer shares, patience300, and the original eta0 guided-score ranking. The specified no-neighbor ablation uses gamma0. Main-method defaults in the repository-root configuration remain available separately.

| Directory | Purpose |
|---|---|
| [stratified_scope_20261104_1933](stratified_scope_20261104_1933/README.md) | Step0 training-only graph, spectral features, frequency-stratified hs labels and matched hr control; Toys/Beauty K12, Sports K16 |
| [fixed_shares_clean_toys_20261104_2045](fixed_shares_clean_toys_20261104_2045/README.md) | Reviewed Toys calibration, fixed-share comparisons, patience alignment, full-state continuation, and evaluation-only reverse-trajectory/score diagnostics |
| [frozen_protocol_extension_20261105](frozen_protocol_extension_20261105/README.md) | Fourteen new fits in three batches: Beauty/Sports100, Toys100 module controls, then Toys101/102; three-seed aggregation and evaluation-only efficiency benchmarks |

Install the additional analysis and orchestration dependencies in the project's Python environment:

```bash
python -m pip install -r requirements-experiments.txt
```

`bn` counts every retained successor once: a retained chain A,B,C,D,E contributes B,C,D,E. Training samples are not expanded. `bo` counts the final training target only. Both use the same float32 square-root-plus-one normalization.

`nn_clean` counts history-to-history pairs at lags1..3, weighted by inverse lag. It excludes the sample's final supervised target. One training-built table, with the existing self prior, Top16 selection and normalization, serves every split. `n0` sets the destination mixture coefficient to zero.

Root has one catalog level with shares `[1.0]`; hs/hr.10 have community/catalog shares `[0.1,0.9]` and no learned gate. Reviewed partition labels stay fixed across training seeds. Three sampling streams evaluate one fitted model; they are averaged before computing mean and sample SD across training seeds100/101/102.

The runners target the preserved local experiment tree and record exact data/result paths. They require the prepared splits, reviewed partition artifacts and referenced completed results. These inputs, weights, logs, generated diagnostics, and local design notes are excluded from Git. The frozen numerical snapshots preserve each experiment's implementation; they do not change the root entry point's preprocessing defaults.
