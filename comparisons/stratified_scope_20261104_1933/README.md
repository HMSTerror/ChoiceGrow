# Stratified-scope CPU diagnostics

This isolated comparison implements Step 0 requested on 2026-11-04. Existing code, completed experiments, checkpoints, and task queues are read-only inputs. No GPU training or model inference is launched here. Step 1 requires the user to review Step 0 first.

Run with `E:/anaco/python.exe -B -u diagnostics_step0.py` from this directory. CPU thread counts are limited to two. Results are written to `diagnostics/`, partition artifacts to `partitions/`, and source input paths to the JSON report.

Generated English results are in `diagnostics/report.md` and `diagnostics/check_a.json`, `check_b.json`, `check_c.json`. Reports and partition arrays remain local and are excluded from Git. `test_diagnostics.py` contains six small CPU checks, including the paired-deletion zero-probability rank regression. The separate local review formatter and artifact verifier are not part of the published English source subset.

`finalize_step0.py` reuses saved partition artifacts to finish or regenerate the neighbor/statistics portion without rerunning SVD or clustering. Then run `supplement_evidence.py` for background and historical Real-Scope audits. This does not train or evaluate a model. Step0 was subsequently reviewed; the approved fixed-share training and later frozen extension live in the [comparison suites](../README.md). The original validation and early-stopping logic was retained without a minimum-step guard.

`scope.py` owns history-only graphs, spectral features, stratified capacity assignment, and matched random controls. `neighbors.py` owns history-only and legacy neighbor statistics. Both are independent of the active training source tree.

Full frequency strata have exactly M items per cluster. The final incomplete stratum uses shared floor/ceiling capacities with size difference at most one. Equal counts in a frequency band do not imply exactly equal background masses or identical within-band frequency distributions. These deviations are measured.

The existing evaluator already resets its validation RNG each call. Existing EMA ramps its decay. Step 0 checks these facts instead of assuming the supplied causal explanations are established. Training-versus-validation gaps are descriptive; paired removal of each training sample's own final-target contributions more directly isolates self-inclusion in neighbor statistics.
