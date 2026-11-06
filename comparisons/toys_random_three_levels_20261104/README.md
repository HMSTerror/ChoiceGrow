# Toys: native two levels versus original random three levels

The user paused the new-neighbor/old-b suite to run this comparison. Its coordinator
and two workers remain suspended with their training state in memory. Do not resume
them automatically, terminate them, or mistake their selected weights for full
optimizer checkpoints. `../all_positions_old_background_three_datasets_20261104/runs/pause.json`
records their process IDs and creation times for a later user-authorized resume.

Use all-position lag3 neighbors and prefix-count b on Toys with training seeds
100/101/102. Reuse all three completed native-community/catalog two-level runs.
Train only the three original random64/random1024/catalog variants. The permutation
and rank construction is copied from Choice/experiments/diffusion.py:36, with
partition seed100 fixed across training seeds. The resulting block counts are
187 fine, 12 coarse and one catalog block for the retained 11,924-item catalog.

Both structures learn their rate shares with the same three-feature/16-hidden-unit
gate. Native two-level shares start at 0.2165/0.7835. Random three-level shares
start at 0.1920355/0.0244645/0.7835, retaining the original fine/coarse split of
the same non-root budget. Total Lambda=-log(0.1), reset0.035, learning rate0.0006,
patience150, exact prepared splits and all other numerical settings are unchanged.
The local code copy extends share validation, learned-rate level count and rate
reporting to three levels; the existing generic exact kernel already supports it.

This comparison changes both the partition and its depth. It cannot separately
attribute a difference to community discovery versus the number of levels.
Selected checkpoints use validation NDCG@10. Each is tested with three sampling
streams, followed by the same train-only long-tail evaluation. Report stream
means within a model and training-seed mean/sample standard deviation across models.
The suite performs no automatic version selection and resumes no other tasks.

```powershell
E:/anaco/python.exe -B test_random_levels.py
E:/anaco/python.exe -B -u run_suite.py --prepare
E:/anaco/python.exe -B -u run_suite.py
```
