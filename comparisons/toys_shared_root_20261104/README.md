# Toys: new background, no neighbors, no hierarchy

Run three fresh training seeds: 100, 101, 102. The background is the current
prefix-count b: count B, C, D, E once for A -> B -> C -> D -> E, then normalize
sqrt(count + 1). Model-training samples are not expanded.

The numerical code and protocol.py are copied unchanged from the existing
Toys Root experiment. Each seed reuses the exact prepared splits, base, and
neighbor tensors from the completed all_positions_three_datasets_20261104
native-community run. Setting neighbor_mix=0 bypasses the neighbor tables
and gives rho=b. The partition has only one catalog-wide block, and its
single rate is the original total_budget; dynamic_rate_gate is disabled.

The generator therefore reduces to
Q=((1-reset_rate)*Lambda+reset_rate)*(G-I), with G[i,j]=b[j].
Preparation checks this kernel against the analytic matrix exponential.
The denoiser continues to condition on user history.

Keep lr=0.0006, patience=150, validation_interval=10, reset_rate=0.035,
Lambda=-log(0.1), and every other matched training/inference setting.
Select the best checkpoint by validation NDCG@10. Test with three sampling
streams and compute long-tail accuracy and exposure from full-catalog ranks.
The tail rule is training frequency < 12: 9367 items, 1035 tail targets,
1942 total test targets. Streams are not independent training runs.

## Execution

    E:/anaco/python.exe -B -u E:/Grow/comparisons/toys_shared_root_20261104/run_ablation.py --prepare
    E:/anaco/python.exe -B -u E:/Grow/comparisons/toys_shared_root_20261104/run_ablation.py

The persistent coordinator waits for the random-three-level queue and the
neighbor-enabled Root queue to drain. It then admits at most two globally
active workloads, checking actual process identities and GPU memory before
each launch. Its distinct run_ablation.py entry point avoids blocking the
upstream Root coordinator, which checks for other run_suite.py coordinators.
The old-background suite remains suspended.

Progress is saved in runs/status.json and runs/manifest.json. Per-seed files
are under runs/shared_root/seed_*. Results and paired comparisons are saved
in runs/summary.json and runs/summary.md. The neighbor_root comparison isolates
neighbor weighting; the full two-level method differs in both factors.

No optimizer state is stored. Interrupted partial training is marked failed;
only evaluation after a completed fit may be retried.
