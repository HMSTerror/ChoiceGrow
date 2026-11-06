# All-position neighbors with the original background

The user cancelled the Toys four-arm factorial on 2026-11-04. Finish all nine
new-neighbor/new-background runs in `../all_positions_three_datasets_20261104/`
first, including their tests and long-tail analyses. Then run Toys, Sports and
Beauty with training seeds 100, 101 and 102 using new neighbors and the original b.

Only the background statistics change relative to that suite. Neighbors count
all retained sequence positions with lag3 and weights 1/distance. The original
background counts only each training row's final target and normalizes
sqrt(count+1). Model-training samples do not expand. Keep the same data splits,
neighbors, native community partitions, neural architecture and training settings.
All runs use learned community/catalog rates initialized at 0.2165/0.7835,
learning rate 0.0006, patience150, and two concurrent GPU workers.

The completed Toys seed100 all-position/target run is reused after verifying
its configuration, selected checkpoint, data splits and structural tensors.
The remaining eight fits start only after all nine predecessor jobs complete
and the predecessor coordinator exits. This suite performs no factorial,
version selection, training restart or extra seed runs.

Each selected checkpoint is tested with three sampling streams; long-tail
metrics reuse the same train-only popularity masks as the current suite.
Average streams within each model, then report the mean and sample standard
deviation across three training seeds. Test reports also retain learned shares.

`runs/status.json` records the dependency and per-job progress. `summary.json/md`
contain completed results and seed aggregates. During the wait, this coordinator
uses no GPU. The cancelled factorial's historical files remain available.

```powershell
E:/anaco/python.exe -B run_suite.py --prepare
E:/anaco/python.exe -B test_schedule.py
E:/anaco/python.exe -B -u run_suite.py
```
