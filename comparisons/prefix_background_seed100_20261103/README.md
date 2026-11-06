# Prefix-target background comparison

This isolated seed-100 experiment changes only the counts used to construct the
shared background distribution `b`. For retained history `[A,B,C,D]` and final
target `E`, expand `[A]->B`, `[A,B]->C`, `[A,B,C]->D`, `[A,B,C,D]->E` for counting
purposes. Count `B,C,D,E` once each, excluding `A` and padding. Repeated item
occurrences remain separate events. Keep `b = normalize(sqrt(count + 1))`.

Prefix expansion does not add training examples or modify neighbor tables.
Histories remain capped at ten items; older events outside the prepared histories
are not recovered. Only training rows contribute statistics.

The reference suite is `E:/Grow/runs/raw_seed100_lr0006_patience150_20261103`.
Reuse its exact prepared splits, neighbor tables, weights, and native community
labels. Train a new model for each of Toys, Beauty, and Sports with seed 100,
learning rate 0.0006, patience 150, no epoch limit, validation every ten epochs,
and learned community/catalog rates initialized to 0.2165/0.7835. All other
training and sampling settings match the reference suite.

`code/` freezes the current main-method modules. Only `code/artifacts.py` differs
from the main implementation; it replaces the background and copies the other
baseline artifacts unchanged. `snapshot.json` records the source revision and
file hashes. The main method's source and existing experiments are untouched.

```powershell
E:/anaco/python.exe -B test_background.py
E:/anaco/python.exe -B run_suite.py --prepare
E:/anaco/python.exe -B -u run_suite.py
```

Prepare and start the suite only once. The coordinator starts two GPU workers
and queues the third dataset. Each worker trains from scratch, selects by
validation NDCG@10, evaluates the best checkpoint with three sampling streams,
and derives long-tail metrics from the cached recommendations. Three streams
are evaluations of one trained model, not three training seeds.

Long-tail labels use the unchanged training-history-plus-target frequency
definition, with the top-20% boundary assigned to the head group. Neither the
tail definition nor evaluation candidates are changed by this experiment.

Outputs live under `runs/` and are ignored by Git. `runs/status.json` tracks
workers and their best validation NDCG@10/epoch. Each dataset has its own logs,
weights, `verification.json`, `training.json`, `report.json`, and `long_tail.json`.
`runs/comparison.json` and `runs/comparison.md` pair the baseline and comparison
results as datasets finish. Regenerate these summaries without training:

```powershell
E:/anaco/python.exe -B run_suite.py --summary-only
```
