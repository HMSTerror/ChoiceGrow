# Toys fixed-share hierarchy comparison

This suite implements the reviewed Step 1/2 protocol. All runs use Toys training seed100, bn background, Lambda=-log(0.1), reset=.035, learning rate=.0006, and the unchanged training/validation/early-stopping code. Validation remains one fixed RNG stream every 10 epochs, patience150. There is no min_steps setting. Test metrics average three sampling streams of one fitted model.

The user has selected **1A: bn + nn_clean + gamma=.02** for Step 2. `runs/step1_selection.json` records this explicit decision. It supersedes the earlier validation-argmax recommendation of 1B; the user reviewed both validation and test results. Step 2 does not run another automatic neighbor selection. Root share0 reuses the completed 1A checkpoint, whose logged best validation NDCG@10 is .04964138047711128 at epoch1520.

The 2026-11-05 protocol revision replaces queued share.80 with share.10 and gives the four fresh share.10/.50 fits patience300. The original `hr_s03` fit finishes unchanged at patience150. The completed patience150 `hs_s03` result remains intact in its original folder and is recorded under manifest `historical_results`. The user explicitly chooses a fresh seed100 hs.30 rerun in `runs/step2/hs_s03_extended/`, training exactly1670 epochs. It does not load the old EMA-only weights. This rerun bypasses early stopping to reach the fixed horizon, retains validation every10 epochs, and selects its test checkpoint by the best validation NDCG@10 over all1670 epochs.

The six structural jobs have finished. A separate `patience_alignment.py` now audits the .10 fits and supplies the missing Root patience300 control under `runs/patience_alignment/`, preserving the original completed suite. `log_audit.md/json` replay the exact strict-improvement and validation-epoch stopping rules; the hypothetical patience150 hs1450/hr510 weights are no longer retained. Their test scores cannot be inferred from logs. The Root control is a fresh same-seed replay because the old Root has only EMA checkpoints, without AdamW/live model/RNG state. Its configuration differs from1A only in patience150→300, its structure/data references are copied unchanged, and it saves complete state every epoch. `trajectory_audit.json` compares the replay's shared log prefix to the original, and `retained_weight_audit.json` checks exact EMA tensor agreement at epochs1520 and1670. The replay runs until natural patience300 stopping, without a hard epoch limit. `comparison.md/json` contains Root/hr.10/hs.10 under patience300. A changed selected Root model gets a fresh3-stream test and tail evaluation; an identical selected epoch/model/validation may reuse the original prediction caches with explicit provenance.

`full_state_training.FixedHorizonTrainer` inherits the frozen numerical Trainer and changes only stopping and additional state storage. It and the queued hs/hr.50 fits now write `training_state.pt` after every completed epoch through the main `training_state.TrainingStateStore`. The snapshot contains live model, EMA, AdamW, process/gate, EMA update count, CPU/CUDA/Python/NumPy RNG states, the DataLoader shuffle generator, best-validation metadata, the selected best checkpoint, the completed log prefix and elapsed-time metadata. Disk round-trip tests check continuation, including shuffled batches and live/EMA weights. The currently active .10 fits remain on their already-loaded original writer; no resumable state is fabricated for them.

Resume is explicit: after ensuring the worker is stopped, a full-state job can use `E:/anaco/python.exe -B -u run_experiments.py --worker-job <job.json> --resume`. If extending its stopping horizon, update only its resolved configuration's total epochs/patience before this explicit command; the snapshot rejects numerical-setting changes. Coordinate the job state with the suite before manually resuming a completed worker. Automatic failure recovery never restarts a training fit. Original EMA-only runs remain evaluation-only.

After the user enabled continuous admission, the original patience150 `hs_s05` worker briefly started. It was stopped to apply the new protocol, and its partial files, the unused old `hr_s05` configuration, and canceled .80 configurations are preserved in `runs/archived/patience150_queue_before_revision/`. Fresh .50 fits use separate recreated directories. `runs/protocol_revision_patience300.json` records the prior manifest and the preserved hr.30 process identity.

Before the manual hold, the old coordinator selected 1B and started gamma=.10 `hs_s03`. It was stopped at epoch97. All six old Step 2 directories, including that partial fit, are preserved in `runs/archived/step2_gamma010_before_manual_1A/`. None contributes to the new gamma=.02 results. The historical recommendation remains in `runs/step1_selection_automatic_before_manual_hold.json`. `runs/manual_decision_hold.json` records the stop; `runs/step2_activation_1A.json` records the fresh configurations and their verification. The interrupted fit has no optimizer state and is not resumed.

Step 1 is complete. Step 2 keeps up to two suite workers active, filling each free slot immediately from the queue:

1. Train 1A: nn_clean + Root, gamma=.02.
2. Train 1B: nn_clean + Root, gamma=.10, alongside 1A when memory permits.
3. Reuse completed nn + Root gamma=.02 seed100 as candidate1C. Its original best validation NDCG@10 is .0471012117 at epoch1130. Reuse test predictions and compute missing final validation-tail results.
4. Use the user-selected 1A configuration unchanged except for partition and fixed shares.
5. Keep active .10 fits running; prioritize the approved fresh fixed1670 hs.30 rerun for the next free slot; then admit the .50 fits with patience300. Once any worker finishes training and final evaluation, the next queued job may start even if the other worker has a different share. `runs/execution_control.json` has `step2_pair_by_share=false`. Both partitions are loaded from the reviewed Step 0 artifact, each with K=12. They are never reclustered for a training run.

There are eight new fits and one reused fit. Run1C does not introduce an unspecified extra nn gamma=.10 experiment. Existing runs and their outputs remain unchanged. The new coordinator leaves unrelated Beauty/Sports workers running and admits a new job only when at least1800 MiB GPU memory is available.

The coordinator adopts an existing worker by its exact job-spec command; replacing only the coordinator does not interrupt or restart its fit. `runs/coordinator_handoff_parallel.json` records the historical 1A handoff. Current coordinator output is in `runs/full_state_launcher.log` and `full_state_launcher.err.log`; prior launcher output remains separate. The frozen numerical source and validation streams are unchanged. Job metadata records the patience override and the one explicit fixed-horizon rerun. `runs/full_state_policy.json` lists jobs opting into the per-epoch complete-state writer.

`transition.py` adds explicit fixed shares with no gate. All other numerical modules in `code/` are snapshots of the existing main implementation. The generator retains its shared background reset and exact kernel. Zero-share controls are supported; a single `[1]` is Root. Initial shares are absent from these new configurations because their shares stay fixed for the entire experiment.

Every newly trained model gets a final evaluation of its validation-selected checkpoint: one validation stream saved to `validation_recommendations.npz`, and three test streams saved to `recommendations.npz`. This supplies validation/test Overall and Tail NDCG@10 without altering training-time validation or stopping. Tail accuracy filters rows by the true target after full-catalog ranking. Tail membership is recomputed from the actual retained training histories plus targets, not copied from an older preprocessing protocol.

Start from this directory:

```powershell
E:/anaco/python.exe -B -u run_experiments.py --prepare
E:/anaco/python.exe -B -u run_experiments.py
```

Preparation is already done after the initial launch; do not repeat it. `runs/launcher.pid`, `launcher.json`, `manifest.json`, `status.json`, and per-run `worker_status.json` describe actual progress. Full log scans report the current best validation result, independently of the latest epoch. `--summary-only` refreshes output tables without training.

`runs/summary.json` and `summary.md` retain active Step 1/2 entries. `runs/step2_comparison.json` and `.md` contain the selected Root plus six current structural jobs, including the actual stop rule (patience or fixed horizon), stop epoch, logged best validation NDCG@10 and epoch, test overall HR/NDCG@10, test tail HR/NDCG@10, and tail exposure@10. Tables refresh as workers finish. Final plots compare both structures with the same reused Root point.

Failures stop the chain and preserve the actual traceback. Completed training may resume missing evaluation; partial training checkpoints omit optimizer state and are never silently restarted or described as exact resumes. `runs/execution_control.json` can hold future jobs or request stopping after the active worker exits.
