# Agent Note: isolate structural diagnostics before GPU experiments

Status: implemented

## Problem

The requested design mixes testable construction choices with unverified causal claims. Existing results must remain unchanged, and Step 1 must wait for review of CPU diagnostics.

## Decision

Step 0 lives entirely in this timestamped comparison. The neighbor table and clustering graph use training histories only in clean mode; the background remains bn, including supervised targets as explicitly requested. Clusters share per-stratum capacities with a random control. The incomplete stratum uses floor/ceiling capacities, avoiding the reference snippet's unbalanced truncated-column allocation. Both partitions have identical counts per stratum, not necessarily identical weighted mass.

The diagnostic records current CRN behavior, EMA ramp-up, matched-epoch validation values, and exact neighbor effects of removing each sample's own final-label edges. Existing Infomap and clean Infomap are compared, with a clean adjacent-only control separating label exclusion from changing the graph's lag window. Only CPU analysis runs.

The unstratified size-only baseline uses an exact transportation LP with HiGHS interior-point solving and crossover to an integral solution. It minimizes the same cosine assignment objective as repeated cluster slots, without allocating a full-catalog square matrix. The default automatic LP method is replaced after excessive runtime; this changes the solver, not the constraints or objective.

## Alternatives considered

- Copying the supplied remainder assignment verbatim: rejected because it can leave the final clusters underfilled or empty in the remainder.
- Treating Train/Val gaps as a causal proof of leakage: rejected because user cohorts and support differ; paired edge deletion is reported as a more direct neighbor-statistics intervention.
- Starting GPU calibration immediately: excluded by the user's explicit Step 0 review gate.
- Changing the main source tree or existing experiment outputs: excluded to preserve ongoing and completed work.
- Five-fold graph construction: not used; one shared history-only table supplies identical structural artifacts to train, validation, and test. Its self-history influence is disclosed.
- Disabling LP crossover: rejected after a real-data probe returned fractional memberships (maximum fractionality about 0.362) and little runtime benefit. The baseline retains integral crossover solutions and explicit capacity checks.

## Consequences

The new construction is reproducible and does not create a separate graph density for held-out users. It removes direct final-label contributions but retains training-history self-influence and a target-inclusive global bn background. Spectral affinity is collaborative structure, not verified product semantics. Frequency balance, assignment fidelity, and predictive block mass are measured rather than assumed. Fixed-share training and early-stop changes remain gated until Step 0 is reviewed.

## Numerical correction and reporting definitions

Paired deletion now reconstructs each destination by adding nonnegative corrected source rows instead of subtracting normalized mixtures. A catalog-sized tied-background fixture reproduced a positive cancellation residue of 2.7756e-17 at a mathematically zero target; its diagnostic rank changed from 17 to 7. Reconstructing the mixture preserves the exact zero and the expected rank. This also avoids large sparse LIL row mutations. The six CPU checks include this regression.

The saved training tables use float32 normalization. Replaying the original implementation reproduces their IDs and weights bitwise; the diagnostic float64 version has identical support and maximum rounding deviation 1.0218e-7. These dtype effects are recorded separately from the zero-support correction.

Clustering fidelity and Head-share balance use the requested top floor(0.2*N) items ranked by bn. Existing long-tail evaluation instead uses a strict history-plus-target count boundary. Both definitions are persisted, and the tables name their denominator. Historical Real-Scope artifacts were found for all three datasets; their arrays differ from current inputs and an original-ID mapping has not been established, so only their own-background balance statistics are reported.
