# Publication-readiness audit for DreamerV3 on TTCP CAGE 2

**Decision — GO for the descriptive LinkedIn comparison figures.** The six-run development curves, final seed distributions, and matched frozen-opponent figures are ready. The resource figure remains conditional on its measurement labels. The evidence does not establish hierarchical superiority, equivalence, or a leaderboard result.

## Scope and integrity

The comparison includes **all three host-aware flat replications and all three host-aware hierarchical runs**, seeds **0, 1, and 2** in each arm. The earlier flat pilot is a separate historical reference. Interrupted flat attempts are outside the restarted six-run cohort and contribute no results.

Both supervisors exited with status **0**. All six workers were recorded as completed with exit code **0**, and each acknowledged a saved final checkpoint. The individual completion records, resolved simulator and training seeds, and checkpoint payloads agree with the supervisor records. Every worker reached **16,899 updates and 507,450 interactions**. The checkpoint schemas identify the flat and hierarchical arms as versions **4** and **5**, respectively. Independently calculated SHA-256 hashes match the individual and consolidated reports.

- Flat source: `cage2-host-aware-flat-replication-v1.1`, commit `31d1b0f8e53c329daaeda37a9f6b87335d6095d6`. The sealed manifest is at `dbb99fe1d3529fc9d47dc83e01bb03dd5405885f`. [Supervisor record](runs/cage2_flat_replication_20261126T033743Z/group/execution_summary.json) · [Flat exit status](runs/cage2_flat_replication_20261126T033743Z/exit_status) · [Manifest snapshot](runs/cage2_flat_replication_20261126T033743Z/group/job_001/results/cage2_host_seed0/manifest.json)
- Hierarchical source: `dreamerv3-hierarchy-experiment-v3`, commit `79b36459adac8c37f86bc97e40bae3ea02e4277b`. The sealed manifest is at `86a51e5e5ebf6b652221afc6e53a9799ff7a01a2`. [Supervisor record](runs/hierarchy_v3_20261125T153127Z/group/execution_summary.json) · [Hierarchy exit status](runs/hierarchy_v3_20261125T153127Z/exit_status) · [Manifest snapshot](runs/hierarchy_v3_20261125T153127Z/group/job_001/results/cage2_host_seed0/manifest.json)

The run-local manifest snapshots match their committed manifests. The annotated tag objects and peeled commits match the recorded provenance. Every recorded source-file fingerprint was rechecked against its tagged Git blob. The source inventory covered **345 files** for flat and **341** for hierarchy. No checkpoint, configuration, tag, or implementation file was modified.

| Arm | Seed | Final checkpoint SHA-256 |
|---|---:|---|
| Flat | 0 | `db5c5e65ddda372b0e32f0af38f0cefb0698d722fe6156e4052735d19b452221` |
| Flat | 1 | `0f944093bea21d5991a7cc4fa9de3bb11057ae07c20b647cc477a9645da1849d` |
| Flat | 2 | `2cb97f48931bfae0c55b1079d557c64c881d5fb723cf9bf27531c809caa30b1e` |
| Hierarchical | 0 | `b531a0a749db3a18712b2ae8e34296e24899dcdbe2d3bcee6a9adbd29ff572fd` |
| Hierarchical | 1 | `ebba2b04745979047d606358c61fce98ac8d914d8ff4f3450a760dd689dbe48e` |
| Hierarchical | 2 | `b41694eda82286519a7fe5e137336e1f83d30b3335ba6fa85dcdae691d423cdd` |

The checkpoint paths are in the two supervisor records. The fingerprint and final-save acknowledgement for each worker are in the [flat consolidated report](runs/cage2_flat_replication_20261126T033743Z/group/final_report.json) and [hierarchical consolidated report](runs/hierarchy_v3_20261125T153127Z/group/final_report.json).

### Comparability and confounds

- The resolved `env`, `world`, `ac`, and `train` settings agree within each arm except for the simulator and training seed fields. For the same seed across arms, the only difference in those fields is `ac.hierarchical`. Both use the host-aware encoder, Scenario 2 Blue against B-line, the 30-step cap, FP32, and disabled auxiliary losses. The pinned simulator revision, package versions, and native wheel fingerprint agree. Replay prefill was 480 transitions, followed by 30 interactions per update. The hierarchy uses eight goals with 16-dimensional embeddings and a four-action interval. [Flat resolved configuration](runs/cage2_flat_replication_20261126T033743Z/group/job_001/results/cage2_host_seed0/resolved_config.json) · [Hierarchical resolved configuration](runs/hierarchy_v3_20261125T153127Z/group/job_001/results/cage2_host_seed0/resolved_config.json) · [Sealed manifests above](#scope-and-integrity)
- Initialization records show fresh replay and optimizers, no source checkpoint, matched collection-RNG seeds, and identical recorded **non-encoder world** initialization hashes for the same seed across arms. The actor and critic hashes differ, as expected for different policy and value architectures. The intervention is the implemented hierarchical actor–critic package, including its temporal credit assignment. It does not isolate a manager, worker, goal embedding, or value-head mechanism. [Flat initialization record](runs/cage2_flat_replication_20261126T033743Z/group/job_001/results/cage2_host_seed0/initialization.json) · [Hierarchical initialization record](runs/hierarchy_v3_20261125T153127Z/group/job_001/results/cage2_host_seed0/initialization.json)
- The two cohorts were executed separately under different source tags. The tagged model, trainer, and root lockfile agree. Relevant differences include orchestration/reporting and optional opponent injection, which preserves B-line as the default. Neither the resolved configuration nor the archived scores identify an opponent change between training cohorts. Separate execution windows and the absence of a single common-tag experiment remain limitations, particularly for attributing resource differences to architecture. [Flat source tag and hierarchy source tag](#scope-and-integrity)
- The original pilot's `−28.64` is historical context. Flat replication seed 0 reproduces its individual episode returns at every archived development checkpoint, within numerical precision. The final checkpoint SHA-256 differs from the historical pilot, so the reproducibility claim concerns recorded returns. The historical policy is neither a fourth independent flat seed nor a substitute for a matched flat checkpoint. [Historical final summary](runs/cage2_host_aware_pilot_seed0_20261124/final_summary.json) · [Historical final episodes](runs/cage2_host_aware_pilot_seed0_20261124/evaluation_16899.json)

## Independently recomputed development statistics

Episode returns were rebuilt by summing per-step rewards in each archived episode and checked against each saved total and checkpoint-level mean. Every milestone and seed contains 20 episodes on the same development seed block, `1000000000–1000000019`, with greedy actions and posterior modes. Final arm means give each independently trained seed equal weight. The across-seed statistic is the sample SD of the three policy means.

| Seed | Flat mean | Flat episode SD | Hierarchical mean | Hierarchical episode SD | Hierarchical minus flat |
|---:|---:|---:|---:|---:|---:|
| 0 | −28.640 | 11.845 | −37.395 | 9.087 | −8.755 |
| 1 | −34.890 | 6.136 | −25.840 | 11.581 | +9.050 |
| 2 | −35.290 | 5.997 | −36.090 | 1.713 | −0.800 |
| **Equal-weight mean** | **−32.940** | | **−33.108** | | **−0.168** |
| **Across-training-seed sample SD** | **3.729** | | **6.328** | | **8.919** |

The episode SD columns describe returns within one fitted policy. An episode bootstrap conditions on that policy and estimates variability of its mean over episodes. For example, resampling 20 episodes 20,000 times with RNG seed `20261124` gives flat seed 0 a 95% percentile interval of **[−33.340, −23.295]**. The same method gives hierarchy seed 0 **[−41.240, −33.600]**. The reused development block makes these intervals descriptive of that evaluation regime. Neither the episode intervals nor the 60 episodes per arm estimate uncertainty across 60 training runs. The three paired differences vary in direction, and the small mean difference gives no basis for superiority or equivalence.

Raw inputs are `group/job_*/results/cage2_host_seed*/development_<update>.json` in the [flat run](runs/cage2_flat_replication_20261126T033743Z/group) and [hierarchical run](runs/hierarchy_v3_20261125T153127Z/group). The recomputed means and sample SDs agree with both consolidated reports.

## Matched held-out evidence and leaderboard scope

All six final checkpoints are now covered by the same frozen protocol. The three new flat replication checkpoints were evaluated using the unchanged rollout used for the hierarchy. The nine hierarchical cell logs were retained, checked against the archived checkpoint hashes, and recomputed from their raw episode rewards. The historical flat pilot stays in its separate evaluation and is excluded from every result below.

Every checkpoint–opponent cell has 100 episodes on the same simulator seeds `1837491000–1837491099`. Evaluation used Scenario 2, a 30-step horizon, greedy Blue actions, deterministic posterior modes, and native undiscounted rewards. Matching Python and dependency versions, lockfile hash, simulator revision, scenario hash, CPU thread count, and seed schedule constrain the protocol comparison. The flat cells were collected later than the hierarchy cells.

[Consolidated matched results](runs/cage2_frozen_matched_20261126T133406Z/report.json) · [Checkpoint and evaluation provenance](runs/cage2_frozen_matched_20261126T133406Z/provenance.json) · [Original frozen archive](runs/cage2_frozen_opponents_20261125T213227Z/report.md)

| Frozen checkpoint | B-line mean ± episode SD | Meander mean ± episode SD | Sleep mean ± episode SD |
|---|---:|---:|---:|
| Flat replication, seed 0 | −30.974 ± 9.668 | −26.509 ± 8.013 | 0.000 ± 0.000 |
| Flat replication, seed 1 | −37.574 ± 6.001 | −27.751 ± 5.560 | 0.000 ± 0.000 |
| Flat replication, seed 2 | −36.282 ± 6.312 | −27.535 ± 4.705 | 0.000 ± 0.000 |
| Hierarchical, seed 0 | −39.970 ± 8.644 | −27.726 ± 5.001 | 0.000 ± 0.000 |
| Hierarchical, seed 1 | −30.324 ± 11.130 | −21.923 ± 5.930 | 0.000 ± 0.000 |
| Hierarchical, seed 2 | −36.124 ± 1.957 | −29.757 ± 5.413 | 0.000 ± 0.000 |

Arm summaries give equal weight to the three independently trained checkpoint means. The paired difference is hierarchy minus flat at the same training-seed label. The sample SD in this table is across training seeds or seed-pair differences, as indicated. Episode SD above characterizes returns within one fixed checkpoint.

| Opponent | Flat mean ± seed SD | Hierarchical mean ± seed SD | Paired differences, seeds 0 / 1 / 2 | Mean difference ± pair SD |
|---|---:|---:|---:|---:|
| B-line | −34.943 ± 3.498 | −35.473 ± 4.856 | −8.996 / 7.250 / 0.158 | −0.529 ± 8.145 |
| Meander | −27.265 ± 0.664 | −26.469 ± 4.066 | −1.217 / 5.828 / −2.222 | 0.796 ± 4.386 |
| Sleep | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 / 0.000 / 0.000 | 0.000 ± 0.000 |

Report these as descriptive opponent-specific results for the matched-budget checkpoint cohort. Three training pairs are insufficient for a superiority or equivalence claim. The official submission evaluator covers 30, 50, and 100 steps, and leaderboard validation uses 1,000 episodes per horizon–opponent combination with a different aggregation. [Vendored TTCP CAGE 2 protocol](environments/cage2/vendor/cage-challenge-2/README.md#evaluation-of-submissions)

## Figure readiness

| Proposed figure | Status | Source and axes | Uncertainty and caption requirements |
|---|---|---|---|
| Matched-budget learning curves | **READY** | Use all six `development_curve.json` files and their `development_<update>.json` episode records. X is environment interactions. Y is mean development episode return. Plot individual seed traces at 510, 127,800, 254,340, 380,910, and 507,450 interactions. | Each point averages 20 development episodes. If an arm envelope is shown, label it as sample SD across three seed means. Connect points as a visual guide between measured checkpoints. An optional pilot reference must be labeled historical and excluded from the six-run summary. |
| Final development return across seeds | **READY** | Use the final episode records at update 16,899 and the two consolidated reports. X is actor–critic variant. Y is mean development episode return at 507,450 interactions. Plot all three seed means in each arm. Matching seed labels or connecting seed pairs makes the disagreement in direction visible. | An arm summary may show mean ± sample SD across three seeds. Any within-policy interval should be displayed separately and labeled as a conditional episode bootstrap. Caption the result as descriptive and avoid superiority, equivalence, or significance claims. |
| Frozen checkpoint evaluation by opponent | **READY** | Use the [matched cell logs and summary](runs/cage2_frozen_matched_20261126T133406Z/report.json). X is opponent and Y is mean frozen-policy episode return. Plot all three seeds per arm. | Label 100 shared evaluation episodes per cell at 30 steps. For individual checkpoint error bars use labeled episode SD. If showing equal-weight arm summaries, separately label sample SD across three training-seed means. |
| Training throughput and GPU temperature | **CONDITIONAL** | Use both `execution_summary.json` and `thermal.jsonl` files. Report group throughput with X as the run cohort and Y as aggregate environment interactions per second. For the temperature panel, use elapsed wall time since the first watchdog sample on X and GPU temperature in °C on Y. | Treat throughput and telemetry as observations from one group launch per cohort. A temperature range summarizes observed samples. It is not a confidence interval. Put the observed concurrency of three in the caption. Avoid an architecture-efficiency claim, because the groups ran in separate windows and no repeated, controlled throughput crossover was archived. |

The supervisor reports **101.11 aggregate interactions/s** for flat and **90.70** for hierarchy, computed from completed group interactions divided by the first-worker-to-last-worker training window. Watchdog samples with numeric GPU temperature give flat **49 to 68 °C**, mean **60.07 °C**, across **1,506 samples**. Hierarchy gives **42 to 68 °C**, mean **59.55 °C**, across **1,679 samples**. The means are sample-weighted over the logged watchdog windows, including launch and shutdown edges. [Flat supervisor record](runs/cage2_flat_replication_20261126T033743Z/group/execution_summary.json) · [Flat telemetry](runs/cage2_flat_replication_20261126T033743Z/group/thermal.jsonl) · [Hierarchical supervisor record](runs/hierarchy_v3_20261125T153127Z/group/execution_summary.json) · [Hierarchical telemetry](runs/hierarchy_v3_20261125T153127Z/group/thermal.jsonl)

## Scientific claim readiness

| Claim | Status | Publication wording or restriction |
|---|---|---|
| All six runs completed the same interaction and update budgets. | **READY** | Report the fixed budget and show all three training seeds per arm. |
| Flat seed 0 reproduced the archived pilot’s development returns. | **READY** | Specify matching recorded returns and distinguish the two checkpoint files. |
| The two arms had similar observed final development means. | **READY** | State both means, the three seeds per arm, and the direction-changing seed differences. This is descriptive. |
| The development results are a matched-budget actor–critic comparison. | **CONDITIONAL** | The budget, environment, dependency records, development seeds, and shared non-encoder world initialization are matched. Qualify the separate tags and cohorts and describe the comparison as one of the implemented actor–critic variants. |
| The matched frozen policies show opponent-specific behavior on fresh 30-step seeds. | **READY** | Show all six checkpoints and all opponents, with paired seed differences and distinct episode and training-seed SDs. |
| Hierarchy or flat is superior, equivalent, or demonstrably more efficient. | **NOT SUPPORTED** | Three seed pairs with mixed effects, separate launches, and development-only comparison cannot establish these claims. |
| The matched frozen results establish superiority, equivalence, broader generalization, or leaderboard standing. | **NOT SUPPORTED** | Three training pairs and a local 30-step protocol support only descriptive inference within the evaluated setting. |

## Remaining evidence

1. **For an architecture-effect or equivalence claim** — More independent paired training seeds, a prespecified meaningful effect or equivalence margin, and a common-source analysis sized for training-seed uncertainty.
2. **For an efficiency claim** — Repeated or counterbalanced benchmarks under controlled hardware conditions with phase-aligned throughput and telemetry.
3. **For a leaderboard claim** — Evaluation using the official horizons, opponent protocol, and aggregation.

**GO for the descriptive final LinkedIn comparison figures.** Use all six checkpoints in the development and matched frozen-opponent panels. Keep the historical pilot separate. Include the resource panel only with the qualifications above. No policy training, tuning, or selection was performed for this evaluation.
