# Closeout artifact and provenance inventory

Paths beginning with `runs/` resolve against the **main workspace checkout**. They are preserved, ignored artifacts. Short evidence IDs map to the [claims matrix](claims.md). Reports and software in this directory are Git-tracked documentation. Completion status and exit codes below come from saved manifests.

## Diagnostic runs

| ID and run root | Report and measurements | Execution and source |
|---|---|---|
| **T1** `runs/minihack_tmaze_representation_diagnostics_20261126/` | `failure_mode_report.md`, `diagnostic_results.json`, and 65 JSON records under `checkpoints/`. The records include fit/validation/test IDs, stage distances, posterior entropy/KL/JS, per-factor utilization, samples, controls, and checkpoint provenance. | `execution_manifest.json`, `completion_manifest.json`, `launch_command`, `console.log`, `exit_status`, `attempts.tsv` and `supervisor_20261126T181035064517Z/`. Session `dreamer-memory-diagnostics`, Python worker 497093, exit 0. Model source `4746126`; diagnostics source bytes recorded by filename and SHA-256, later committed as `e228ab6`. |
| **T2** `runs/frozen_memory_integration_20261126/` | `tmaze/report.md`, `tmaze/results.json`, `tmaze/seed{19,23,29}.json`. Contains per-level decision outcomes, right-action probabilities, values, predicted reward margins, paired recurrent distances, selected reset factor indices, and six control conditions. | Root `completion_manifest.json` and `tmaze/execution_manifest.json`, `supervisor_20261126T183740299246Z/job_001/console.log`. Session `dreamer-frozen-memory-studies`, Python worker 516054, exit 0, source `519001c`. |
| **C1** Same integration run | `cage2/report.md`, `cage2/results.json`, `cage2/archive_parity.json`, `cage2/replay_audit.json`, `cage2/offline_session_labels.npz`. Includes split IDs, label support, forecast parity, visible-field reconstruction, paired history ablations, classification metrics, and per-factor measures. | `cage2/execution_manifest.json`, same root completion manifest, `supervisor_20261126T183740299246Z/job_002/console.log`. Python worker 516180, exit 0, source `519001c`. |
| **C2** `runs/frozen_history_baselines_20261126/` | `history/report.md`, `history/results.json`. Exact test metrics, paired episode-bootstrap intervals, parameter counts, time baseline, source archive digest, and attention gate. | `completion_manifest.json`, `history/execution_manifest.json`, `launch_command`, and `supervisor_20261126T212114480865Z/job_001/console.log`. Session `dreamer-frozen-memory-studies`, Python worker 613898, exit 0, source `c0ca14b`. |
| **C3** `runs/cage2_final_history_20261126/` | `cage2-final/report.md`, `cage2-final/results.json`, `cage2-final/fit_aligned.npz`, and `cage2-final/validation_aligned.npz`. Exact matching keys, eligibility, elapsed/recency counts, one fit pair, feature ties and unopened test are recorded. No locked probes or test arrays were produced. | Root `completion_manifest.json`, `cage2-final/execution_manifest.json`, `launch_command`, and `supervisor_20261126T213737099902Z/job_001/console.log`. Session `dreamer-cage2-final-history`, Python worker 624480, exit 0, source `5219eb6`, approved plan `0d04596`. Initial filename error in root console stopped before a worker or any simulator steps. |

Root completion manifests also record the supervisor and launcher process IDs, commands, watchdog logs and exit states. Per-worker execution manifests identify the Python PID. The UV launcher PID may differ from its Python child. See the [figure inventory](figures.md) for all editable SVG and PNG paths.

## Frozen checkpoint and raw input anchors

| Family | Preserved checkpoint or dataset paths | Provenance and reuse |
|---|---|---|
| T-maze baseline seeds 19, 23, 29 | `runs/minihack_tmaze_memory_20261126T140604Z/comparison/job_00{1,2,3}/results/flat_seed{19,23,29}/` with the job and seed numbers paired in order. Each has `initial.pt`, `update_{200,400,…,4000}.pt`, `latest.pt`, configuration, `complete.json`, `memory_report.json`, `heldout_trajectories.json`, and historical `trajectories_*.json`. | T1 evaluated initial, 19 interval checkpoints and final per seed plus two smoke checkpoints. For each seed the numbered final and `latest.pt` world states were checked by the original diagnostic. Historical figure and original stage names remain preserved in the comparison root. |
| T-maze smoke | `runs/minihack_tmaze_verification_20261126T135143Z/smoke/latest.pt` (update 12) and `runs/minihack_tmaze_preflight_20261126T140032Z/supervisor/job_001/results/flat_seed19/latest.pt` (update 2). | T1 records `smoke_seed19_12.json` and `smoke_seed19_2.json` pin hashes. The smoke runs have different training histories from the three baseline runs. |
| Original CAGE2 baseline | `runs/cage2_blue_cycle_seed0_20261123/latest.pt` is the original update-1 checkpoint. `runs/cage2_blue_learning_seed0_20261123/` holds updates 26, 4244, 8462, 12681 and 16899 plus `latest.pt`, action mapping, resolved configuration, periodic evaluation and scorecard. | [Original baseline manifest](../../environments/cage2/reproducibility/original_baseline.json) and tag `cage2_baseline_v1` at `1adc2a0`. The independent backup location is user-confirmed. Remote bytes remain unverified. |
| CAGE2 host-aware flat pilot | `runs/cage2_host_aware_pilot_seed0_20261124/` contains updates 1, 4244, 8462, 12681, 16899 and latest. `runs/cage2_latent_analysis_host_pilot_20261124/` contains `frozen_features.npz`, episode seed metadata, split, original numerical results and exact settings. | Archive SHA-256 `4a8e1ce2d7a1177f135f66ce8b7b9e977e9785a23fe979b977a934c4fb89b6f5`. 360 episodes, 10,800 decision rows. This is distinct from the matched flat seed-0 checkpoint. Its 200-dimensional h, 32×30 z/probabilities, observations, actions, rewards and returns support C2. Encoder activations and privileged labels are absent. |
| CAGE2 matched flat and historical hierarchy | `runs/cage2_flat_replication_20261126T033743Z/group/job_00{1,2,3}/results/cage2_host_seed{0,1,2}/` and `runs/hierarchy_v3_20261125T153127Z/group/job_00{1,2,3}/results/cage2_host_seed{0,1,2}/`. These families preserve finals and numbered saves at 1, 4244, 8462, 12681 and 16899. | `runs/cage2_frozen_matched_20261126T133406Z/provenance.json` records all six checkpoint fingerprints, opponents and seed alignment. The matched flat `flat_seed0_b-line.jsonl` supplies 100 episodes for C1/C3. Hierarchical archived evaluations are referenced by this provenance under `runs/cage2_frozen_opponents_20261125T213227Z/`; no new hierarchy analysis ran in closeout. |

### Independently checked at closeout

| Weight | SHA-256 |
|---|---|
| Original CAGE2 update 1 | `94f378899c0422e0af2db2e07cbb13eab9b27a1448cfbe2deb955617c0e5d2ff` |
| Matched flat CAGE2 seed 0, update 16899 | `db5c5e65ddda372b0e32f0af38f0cefb0698d722fe6156e4052735d19b452221` |
| T-maze seed 19 final | `6bd3fa64d101a755099ecb16a54ccdc39d7cb8ed162db634a585f9b0ccecc2c7` |
| T-maze seed 23 final | `48b844693d9c62c4681fff141951e2e662cf7baa292dc6785975bd9e094f94d3` |
| T-maze seed 29 final | `96eb35e01302e7bcf62369b14b212ad09660af1c5504b1f91fda5834f84351b3` |

C3 also matched all four SHA-256 entries in [the approved plan](../../environments/cage2/history_common_support_plan.json) against the stored pilot results, pilot labels, matched provenance and action archive. Other checkpoint fingerprints remain recorded provenance. The local baseline tag object is `bab6727939e0d237de5566fa782e89f73c82b5aa`.

## Versioned source and execution profiles

| Purpose | Tracked path or revision |
|---|---|
| T-maze model/training and sealed manifest | `4746126` and `3def249` in the research source and manifest worktrees. T1 CLI, JSON, report and shell launcher under `environments/minihack/` at `e228ab6`. The manifest's root revision predates that source commit and contains explicit source hashes. |
| Assessment and modular implementation | `76b34d1` architecture assessment, `63b1d84` shared diagnostic core, `519001c` intervention and labeled pilot, `ccb35f1` separated adapters and statistical utilities, `c0ca14b` temporal probes. |
| C3 plan and runner | `0d04596` approved plan; `c332948` matching, locked probe, and study/report implementation; `5219eb6` corrected plan binding. [Study plan](../cage2_history_followup_plan.md) and [machine-readable constraints](../../environments/cage2/history_common_support_plan.json). |
| Resolved study configs | `environments/minihack/tmaze_bottleneck.json`, `environments/minihack/tmaze_intervention.json`, `environments/cage2/cage2_memory_pilot.json`, `environments/cage2/archived_history_baselines.json`, `environments/cage2/history_common_support_plan.json`. |
| Reusable source and tests | `dreamerrl/eval/memory/`, tests `test_memory_diagnostics.py`, `test_history_probes.py`, `test_common_support.py`, and historical `environments/cage2/test_latent_probe.py`. [Framework documentation](../frozen_memory_diagnostics.md) and [adapter assessment](../memory_diagnostics_architecture.md). |

Execution manifests record Python 3.10.19, Torch 2.9.1+cu130, NumPy 1.26.4 and Matplotlib 3.10.9. The root lock hash is `7bb7f784c6b02c70f640052e5b3348f8f043089d4ee1f98e252650a01faefa38`. MiniHack uses Gymnasium 1.2.0, MiniHack 1.0.2 and NLE 1.3.0. CAGE2 uses Gymnasium 1.4.0 and the pinned CybORG revision `26ce1c1253fa9e2e73f25e6a7f2da32860c11257`. Historical baseline reproduction retains its own tag and lock. No frozen output is required in Git.
