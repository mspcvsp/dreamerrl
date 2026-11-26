# MiniHack T-maze preflight verification

**GO for the proposed three-seed baseline, pending approval of this report.** The preflight validated the task, the frozen diagnostics, and the supervised launch path with a two-update smoke worker. No full training was launched.

## 1. Environment validation

| Criterion | Result | Evidence |
|---|---|---|
| Balanced randomized cue assignments | **PASS** | Reset draws independently shuffled left/right pairs. The seeded 16-episode test obtained eight episodes per cue and reproduced the same order and observations on reseed. |
| Hidden cue and matched observations | **PASS** | Opposite explicit cues on the same native seed differed in one reset cell. Subsequent glyph arrays were byte-identical for the same trajectory through the decision. The eight fixed stem moves had identical observations even when the exposed stem actions differed. Cue labels are absent from step info. |
| Both branch rewards and termination | **PASS** | Both choices were tested with both cues. Reward is zero and episodes continue along the stem. Correct choice gives +1, incorrect choice gives −1, and the choice terminates the episode. |
| Random and oracle baselines | **PASS** | On 256 level seeds with both cue assignments, independently sampled random choices scored **50.0%** and the cue-aware oracle scored **100.0%**, with 512 episodes per baseline. |

[Environment tests](/home/mspcvsp/Documents/dreamerrl/.experiments/minihack-tmaze-memory/dreamerrl/tests/functional/test_minihack_tmaze.py) · [Test results](runs/minihack_tmaze_preflight_20261126T140032Z/tests.log) · [Baseline record](runs/minihack_tmaze_preflight_20261126T140032Z/baselines.json)

## 2. Memory probe validation

| Criterion | Result | Evidence |
|---|---|---|
| Episode- and level-disjoint split | **PASS** | The 80 preflight levels split into 48 fitting, 16 validation, and 16 test levels. Both opposite-cue episodes from a native level stay in the same split. Each split is cue-balanced. |
| Prevent label, seed, and future-observation leakage | **PASS** | Inputs are only the representation at the logged lag after processing the current prefix. Labels are probe targets. Level seeds are grouping keys. Feature moments fit on the fitting fold, ridge regularization is selected on validation, and accuracy is scored on held-out levels. The post-cue observation-embedding probe remained at 50% at every logged lag. |
| Observation embedding, h, z, and concatenated h/z | **PASS** | The frozen smoke checkpoint produced all 24 feature-by-lag probe scores. Lags 0, 1, 2, 4, 6, and 8 were checked against the simulator's elapsed step field. |
| Majority and shuffled-label controls | **PASS** | Majority-class accuracy was 50% throughout. The fixed shuffled-label control permutes fitting and validation targets within their own folds and evaluates against true test targets. At lag 8 it scored 50% using the observation embedding and 0% for the latent probes. This single permutation is a negative control and does not estimate a null distribution. |
| Frozen input model | **PASS** | The diagnostics check that both world and actor parameters have gradients disabled. Feature collection uses inference mode. Only the offline ridge probe is fitted. The source checkpoint SHA-256 is recorded in the report. |

[Probe table, split IDs, and checkpoint fingerprint](runs/minihack_tmaze_preflight_20261126T140032Z/memory_report.json) · [Raw preflight trajectories](runs/minihack_tmaze_preflight_20261126T140032Z/heldout_trajectories.json)

## 3. History-swap validation

| Criterion | Result | Evidence |
|---|---|---|
| Matched opposite cue histories | **PASS** | Each counterfactual pair holds the native level seed, all post-reset observations through the junction, and every predecision action fixed. Only the initial cue changes. |
| Initial RSSM state and stochastic sampling | **PASS** | Each trajectory initializes the RSSM with the same zero previous action and first-observation reset flag. Diagnostic filtering and imagined successors use deterministic latent modes. The identical-history reruns verified repeatability. |
| Separate prediction and decision metrics | **PASS** | The report distinguishes change in right-action probability, change in the predicted right-minus-left reward margin, greedily selected branch, and observed accuracy. Prediction margins evaluate both hypothetical branch actions from each history. |
| Identical-history, neutral cue, and irrelevant-observation controls | **PASS** | Identical histories and cue-neutralized matched histories gave zero output changes. A cue-independent marker in a reserved cell changed only that cell at reset. Its maximum absolute changes were 0.00587 in right-action probability and 0.03960 in predicted reward margin. The marker test measures background sensitivity and is not assumed to give zero. |
| Interpretation | **PASS** | The intervention demonstrates a sensitivity measurement with known matched inputs. It is not a causal identification result. |

For this smoke checkpoint, the actual cue swap moved right-action probability by **−0.00649** on average and shifted the predicted reward margin by **+0.04172**. Greedy choice did not switch and both factual and history-swapped forced-stem accuracy were **50%**. The lag-8 latent probes decoded the cue on this small test fold, but the policy showed no useful branch selection. The irrelevant-marker response is of similar magnitude to the smoke checkpoint’s cue response, so these checks validate the method and leave cue-specific model use unestablished. [Per-level sensitivity and controls](runs/minihack_tmaze_preflight_20261126T140032Z/memory_report.json)

## 4. Training readiness

| Criterion | Result | Evidence |
|---|---|---|
| CUDA smoke and regressions | **PASS** | The original CUDA smoke completed 12 updates and 864 interactions. The current preflight passed 24 environment, MiniHack cycle, supervisor, and thermal watchdog tests. |
| Independent seeds and fixed budgets | **PASS** | The proposed configuration fixes seeds 19, 23, and 29, identical hyperparameters, 4,000 updates per worker, four vector environments, and 18 collection steps per environment per update. This gives 288,000 environment interactions per seed. |
| Checkpointing, logging, thermal watchdog, and output paths | **PASS** | A bounded worker ran two updates and 144 interactions through the existing supervisor with the real watchdog. It exited 0 and produced numbered checkpoints, latest checkpoint, metrics, raw trajectories, telemetry, and the consolidated JSON and retention plot under main runs. |
| Existing learning algorithm | **PASS** | The learner and model files have no changes in the research worktree. The MiniHack CLI change exposes its existing update hook to the study runner for checkpointing and evaluation. The formal configuration uses flat FP32 DreamerV3 without hierarchy or auxiliary losses. |
| Full-run resource estimate | **PASS** | The original single-worker smoke took a median 0.192 s/update and recorded about 113 MiB of CUDA tensor allocation. That excludes process and driver overhead. At three workers, plan for **30–50 minutes** with evaluation and I/O, with a 60-minute safety cutoff. A long three-worker benchmark has not been run. |

The launch checks require one CUDA GPU, 36 GiB available memory for the three workers, and at least 20 GiB free disk. The observed host had about 113 GiB available and 3.3 TiB disk free. A smoke checkpoint was about 36 MiB. Larger replay buffers will increase final checkpoint sizes. [Resource inputs](runs/minihack_tmaze_preflight_20261126T140032Z/resource_estimate_inputs.json) · [Original CUDA smoke log](runs/minihack_tmaze_verification_20261126T135143Z/smoke.log) · [Supervisor smoke summary](runs/minihack_tmaze_preflight_20261126T140032Z/supervisor/execution_summary.json) · [Generated smoke report](runs/minihack_tmaze_preflight_20261126T140032Z/supervisor/final_report.md)

## Limitations and gate

- The forced route and fixed eight-step decision delay isolate memory use in this environment. The lag curve measures retention within that interval. Generalization to variable-length routes, navigation, or sampled latent evaluation remains open.
- Level disjointness ensures that the same native seed never spans fit, validation, and test. All levels use the same T layout, so this does not measure geometry generalization.
- The preflight diagnostic levels were **782200–782279**. The formal frozen diagnostic now reserves **783100–783179** to keep its test levels unseen during preflight. The training and development budgets are unchanged.
- Use all completed training seeds. The proposed per-seed targets are 80% natural branch accuracy, 75% forced-stem accuracy, and 75% decision-time h-probe accuracy. Keep observation-only decoding near chance and report policy and reward-model sensitivity separately. Preflight validates measurement, not attainment of these full-training targets.
- **GO for baseline training after explicit approval.** Commit the reviewed source and seal the execution manifest before launch. Do not proceed from the smoke manifest. [Proposed configuration and launch commands](/home/mspcvsp/Documents/dreamerrl/.experiments/minihack-tmaze-memory/environments/minihack/TMAZE_MEMORY.md)
