# Frozen memory diagnostics

The shared implementation lives under [dreamerrl/eval/memory](../dreamerrl/eval/memory/). The [architecture assessment](memory_diagnostics_architecture.md) describes the adapter boundaries and frozen checkpoint inventory. Historical T-maze outputs and CAGE2 archive outputs are read-only inputs. Preserve their original source and use a new output directory for each study definition.

## Adapter use

An environment adapter returns `Episode` records with one observation and one outgoing action/reward per decision frame. Keep simulator labels in a separate `Labels` record keyed by episode, aligned to the pre-action observation. Use a group key shared by related histories. An environment adapter supplies the model-ready input and verifies observation and action schemas. Implement an explicit intervention only when the test can validate the invariant portion of a matched history. The T-maze adapter validates saved cue, identical, neutral, and marker histories. The CAGE2 adapter replays saved Blue actions and takes read-only simulator snapshots before each action.

The flat Dreamer model adapter validates a checkpoint against a trusted local SHA-256 and budget, loads the state dictionaries strictly, and exposes the actual encoder and recurrent filter. Available stages include raw head logits, pre- and post-unimix probabilities, argmax or sampled hard state, deterministic recurrence and their concatenation. Its prior rollout and actor/online critic readouts use the same model. Other model adapters can implement the protocol with their own state type. Hierarchical policies will require explicit policy carry.

Group by level or episode before fitting. The shared linear/RBF classifier uses fit-only scaling, validation-only selection and fixed held-out predictions. The migrated CAGE2 ridge retains its original scaling and regression outputs. The legacy CAGE2 import path forwards to the shared implementation. Use paired episode resampling and preserve positive/negative coverage with reported uncertainty. CAGE2’s bounded validation describes one checkpoint and one opponent.

## Studies

- **T-maze reset-state intervention.** Reuse the accepted artifact split. Rank fit-level hard posterior draws by a fit-only log-likelihood ratio and select two reset states. Inject them into factual and swapped histories. Compare to cue-neutral, intact, identical and irrelevant-marker controls. Actor probabilities, branch reward margins, critic values and h separation quantify the intervention. The reset states may be rare under the natural posterior, and this study leaves causes of their natural collapse unresolved.
- **CAGE2 archive and labeled replay.** First verify the saved pilot NPZ and reproduce its historical observation-only ridge metrics. Then load one matched flat checkpoint and replay 24 archived B-line episodes. Verify greedy action agreement, reward/component parity and horizon semantics. Privileged labels describe active Red sessions in the pinned simulator. Visible-evidence decoding and observation reconstruction use BlueTable fields. The history-truncated control holds current and incoming observations/actions to the same recorded decisions, with zero recurrent carry at each current frame. Probe confidence depends on held-out host support. This pilot does not authorize broader evaluation.

## Reproduction

Use the repository’s locked uv profiles and the pinned CybORG vendor. Run from a clean, committed source worktree using the main checkout’s provisioned environments and artifacts.

```bash
bash scripts/run_frozen_memory_studies.sh /absolute/main/runs/frozen_memory_studies
```

The launcher uses detached tmux session `dreamer-frozen-memory-studies`, sequential CPU workers, and the existing supervisor and watchdog with the original thresholds. Study configs are versioned beside the adapters. Execution manifests, worker logs, exit codes, checkpoint identifiers, hashed inputs and figures are written under the requested ignored output directory. A T-maze restart reuses complete per-seed records from the same revision. A CAGE2 restart verifies an existing complete record or reruns its bounded validation. The exit handler writes the combined report and completion manifest.

For unit and integration tests, run `dreamerrl/tests/functional/test_memory_diagnostics.py` in the locked dev profile and rerun its MiniHack-marked tests in the legacy profile. Set `DREAMER_MEMORY_ARTIFACT_ROOT` to the main checkout for optional numerical parity against the accepted T-maze checkpoint. CAGE2 archive numerical parity is also the pilot’s first execution gate.
