# Technical report outline

**Working title:** Stagewise information loss and historical recoverability in frozen DreamerV3 agents

Use the [claims matrix](claims.md) for claim IDs and the [artifact inventory](artifacts.md) for evidence. The report should lead with the frozen diagnostics and use behavioral scores as a separate result.

## Abstract

- Study question and scope. Distinguish encoding, transmission, retention, and use.
- T-maze categorical bottleneck across three seeds and local downstream readout results after reset injection.
- CAGE2 evidence boundaries. Archived behavior parity, a bounded selected-host pilot, archival history baselines, and final common-support failure.
- Conclusion. Architecture-wide generalization and a confirmatory CAGE2 history estimate remain outside the evidence.

## 1. Experimental systems and frozen provenance

- Describe T-maze cue timing, matched levels, forced eight-step stem, branch and terminal reward, checkpoint schedule, and smoke models.
- Describe the CAGE2 serial Blue observation/action schema, scenario, opponent and native cap. Explain terminal versus truncation.
- Distinguish original baseline, host-aware pilot archive and matched flat checkpoint. Explain historical hierarchy archives and their exclusion from this evaluation.
- Table of checkpoints, environment and source revisions, model identities, frozen weights, run IDs, and sample units. Cite T1–T2 and C1–C3.

## 2. Diagnostic methods

- Stage diagram from current observation through encoder, raw logits, pre/post-unimix probabilities, hard state, and combined recurrent state. Define the post-transition reset h and current posterior z.
- Matched opposite cues, level/episode grouping, synthetic linear and XOR checks, label-swap null, identical/neutral/marker controls and sampling coupling. Define what each tests.
- Linear and RBF probes, fit-only preprocessing, validation selection, grouped uncertainty and conditional single-draw Bayes bounds.
- Frozen reset-state injection and downstream measurements. State selection on fit levels, separate probes across intervention conditions, retained recurrence, reconstructed binary log odds, online critic, prior-mode branch reward and observed action.
- CAGE2 archive parity and privileged-label isolation. Distinguish visible reconstruction from active-session truth. Describe fixed-carry controls.
- Archive-only temporal analysis. Return-to-go, three available feature families, time lookup, fit-only projection, fixed history length, comparable MLP budgets, paired episode intervals and attention gate.
- Predeclared CAGE2 common-support study. Recent session conditional on inactivity, exact present-input key, episode-disjoint matching, independent pair minima, locked model and unopened test criteria.
- Replay audit. Define supervised coverage, detached burn-in, actual local gradient checks and the limits of the graph analysis.

## 3. Results

- **Encoding and transmission** (T1-E/Z): lead with stage retention figure and a table of single-draw bounds, paired posterior differences, argmax equality and probes.
- **Temporal retention** (T1-T, T2-T): checkpoint onset resolution, smoke discrepancies, and injected-state attenuation at junction.
- **Reward prediction and utilization** (T2-R/A/B): separate projected reward margins, actor response, critic scalar, controls, and branch behavior. Report precision and cue-symmetry limits.
- **CAGE2 compatibility and pilot** (C1-B/E/Z/T/A/G): archived parity, selected-host held-out scores, visible reconstruction, history increment and carry sensitivity. Indicate sample sizes on all tables.
- **CAGE2 archive-only return baselines** (C2-T): absolute and paired ΔR², including time baseline and missing features.
- **CAGE2 closeout gate** (C3-I): eligible, matchable and disjoint-pair counts, matched age and elapsed time, tied features and unopened cohorts. Make the absence of an effect estimate explicit.
- **Gradient path and reconstruction fidelity** (T1-G, C1-G): show window diagrams and performance on cue/positive fields alongside aggregate accuracy. Keep mechanisms separate from confirmed structural results.

## 4. Discussion and limitations

- Distinguish an exact categorical collision, weak stochastic signal, and local downstream under-response in these T-maze checkpoints.
- Discuss why CAGE2 carry sensitivity alone does not prove use of privileged compromise information.
- Discuss limited geometry and seed variation, test-set sizes, intervention state support, archive target confounds, precision and prior mode, and conditional bootstrap intervals.
- Treat lack of common support as an identifiability finding scoped to the archived policy, target and matching definition.
- State that no new experiment or architecture change is a completion requirement. List separately authorized future options with a decision criterion for each.

## Supplement and reproducibility

1. Complete artifact and Git provenance tables, including baseline preservation and hashes.
2. Per-seed and checkpoint-level T-maze results, probe controls, factor utilization, and burn-in audit.
3. CAGE2 archive alignment, visible target coding, phase-specific coverage and ignored raw array paths.
4. Models, seed grouping, exact statistical procedure, held-out gate details and test coverage.
5. Source tree, locked dependency profiles, supervisor/thermal execution manifests, and editorial figure inventory.
