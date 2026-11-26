# DreamerV3 experimental closeout

**Campaign closed on November 26, 2026.** The final CAGE2 study executed successfully and stopped at the approved development common-support gate. Broad CAGE2 representation evaluation remains **NO-GO**. Future research is outside this closeout.

[Artifact inventory](artifacts.md) · [Evidence-to-claim matrix](claims.md) · [Figure inventory](figures.md) · [Technical report outline](outline.md)

## Executive synthesis

**T-maze.** The earliest demonstrated bottleneck in the three studied flat DreamerV3 runs was conversion of a weak, consistently decodable posterior signal into a single hard categorical state. With the cue rendered at reset, held-out linear probes gave 100% accuracy from raw and preprocessed observations, encoder embeddings, raw posterior logits, and probabilities before and after unimix. Every matched cue pair had the same argmax. Sampled-state linear probes were at 50% and RBF accuracy averaged 50.04%. Equal-prior single-draw Bayes upper bounds were 50.07–50.18% under the stated factorization. These results apply to the tested seeds, cue construction, held-out levels and histories.

**T-maze intervention.** Fit-selected reset states transmitted an artificial distinction through the recurrence. It was still 100% linearly decodable at the junction with separate factual and swapped probes, despite attenuation to 0.55–1.18% of the lag-1 paired recurrent L2. Greedy branch accuracy was 50% under factual, swapped, and neutral conditions. The measured branch-policy probability shifts were tiny, and prior-mode predicted reward margins did not reverse with the cue. These local results provide evidence of weak task-aligned readout for the injected states. They leave the prior transition and reward head unresolved as separate mechanisms. Near-equal critic values do not by themselves diagnose critic failure, because the two cues can have the same optimal value. The injected states were selected from tails of fit-level distributions and can be rare under ordinary inference.

**CAGE2.** The 24-episode labeled pilot reproduced the 720 archived decisions, rewards, available reward components and episode boundaries for one matched flat checkpoint. In five test episodes, the selected Enterprise2 active-session target was already 98.08% balanced-accuracy decodable from current Blue observations. The posterior and argmax scored 100%, while sampled z scored 95.98%. Adding recurrent state improved the observation-plus-time baseline by 1.92 points with a paired 95% interval of 0–4.35. The final study used a separate recent-session target that required exact matches. It could not form enough independent pairs, so there is no confirmatory estimate of historical recoverability. The T-maze collapse does not transfer to the tested CAGE2 target.

## Final CAGE2 disposition

| Gate | Result |
|---|---|
| Execution | Exit 0. Replayed 40 development episodes and 1,200 native simulator steps. Archived greedy actions, rewards, available components and boundaries matched under the frozen checkpoint and pinned simulator. |
| Fit common support | 512 eligible frames, 3 with any exact opposite-label match in another episode, 1 selected independent pair. Minimum 10 pairs. |
| Validation common support | 219 eligible frames and 0 matchable frames or selected pairs. Minimum 4 pairs. |
| Qualifying pair | Elapsed frame 29. The last active session was 4 frames earlier for the positive and 5 frames earlier for the negative. All 52 current Blue bits, elapsed time and incoming action matched. |
| Tie controls | Current-input and fixed-carry features tied exactly on the selected fit pair. There were no fitted probe scores. |
| Confirmatory test | No probes were fitted. The reserved 36 test episodes stayed unopened. |

The gate addressed **experimental identifiability**. Under this eligibility rule, archive and matching definition, there was too little independent overlap to measure historical recoverability. The outcome supplies no estimate of recurrent-memory performance on the proposed matched target. The report and JSON preserve elapsed-time and recency coverage as descriptive evidence.

## Archive-only history baselines

The archived return-to-go study completed with exit 0. It used the fixed 216/72/72 episode split from 360 host-aware flat-pilot episodes, a common time covariate, a fit-only seeded projection, and validation-selected learning rate and epoch. It compared a current linear probe and near-budget-matched current/five-frame MLPs (4,096 and 4,009 trainable parameters). Test episodes were the units for paired bootstrap uncertainty.

| Feature source | Current linear test R² | Current MLP | Five-frame MLP | Paired fixed minus current ΔR² (95% interval) |
|---|---:|---:|---:|---|
| Raw observation | 0.60461 | 0.74173 | 0.75526 | +0.01352 [0.00302, 0.02329] |
| Posterior probabilities | 0.61563 | 0.75549 | 0.73900 | −0.01649 [−0.04786, 0.01314] |
| Combined RSSM | 0.64165 | 0.73436 | 0.73847 | +0.00411 [−0.00216, 0.01144] |

Time-only lookup had test R² **0.59394**. There was a small positive increment for raw-observation history in this archive and no resolved increment for posterior or combined-state histories. These are descriptive, policy-conditional return forecasts. The archive lacks encoder embeddings and offline hidden-session labels, and the comparisons do not establish policy use, a need for attention, or a compromise-memory gain. No GRU or transformer probe was fitted.

## Readouts, behavior, and credit assignment

- **Reward and policy.** In T-maze the true right-minus-left immediate branch reward contrast across cues is +4. Across the injected-state conditions, the factual prior-mode margin changed by at most 7.39×10⁻⁶ and kept the same sign for both cues. The largest changes were 1.67×10⁻⁶ in right probability, 3.05×10⁻⁴ in probability-derived right-minus-left logit margin, and 1.19×10⁻⁶ in online critic value. Raw actor logits were unavailable. Identical-history and the specified irrelevant-marker reruns matched intact junction outputs exactly. The evidence concerns the tested states and one-step prior-mode prediction.
- **CAGE2 utilization and competence.** Replacing the carry while holding the current observation and incoming action fixed changed greedy choices on 17.33% of held-out pilot frames. Actor total variation averaged 0.1767 and absolute value change averaged 9.50. This shows carry sensitivity on these trajectories. It does not identify what the policy used or how the intervention would change simulator outcomes. Historical behavior was reproduced for the checked subset. No new competence estimate was made for the broader cohort.
- **Temporal gradients.** T-maze had ten replay rows and eight supervised steps. The only window supervising terminal branch reward received reset through detached burn-in, with no direct within-window gradient to reset encoding. CAGE2 replay had 6,451 complete 31-row episodes, 16 supervised rows and eight burn-in rows. A final-row loss could reach no reset input in those windows. An earlier late nonzero reward could share a window with reset in 6,451 of 103,216 eligible starts. A structural path permits a gradient and leaves its magnitude unresolved. Regression tests verified detached-prefix behavior on a small diagnostic graph.
- **Reconstruction.** T-maze reset reconstruction was 99.1% across cells and 50% at the cue. In the CAGE2 pilot, aggregate BlueTable bit accuracy was 98.03%, exact visible compromise-field accuracy was 93.13%, and exact accuracy on positive compromise fields was 53.59%. BlueTable measures defender-visible fields. Replay truncation, cue weighting and learning-objective choices remain plausible mechanisms in T-maze.

## Reproducibility and software status

The evidence inventory records source revisions, profile and simulator pins, checkpoint and configuration identities, worker logs and execution manifests. Closeout independently hash-checked the three final T-maze weights, the final CAGE2 flat weight and four pinned input artifacts, and the original CAGE2 update-1 weight. The recorded independent backup location remains user-confirmed with remote bytes unverified. Frozen artifacts and the immutable baseline tag were preserved.

The reusable platform is tracked in the local `study/cage2-final-history` branch. It includes environment and model adapters, categorical statistics and explicit Bayes assumptions, replay and local input-gradient tools, fixed and causal temporal probes with an attention gate, exact matching, and separate visualization and artifact writing utilities. The active main checkout remains at `4101a9c`. The pre-closeout source comparison against main found no model, learner or lockfile differences. Generated evidence remains ignored. No remote publication or merge was performed.

Closeout verification ran **23 targeted tests** in the locked development profile and **one MiniHack integration test** in the locked legacy profile. Coverage includes archived numerical parity, stage/readout parity, causal prefixes without cross-episode leakage, parameter budgets and attention gate checks, posterior probability assumptions, replay gradient connectivity, grouped probing, deterministic matching, probe locking, tied controls, and exact binomial thresholds. This was a focused test pass rather than a repository-wide regression suite. No additional experiment or RL training was run.

## Implementation limitations and future research

The model adapter handles flat v4 checkpoints. Historical hierarchical artifacts remain preserved for context. The final study's test and probe branches were exercised with synthetic tests but remained unopened on the archived cohort. The CAGE2 pilot did not persist aligned observations with all privileged labels. Raw actor logits and branch-wise immediate predictions cannot be reconstructed in full from existing T-maze saved margins. Historical absolute paths refer to the original host layout and may require recorded path remapping. Resume and supervisor records distinguish launcher and Python worker PIDs. Most probe uncertainty is conditional on the fitted readout and studied seeds or episodes.

Potential future work includes collecting CAGE2 trajectories with independently supported matched labels, estimating the value of recurrent history beyond observation and time on that supported population, separating the T-maze prior transition from the reward head, and comparing a soft-posterior bypass or revised training objective under a controlled design. These require new authorization and are outside project completion. The current evidence leaves attention after the RSSM unmotivated for either a collapsed T-maze single-draw signal or an unidentified CAGE2 history effect.
