# Beyond Reconstruction: What Does a World Model Actually Remember?

*On inspecting information flow in DreamerV3, from simulated cyber defense to a one-clue maze.*

**Cover** · `cover.png`

**Alt text** · Three stages show a clue entering a world model, remaining detectable in its probabilities, and narrowing at the hard categorical state.

A world model can reconstruct a scene convincingly and still lose the one detail that determines what the agent should do next. I encountered that gap while investigating DreamerV3 for simulated cyber defense. It changed how I read a model dashboard. A high observation-reconstruction score can tell me that the model captures much of what it sees. To understand a decision, I also need to know which information survives the path from observation to action.

The result was especially clear in a deliberately simple maze. A cue was perfectly decodable from the model’s encoder and its posterior probabilities. When the model converted those probabilities to the hard categorical state used by its recurrence and policy, the matched cues produced identical argmax states. Sampling offered almost no usable cue signal either. That result lets me identify a specific bottleneck in the tested checkpoints and ask a second question: what happens if I supply the frozen model with a distinction it ordinarily loses?

## Why use a world model for cyber defense?

A defender rarely sees the full state of a network. It receives observations such as alerts or service status, takes actions, and learns about their consequences later. Yesterday’s clue can matter after it disappears from today’s view. Simulated environments let me study those dependencies without touching an operational network or interpreting a benchmark reward as evidence of real-world readiness.

[TTCP CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2) provides a simulated network and a Blue defensive agent. I used the environment to examine both a flat policy, which selects each action directly, and a hierarchical policy with a slower high-level controller and a lower-level action policy. My implementation follows the core design in Hafner and colleagues’ [DreamerV3 paper](https://www.nature.com/articles/s41586-025-08744-2). An encoder handles the current observation. A recurrent state carries history. A categorical latent provides a compact description of the present. The agent uses a learned world model to train its policy and value estimator on imagined trajectories.

**Figure 1 · `architecture.png`.** Conceptual architecture. Experience updates the world model. Predicted trajectories provide training data for the actor and critic.

**Alt text:** Four stages connect observations and past actions, the encoder and recurrent state, imagined outcomes, and actor–critic training.

The archived CAGE2 comparison used three training seeds per architecture and 100 paired simulator seeds per checkpoint and opponent at a 30-step horizon. On B-line, mean return was −34.94 for the flat cohort and −35.47 for the hierarchical cohort. On Meander, it was −27.27 and −26.47, respectively. Both cohorts returned zero against Sleep. Higher return is better. The direction of the small average difference changed with the opponent, and the training-seed differences varied substantially. I treat those results as a local comparison of frozen agents under one protocol.

**Figure 2 · `behavior.png`.** Archived CAGE2 mean Blue return. Points show the three training-seed means per architecture, each evaluated on the same 100 simulator seeds.

**Alt text:** The flat and hierarchical return points overlap substantially against B-line and Meander. Both architectures score zero against Sleep.

## The maze isolates a question the network cannot

I used [MiniHack](https://arxiv.org/abs/2109.13202) to construct a T-maze with one meaningful cue. A sign at reset identifies the rewarded branch. After the sign disappears, the two cue conditions share the same forced eight-step stem. The layout, actions and later observations are identical within a pair. The branch reward depends on that original clue. This design makes it possible to follow the cue through the model without mistaking differences in later observations for memory.

**Figure 3 · `maze.png`.** Conceptual T-maze. The reset cue determines the rewarded branch. The subsequent stem is identical under both assignments.

**Alt text:** A T-shaped corridor starts with a temporary clue. Eight shared moves lead up the stem to a left-or-right branch.

I evaluated separate checkpoints from three training seeds and kept related level pairs together in the data split. A probe is a deliberately limited reader trained on frozen features. It tells me what that reader can recover on held-out examples. A successful probe can establish that information is available to that reader. It takes a separate intervention to learn whether the acting policy uses it. This is the same distinction that makes [probing methodology](https://aclanthology.org/2022.cl-1.7/) valuable and easy to overinterpret.

## The cue survived encoding and disappeared at the hard state

At reset, raw and preprocessed input, encoder features, raw posterior logits, and probabilities before and after uniform mixing all gave 100% linear probe accuracy. The posterior probability differences were small but systematic across the held-out pairs. After argmax, every tested opposite-cue pair had exactly the same state. A probe on sampled hard states also scored around 50%, and the nonlinear probe did not materially improve on chance.

Chance-level probing alone would be a weak basis for a claim about all possible readers. Here I could also ask what an ideal reader could recover from one categorical draw, given the paired context and equally likely cues. Under the model’s factorized posterior, the measured distributions imply upper bounds of **50.07–50.18%** accuracy across the three final checkpoints. Those bounds apply to one draw under the tested conditions. A probe reading the full probabilities and an agent receiving one hard draw have access to different evidence.

**Figure 4 · `bottleneck.png`.** Reset cue recovery before and after hard-state conversion. The range beside the sampled state is the single-draw Bayes upper bound across seeds.

**Alt text:** Encoder and posterior rows read 100%. Argmax and sampled hard-state rows read 50%. An annotation gives ideal single-draw upper bounds from 50.07% to 50.18%.

Ordinary recurrent and combined states were at chance at the junction. That result is consistent with where the information was lost. The later matched observations could offer no new clue. It also explains a common source of confusion in recurrent models. The deterministic state at reset is formed before seeing that frame’s cue. The current posterior is the first latent route by which the cue can enter the following recurrence.

## I restored a difference and checked the decisions

I next took two distinct hard states selected using only the development levels and injected them at reset. The rest of the trajectory was held fixed. By the junction, separate probes decoded the cue assignment perfectly in the factual and swapped conditions. The differences had attenuated to roughly 0.6–1.2% of their first-step recurrent separation, yet a reader could still distinguish them. Greedy decisions stayed at 50% across factual, swapped and cue-neutral conditions.

I also checked responses that an unchanged greedy choice could hide. The maximum shift in right-action probability was about 0.0000017. Prior-mode predicted branch reward margins stayed on the same side even though the correct rewarded branch reverses with the cue. The scalar online critic barely changed, a result that is harder to interpret because either cue can offer the same optimal value. These were selected artificial states, sometimes rare under normal inference. The intervention shows a local failure to produce a useful response under the tested histories. It cannot isolate the prior transition from the reward head or establish how the actor responds to every possible state.

**Figure 5 · `intervention.png`.** Selected reset states restored junction-state decodability. Greedy branch accuracy remained at chance.

**Alt text:** Two labeled bars show 100% linear junction-state cue decoding and 50% greedy branch accuracy after factual reset-state injection.

The training evidence suggests several explanations worth keeping separate. Aggregate reset reconstruction was **99.1%**, while reconstruction of the task-critical cue cell was **50%**. The eight-row supervised replay window also never covered both the cue at reset and the terminal reward. The terminal-reward window received reset information through detached burn-in, which supplies context without a direct gradient to the reset encoding. Reconstruction weighting, temporal credit assignment and the learning objective are plausible mechanisms. These diagnostics leave their causal contributions unresolved.

**Figure 6 · `training_signal.png`.** Reset reconstruction and the supervised replay window. Aggregate accuracy hides the weak cue-cell score.

**Alt text:** Overall reconstruction is 99.1% and cue-cell reconstruction is 50%. A ten-row timeline shows the cue preceding the eight-row terminal-loss window.

## What carried over to CAGE2?

On one frozen flat checkpoint, replay reproduced all 720 recorded greedy decisions and associated rewards over 24 archived episodes. The selected hidden-session label was already 98.08% balanced-accuracy decodable from current observations on five test episodes, and the posterior and argmax reached 100%. That target did not reproduce the maze’s categorical collision. Replacing the carry while holding the current observation and incoming action fixed changed roughly 17% of held-out greedy choices. This establishes sensitivity to history under the tested intervention. It leaves the specific information responsible for those changes unresolved.

I then tried a stricter question about recent hidden session history. The approved study required pairs identical in current Blue observations, elapsed time and incoming action. Among 40 development episodes, there was one independent fit pair and no validation pairs. The predeclared gate correctly stopped the study before any probe fit or test access. In a separate archive-only return forecasting analysis, five-frame raw observation history improved test R² by 0.014 over a comparable current-state MLP. Confidence intervals for posterior and combined-state histories included zero. Return forecasting and hidden compromise labels concern different tasks.

**Figure 7 · `cage_evidence.png`.** The archive-only return comparison and the separate matched-history support gate. The forecasts use 72 held-out episodes.

**Alt text:** Raw-observation history has a small positive return forecasting interval. Posterior and combined histories span zero. A separate support count shows one fit pair and no validation pairs.

## What I would carry into the next project

I would put the representation question ahead of architecture size. If the information needed for a decision has already disappeared from the hard state, adding attention downstream has little support from this maze result. A comparison that lets a frozen reader see the soft posterior would help isolate what survives compression. A separate controlled study could ask whether replay coverage or the objective helps the model learn to preserve a rare cue. Those are future designs. The existing evidence has a narrower, more useful job: show where information failed to travel and where downstream responses were demonstrably weak.

AI-assisted coding sped up the adapters, tests, and reproducible analysis. I found its most valuable role in making checks easier to write and repeat. A plausible figure label once conflated the encoder with the observation. The fix came from tracing the tensor, pinning the checkpoint and split, and testing the paths on paired data. Human judgment was still needed to choose the counterfactual, set the evidence threshold and stop the final study when the available pairs could not support a conclusion. That discipline is what makes a surprising result worth sharing.

## References and code

- [Hafner et al., *Mastering diverse control tasks through world models*, Nature (2025)](https://www.nature.com/articles/s41586-025-08744-2)
- [TTCP CAGE Challenge 2 and CybORG](https://github.com/cage-challenge/cage-challenge-2)
- [Samvelyan et al., *MiniHack the Planet* (2021)](https://arxiv.org/abs/2109.13202)
- [Belinkov, *Probing Classifiers: Promises, Shortcomings, and Advances* (2022)](https://aclanthology.org/2022.cl-1.7/)
- [DreamerRL implementation and reproduction notes](https://github.com/mspcvsp/dreamerrl)
