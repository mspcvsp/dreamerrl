# Dreamer‑V3 Test Suite Map
_Comprehensive, de‑duplicated, subsystem‑organized overview_

---

## 1. Actor Invariants
- invariants/test_actor_cpu_gpu_determinism.py
- invariants/test_actor_critic_consistency.py
- invariants/test_actor_entropy_invariance.py
- invariants/test_actor_logits_contract.py

**Functional:**
- functional/test_actor_forward_determinism.py

---

## 2. Critic / Value Head Invariants
- invariants/test_value_expectation_consistency.py
- invariants/test_value_head_bin_monotonicity.py
- invariants/test_bin_center_monotonicity.py
- invariants/test_distributional_ce_scale_invariance.py

**Functional:**
- functional/test_distributional_value_readout.py

---

## 3. RSSM / World Model Core Invariants
- invariants/test_rssm_core.py
- invariants/test_rssm_core_cpu_gpu_determinism.py
- invariants/test_rssm_cpu_gpu_equivalence.py
- invariants/test_rssm_deterministic.py
- invariants/test_rssm_deterministic_transition_invariance.py
- invariants/test_rssm_no_input_mutation.py
- invariants/test_rssm_zero_latent_stability.py
- invariants/test_rssm_batch_size_invariance.py

**Functional:**
- functional/test_world_model_observe_step.py
- functional/test_imagination_determinism.py

**Smoke:**
- smoke/test_rssm_core_shapes.py

**Stress:**
- stress/stress_rssm_long_horizon.py

---

## 4. Posterior / Prior / Latent Invariants
- invariants/test_posterior_batch_order.py
- invariants/test_prior_posterior.py
- invariants/test_prior_posterior_KL_monotonicity.py
- invariants/test_prior_posterior_shape_contract.py
- invariants/test_categorical_latent_normalization.py

**Functional:**
- functional/test_categorical_kl.py

**Stress:**
- stress/test_posterior_fuzz.py
- stress/test_kl_randomized.py

**Smoke:**
- smoke/test_prior_posterior_shapes.py
- smoke/test_latent_config.py

---

## 5. KL / Free‑Bits / Stability Invariants
- invariants/test_kl_free_bits.py
- invariants/test_kl_explosion_invariant.py

**Stress:**
- stress/test_kl_randomized.py

---

## 6. Decoder Invariants
- invariants/test_decoder_symlog_consistency.py
- invariants/test_decoder_jacobian_finiteness.py
- invariants/test_decoder_reward_continue.py

**Stress:**
- stress/test_decoder_fuzz.py

---

## 7. Reward / Continue Head Invariants
- invariants/test_reward_head_distributional_properties.py
- invariants/test_continue_head_distributional_properties.py

**Stress:**
- stress/test_reward_head_fuzz.py

---

## 8. Symlog / Symexp Invariants
- invariants/test_symlog_invariants.py
- invariants/test_symlog_inverse_consistency.py

**Functional:**
- functional/test_symlog_symexp.py

**Stress:**
- stress/test_symlog_randomized.py

---

## 9. Imagination Invariants
- invariants/test_imagination_horizon.py
- invariants/test_imagination_horizon_stability_strong.py
- invariants/test_imagination_observe_equivalence.py

**Functional:**
- functional/test_imagination_determinism.py

**Stress:**
- stress/test_imagination_random_actions.py

---

## 10. Observe / Training Step Invariants
- invariants/test_observe_step.py
- invariants/test_training_step.py

**Functional:**
- functional/test_world_model_observe_step.py

---

## 11. Replay Buffer / Environment Determinism
**Functional:**
- functional/test_replay_buffer_determinism.py
- functional/test_env_determinism.py

---

## 12. Two‑Hot / Distributional Utilities
**Functional:**
- functional/test_two_hot.py

---

## 13. Learning Rate / Scheduler
**Functional:**
- functional/test_lr_scheduler_curve.py

---

## 14. Smoke Tests (Shape / Config Sanity)
- smoke/test_bins.py
- smoke/test_network_config.py
- smoke/test_prior_posterior_shapes.py
- smoke/test_rssm_core_shapes.py
- smoke/test_sanity_training_loop.py

---

## 15. Training repeatability and optional learning experiments

`functional/test_training_repeatability.py` replaces the old manual reproducibility
check. It runs by default on CPU, repeats three actual world/actor/critic updates
with the same seed, and compares sampled replay, losses, final parameters/buffers,
and episode returns exactly. It also verifies that each network's parameters
changed. It makes no claims about cross-device or different-seed equivalence.

`manual/test_worldmodel_aux.py` is now a MiniHack Room-5x5 learning experiment:
three seeds, auxiliaries off versus novelty only, equal environment-step budgets,
matched initial actor/critic weights, and held-out greedy policy evaluation. It reports random, untrained, and trained
whole-episode returns, including terminal/truncation rewards. It checks that
novelty is active and losses/returns are finite; passing does not establish learning
or auxiliary benefit. The JSON report contains individual episode returns,
evaluation seeds, budgets, losses, and paired differences between the two arms.

Run from the `cage2_env` Python environment:

```bash
# Normal correctness suite, including same-seed training repeatability
python -m pytest dreamerrl/tests -q

# Short experiment wiring check (not evidence of learning)
python -m pytest dreamerrl/tests/manual/test_worldmodel_aux.py --run-manual --steps=2 --eval-episodes=2 -s

# Longer CPU experiment: --steps is optimizer updates per seed/arm
python -m pytest dreamerrl/tests/manual/test_worldmodel_aux.py --run-manual --steps=1000 --eval-episodes=10 -s
```

The experiment prints the JSON report path in pytest's temporary directory; copy
that report if you want to preserve it across later pytest runs. TensorBoard logs
are written under the same temporary directory. The repository-wide pytest default
is only 20 updates; set the budget explicitly for a learning experiment. Evaluate
results across seeds before making task-success or CAGE-readiness claims.

## 16. Optional PopGym benchmark

PopGym remains available for discrete-cue memory tasks such as RepeatFirst and
RepeatPrevious. Both vector adapters use the same transition contract as MiniHack,
including native reward magnitudes, final observations, per-row same-step resets,
and distinct termination/truncation flags. Non-discrete observations/actions are
rejected explicitly rather than flattened inconsistently.

`popgym/test_popgym.py` now runs in the regular suite for both sync and async
collection and training. `popgym/test_popgym_contracts.py` additionally checks
independent resets, native negative/fractional rewards, whole-episode training
returns, and evaluation across rows with different episode lengths. No quarantine
or cross-seed CV/KL acceptance thresholds remain.

The evaluator uses a separate environment in the CLI, held-out reset seeds,
greedy actions, and posterior modes. It includes the final reward and latches each
row's first episode end so autoresets cannot leak rewards from later episodes.
Model modes and torch RNG state are restored afterward. Evaluation episode counts
are per vector row. Training metrics use native rewards and scalar losses rather
than retaining computation graphs.

```bash
# Short CLI wiring check; not evidence of task learning
python -m dreamerrl.scripts.run_popgym --cpu --steps=2 --seeds 0 --episodes=2
```

The CLI reports untrained and trained returns for every seed using the same
held-out evaluation seeds. A short run establishes execution, not learned memory;
use adequate training budgets and multiple seeds to assess performance. PopGym
remains optional to the MiniHack-to-CAGE plan, but its supported code paths are
covered by the regular tests.

## 17. Bounded algorithm fidelity

`invariants/test_dreamer_fidelity.py` verifies uniform categorical mixing,
straight-through probability gradients, actor sampling/scoring, numerical slow
critic regularization, gradient isolation, EMA update order, and rejection of incompatible pooled-latent checkpoints. Checkpoint and repeatability checks include slow-critic state.
See [the fidelity report](../../docs/dreamerv3_fidelity.md) for the pinned reference
and the implementation's explicit deviations from a full reproduction.

## 18. Latent identity and task horizons

`invariants/test_latent_factor_identity.py` checks every main prediction head and
the actor against factor permutations with identical pooled statistics, including
factor-specific gradients. `functional/test_replay_horizons.py` checks mixed short
and long episodes without stitching/padding, one-step episodes, and replay across
collection chunks. `invariants/test_finite_task_horizon.py` checks finite versus
continuing termination targets, clock features, bounded imagined returns, and
checkpoint/evaluation compatibility. MiniHack evaluation regressions exercise the
same clock feature on both terminated and truncated episodes.

## 19. Standard gated RSSM

`invariants/test_rssm_gru.py` compares the transition to the standard GRU equations,
checks update-gate memory retention and replacement over 40 steps, verifies
factored/flattened latent agreement, and checks sequence gradients to initial state,
latents, actions, and GRU parameters. Existing reset, observe/imagine timing,
batch-order, no-input-mutation, and CPU/GPU tests also exercise the GRU.
Checkpoint/repeatability tests include its parameters and optimizer states;
MLP-RSSM v3 files are rejected before loading. Current checkpoints use v4.

## 20. Offline runs and improvement measurement

`functional/test_offline_monitor.py` covers durable JSONL append/torn-tail recovery,
Rich rendering and redirected-output fallback, paired return comparisons, baseline
identity across resume, final-test/validation seed separation, full RNG preservation,
and real CLI SIGTERM handling at a completed-update checkpoint boundary. The
production evaluator is in `eval/minihack_eval.py`; manual experiments reuse it.

See [offline MiniHack operations](../../docs/minihack_offline.md) for tmux, DGX Spark,
run artifacts, graceful stop/resume, and final policy-versus-baselines evaluation.
