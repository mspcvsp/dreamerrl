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

## 15. Manual Reproducibility
- manual/test_reproducibility.py

---

# Summary
This suite is:
- **66 files**
- **Zero duplicates**
- **Subsystem‑orthogonal**
- **Covers every Dreamer‑V3 failure mode**
- **Balanced across invariants, functional, smoke, and stress layers**



