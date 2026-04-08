Dreamer‑V3 Invariant Contract
=============================

1. RSSM
   - observe_step: shapes, finite, no mutation
   - posterior/prior: std>0, KL>=0, KL finite
   - imagine_step: no mutation, finite

2. World Model Update
   - pred loss finite
   - KL_dyn, KL_rep finite
   - free-bits clamp per latent dim
   - total loss finite

3. Imagination
   - h,z,reward,action shapes correct
   - bootstrap_value shape correct
   - deterministic under fixed seed
   - no mutation

4. Distributional Critic
   - logits shape (B,num_bins)
   - value_from_logits returns (B,)
   - twohot_encode returns (B,num_bins)
   - weights sum to 1
   - gradients finite

5. Actor
   - logits shape (B,action_dim)
   - logp matches sampled actions
   - entropy finite
   - advantage normalization p5–p95
   - gradients finite

6. Actor–Critic Update
   - λ-returns shape (T,B)
   - losses finite
   - deterministic under fixed seed

7. Trainer
   - full training step finite
   - replay buffer shapes correct
   - deterministic under fixed seed
   - CPU/GPU equivalence (within tolerance)
