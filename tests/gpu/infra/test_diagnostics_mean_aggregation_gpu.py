"""
NOTE ABOUT DIAGNOSTICS:
-----------------------
There are *two* diagnostics paths in the system, and this test exists to
ensure we never accidentally collapse them into one.

1. policy.compute_diagnostics(...)
   - Rollout‑time only
   - Mask‑agnostic
   - Lightweight, deterministic
   - Used by GPU/CPU equivalence tests and debugging
   - MUST remain simple and device‑stable

2. Trainer‑level masked diagnostics
   - Training‑time only
   - Fully mask‑aware (respects terminated/truncated timesteps)
   - Used for PPO metrics, drift/saturation/entropy, TensorBoard
   - Operates on minibatch masks, not raw rollout data

Why this test matters:
----------------------
This test validates the rollout‑time diagnostics path *in isolation*.
Even though the trainer computes mask‑aware diagnostics during PPO
updates, GPU/infra tests rely on the simpler policy‑level version to
verify numerical equivalence and device consistency.

DO NOT remove or merge these two diagnostic paths without re‑evaluating:
  • rollout determinism
  • GPU/CPU equivalence tests
  • mask‑aware training‑time metrics
  • TBPTT and state‑flow invariants
"""

import torch


def test_diagnostics_mean_aggregation_gpu(synthetic_trainer, fake_batch):
    device = torch.device("cuda")
    trainer = synthetic_trainer
    trainer.policy.to(device)

    batch = fake_batch(device=device, batch_size=16, seq_len=32)

    diagnostics = trainer.policy.compute_diagnostics(batch.obs, batch.h0, batch.c0)

    # Ensure diagnostics are scalars (mean over hidden units)
    for name, value in diagnostics.items():
        assert value.dim() == 0, f"{name} must be scalar after mean aggregation"

    # No NaNs
    assert all(torch.isfinite(v) for v in diagnostics.values())
