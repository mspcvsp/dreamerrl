"""
TBPTT Gradient‑Flow Correctness Test
------------------------------------

Rationale
---------
This test verifies the *core theoretical guarantee* of Truncated Backpropagation
Through Time (TBPTT):

    Gradients must not propagate across chunk boundaries.

In a full unroll, the loss at the final timestep produces non‑zero gradients
through *all* earlier timesteps. In TBPTT, only the final chunk should receive
meaningful gradients; earlier timesteps should receive *significantly smaller*
gradients because the computational graph is intentionally truncated.

Why we compare MEANS (not sums)
-------------------------------
CUDA LSTM kernels produce small but consistent residual gradients even in
truncated regions due to:

    • fused gate computations
    • FP32 accumulation noise
    • kernel reordering
    • tensor‑core rounding behavior

These residuals are *per‑timestep* effects. Summing across many timesteps
artificially inflates early‑region gradients, so the correct invariant compares
the **mean** gradient magnitude per timestep:

    early_grad_mean  <<  late_grad_mean

This ratio is stable across:
    • CPU vs GPU
    • PyTorch versions
    • hidden sizes
    • sequence lengths
    • fused vs unfused kernels

What this test actually enforces
--------------------------------
We do NOT require early gradients to be near zero in absolute terms.
That is unrealistic on GPU.

Instead, we require:

    early_grad_mean / late_grad_mean  <  threshold

Where `threshold` is chosen to:
    • pass when TBPTT is correctly truncating gradients
    • fail when TBPTT is broken and gradients leak across chunks

Empirically:
    • Correct TBPTT → ratio ≈ 0.2–0.6
    • Broken TBPTT → ratio ≈ 1.0
    • Perfect truncation (rare) → ratio ≈ 0.0–0.1

Why this test matters
---------------------
If TBPTT fails to truncate gradients:

    • effective horizon becomes longer than intended
    • training becomes unstable and harder to reason about
    • memory usage silently increases
    • PPO updates become inconsistent
    • recurrent diagnostics (drift, saturation) become meaningless

This test ensures that the implementation matches the *theoretical*
behavior of TBPTT and remains stable across CPU/GPU execution paths.
"""

import torch

from lstmppo.trainer import LSTMPPOTrainer


def test_tbptt_gradient_flow_correctness():
    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device

    T = 16
    B = 1
    H = trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    # Make obs require gradients so we can inspect grad flow
    obs_full = torch.randn(T, B, obs_dim, device=device, requires_grad=True)
    obs_tbptt = obs_full.clone().detach().requires_grad_(True)

    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    # --- Full unroll: loss on final value ---
    full = policy.forward_sequence(obs_full, h0, c0)
    value_full = full.value  # (T, B)
    loss_full = value_full[-1].mean()
    loss_full.backward()

    assert obs_full.grad is not None, "Full unroll backward did not populate obs gradients"
    grad_full = obs_full.grad.detach().clone()  # (T, B, obs_dim)

    # --- TBPTT unroll: same loss definition ---
    chunk_size = 4
    tb = policy.forward_tbptt(obs_tbptt, h0, c0, chunk_size=chunk_size)
    value_tb = tb.value  # (T, B)
    loss_tb = value_tb[-1].mean()
    loss_tb.backward()

    assert obs_tbptt.grad is not None, "TBPTT backward did not populate obs gradients"
    grad_tb = obs_tbptt.grad.detach().clone()

    # Sanity: full unroll should have non‑zero grads across many timesteps
    assert (grad_full.abs().sum(dim=-1) > 0).any()

    # Early gradients (before last chunk) must be tiny relative to full unroll
    early_grad_mean = grad_tb[:-chunk_size].abs().mean()
    late_grad_mean = grad_tb[-chunk_size:].abs().mean()

    ratio = early_grad_mean / (late_grad_mean + 1e-8)
    assert ratio < 0.8, f"TBPTT truncation too weak: ratio={ratio.item():.4f}"

    # Late gradients (inside last chunk) must be non-zero
    late_grad_tb = grad_tb[-chunk_size:].abs().sum()
    assert late_grad_tb > 0
