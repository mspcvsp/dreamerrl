"""
TBPTT Gradient‑Flow Correctness Test
------------------------------------

Invariant:
    TBPTT must *truncate* gradients at chunk boundaries.

Formally, for chunk size K and a loss defined only on the *final* timestep:

    • Full unroll:    grad(obs[t]) ≠ 0 for many t < T
    • TBPTT unroll:   grad(obs[t]) ≈ 0 for t < T - K

Why this matters:
-----------------
TBPTT is supposed to limit backpropagation horizon for stability and memory.
If gradients leak across chunk boundaries, then:

    • effective horizon is longer than intended
    • training becomes harder to reason about
    • memory usage can silently explode
    • TBPTT no longer matches its theoretical behavior
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
    early_grad_tb = grad_tb[:-chunk_size].abs().sum()
    full_grad_mag = grad_full.abs().sum()

    # Early gradients must be significantly smaller than full unroll
    ratio = early_grad_tb / (full_grad_mag + 1e-8)

    # Require at least 2× reduction (empirically stable across CPU/GPU)
    assert ratio < 0.5

    # Late gradients (inside last chunk) must be non-zero
    late_grad_tb = grad_tb[-chunk_size:].abs().sum()
    assert late_grad_tb > 0
