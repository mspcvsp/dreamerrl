"""
TBPTT Chunk‑Boundary State‑Flow Equivalence Test
------------------------------------------------

This test validates the core invariant of truncated backpropagation‑through‑time
(TBPTT) in the recurrent PPO pipeline:

    Splitting a rollout into TBPTT chunks must produce the exact same
    hidden‑state trajectory as a full unroll.

Invariant:
    For any chunk size K, and any timestep t:
        TBPTT.hn[t] == full_sequence.hn[t]
        TBPTT.cn[t] == full_sequence.cn[t]

Why this matters:
-----------------
TBPTT is used during PPO optimization to reduce memory usage and improve
training stability. However, TBPTT must *not* change the underlying recurrent
state‑flow. If chunk boundaries alter hidden states, the following break:

    • PPO logprob/value alignment
    • PRE‑STEP hidden‑state semantics
    • LSTM diagnostics (drift, saturation, entropy)
    • rollout → trainer → policy round‑trip equivalence
    • determinism across CPU/GPU
    • Dreamer‑lite world‑model training

What this test checks:
----------------------
1. Run a full unroll with policy.forward_sequence().
2. Run TBPTT with policy.forward_tbptt() using chunk_size=K.
3. Compare hidden states at every timestep:
       full.hn[t] == tbptt.hn[t]
       full.cn[t] == tbptt.cn[t]

If this invariant breaks:
-------------------------
Chunk boundaries are corrupting state‑flow, meaning:
    • hidden states leak across chunks
    • PRE‑STEP alignment is violated
    • PPO training becomes nondeterministic
    • TBPTT gradients propagate incorrectly

This test ensures TBPTT is a *pure optimization*, not a change in model
semantics.
"""

import torch

from lstmppo.trainer import LSTMPPOTrainer


def test_tbptt_chunk_boundary_state_flow_equivalence():
    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device

    T = 32
    B = 2
    H = trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    # Synthetic rollout
    obs = torch.randn(T, B, obs_dim, device=device)
    h0 = torch.randn(B, H, device=device)
    c0 = torch.randn(B, H, device=device)

    # --- Full unroll ---
    full = policy.forward_sequence(obs, h0, c0)
    full_h = full.hn  # (T, B, H)
    full_c = full.cn

    # --- TBPTT unroll ---
    chunk_size = 8
    tb = policy.forward_tbptt(obs, h0, c0, chunk_size=chunk_size)
    tb_h = tb.hn  # (T, B, H)
    tb_c = tb.cn

    # --- Invariant: hidden states must match exactly ---
    assert torch.allclose(full_h, tb_h, atol=1e-6)
    assert torch.allclose(full_c, tb_c, atol=1e-6)
