"""
NOTE ABOUT TBPTT DIAGNOSTICS INVARIANTS
---------------------------------------
The ONLY TBPTT invariant guaranteed by the architecture is that the core recurrence (forward_tbptt) produces the same
hidden-state sequence (hn) as a full unroll (forward_sequence). This is the recurrence that the trainer uses during PPO
updates.

Diagnostics are pure functions of (T,B,H) hidden states + masks. Therefore:

    If forward_tbptt.hn == forward_sequence.hn,
    then diagnostics(forward_tbptt.hn) == diagnostics(forward_sequence.hn)

This is the correct and sufficient TBPTT diagnostics invariant.

We do NOT require chunked calls to evaluate_actions_sequence to match a full unroll. That function includes encoder,
AR/TAR heads, and training-time bookkeeping, and is not used for TBPTT recurrence. Enforcing equality there would
over-specify the system and fail for legitimate architectural reasons.
"""

import torch

from lstmppo.trainer import LSTMPPOTrainer


def reference_masked_diagnostics(h, masks):
    m = masks.unsqueeze(-1)

    sat = (h.abs() * m).sum() / m.sum().clamp(min=1)

    sig = torch.sigmoid(h)
    ent = (-(sig * torch.log(sig + 1e-8)) * m).sum() / m.sum().clamp(min=1)

    valid_pairs = (masks[1:] * masks[:-1]).unsqueeze(-1)
    drift = ((h[1:] - h[:-1]).pow(2) * valid_pairs).sum() / valid_pairs.sum().clamp(min=1)

    return sat, ent, drift


def test_tbptt_masked_diagnostics_alignment():
    """
    TBPTT-aligned masked diagnostics:
    Diagnostics computed on forward_tbptt.hn must match diagnostics
    computed on forward_sequence.hn. This is the *actual* TBPTT invariant
    guaranteed by the architecture.
    """

    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device

    T, B = 12, 2
    H = trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    obs = torch.randn(T, B, obs_dim, device=device)
    h0 = torch.randn(B, H, device=device)
    c0 = torch.randn(B, H, device=device)

    masks = torch.randint(0, 2, (T, B), device=device).float()
    if masks.sum() == 0:
        masks[0, 0] = 1.0

    # --- Full unroll via forward_sequence ---
    full = policy.forward_sequence(obs, h0, c0)
    h_full = full.hn  # (T, B, H)

    # --- TBPTT unroll via forward_tbptt ---
    tb = policy.forward_tbptt(obs, h0, c0, chunk_size=4)
    h_tb = tb.hn  # (T, B, H)

    # Core TBPTT invariant: hidden states must match
    assert torch.allclose(h_full, h_tb)

    # Diagnostics must therefore match
    sat_full, ent_full, drift_full = reference_masked_diagnostics(h_full, masks)
    sat_tb, ent_tb, drift_tb = reference_masked_diagnostics(h_tb, masks)

    assert torch.allclose(sat_full, sat_tb)
    assert torch.allclose(ent_full, ent_tb)
    assert torch.allclose(drift_full, drift_tb)
