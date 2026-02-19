"""
Hidden‑State Reset Invariant (Rollout Path)
------------------------------------------

This test validates the rollout‑time reset semantics implemented in
policy.forward_sequence(). In this codepath, the policy receives `done[t]`
flags from the environment and must zero the LSTM hidden state at the *next*
timestep (t+1). This matches Gym/PopGym semantics where done[t] indicates that
the transition t → t+1 ends an episode.

Invariant:
    If done[t] == 1, then the PRE‑STEP hidden state for timestep t+1 must be
    zeroed:  h_{t+1} = 0, c_{t+1} = 0.

Why this matters:
-----------------
Correct reset semantics are essential for:
    • preventing memory leakage across episodes
    • ensuring correct PPO bootstrapping and GAE
    • ensuring TBPTT chunk boundaries do not propagate stale memory
    • ensuring masked diagnostics (drift, saturation, entropy) remain valid
    • ensuring rollout‑time and training‑time state‑flow remain aligned

What this test checks:
----------------------
1. A done flag at timestep t causes hidden states to be zeroed at t+1.
2. Hidden states before and after the reset are non‑zero (no accidental wiping).
3. The hidden state at t+1 matches the result of running the LSTM from a
   zero state on the same observation — proving independence from pre‑reset
   history.

If this invariant breaks:
-------------------------
The agent will leak memory across episodes, PPO advantages will misalign,
TBPTT will propagate incorrect state, and diagnostics will become meaningless.
"""

import torch

from lstmppo.trainer import LSTMPPOTrainer


def test_hidden_state_reset_on_env_reset():
    """
    Hidden-state reset invariant (rollout path):

    • forward_sequence must zero hidden states at the timestep *after* a done flag.
    • forward_sequence must NOT zero hidden states before the done.
    • Next hidden state after reset must be independent of pre-reset state.

    NOTE:
    This tests forward_sequence, not evaluate_actions_sequence, because resets
    are a rollout-time concern, not a training-time concern.
    """

    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device

    T, B = 6, 2
    H = trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    obs = torch.randn(T, B, obs_dim, device=device)

    # done[t] = True means "episode ended after step t (transition t→t+1)"
    done = torch.zeros(T, B, dtype=torch.bool, device=device)
    done[2] = True  # episode ends after t=2 → reset applied at t=3

    # Use non-zero initial state so zeroing is obvious
    h0 = torch.randn(B, H, device=device)
    c0 = torch.randn(B, H, device=device)

    out = policy.forward_sequence(obs, h0, c0, done=done)

    h = out.hn  # (T, B, H)
    c = out.cn  # (T, B, H)

    # 1. Hidden state at t=3 must match a fresh unroll from zero
    manual = policy.forward_sequence(
        obs[3:4],  # only timestep 3
        torch.zeros(B, H, device=device),
        torch.zeros(B, H, device=device),
    )

    # POST‑STEP equivalence: h[3] and c[3] must match the LSTM run from zero
    assert torch.allclose(h[3], manual.hn[0], atol=1e-5)
    assert torch.allclose(c[3], manual.cn[0], atol=1e-5)

    # 2. Hidden states before reset must not be zero
    assert not torch.allclose(h[2], torch.zeros_like(h[2]))

    # 3. Hidden states after reset must not be trivially all-zero forever
    assert not torch.allclose(h[4], torch.zeros_like(h[4]))

    # 4. Next hidden state after reset must match running from zero state
    manual = policy.forward_sequence(
        obs[3:5],  # steps 3 and 4
        torch.zeros(B, H, device=device),
        torch.zeros(B, H, device=device),
        done=None,
    )
    # manual.hn[1] is the hidden state at "global" t=4 when starting from zeros at t=3
    assert torch.allclose(h[4], manual.hn[1], atol=1e-5)
