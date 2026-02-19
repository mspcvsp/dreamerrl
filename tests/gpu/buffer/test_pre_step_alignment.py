"""
PRE‑STEP Hidden State Alignment Test
-----------------------------------

This test verifies the most important invariant of the RecurrentRolloutBuffer:

    The buffer must store the PRE‑STEP LSTM state (h_t, c_t) that was used
    to produce action[t].

Why this invariant matters:
---------------------------
The entire recurrent PPO pipeline depends on PRE‑STEP state correctness:

• Rollout → The policy uses (h_t, c_t) to compute logits[t], values[t], and
  the POST‑STEP state (h_{t+1}, c_{t+1}).

• Buffer → PPO must store the PRE‑STEP state so that training-time evaluation
  (evaluate_actions_sequence) can exactly reproduce rollout-time behavior.

• TBPTT → Each chunk must begin from the true PRE‑STEP state for its first
  timestep; otherwise hidden-state drift accumulates and training becomes
  nondeterministic.

• Diagnostics → Drift, saturation, and entropy metrics are defined over the
  PRE‑STEP hidden-state sequence. Storing POST‑STEP states would silently
  corrupt all diagnostics.

What this test checks:
----------------------
For each timestep t:

1. We capture the PRE‑STEP state (h_t, c_t) before calling forward_step.
2. We store it in the rollout buffer via buf.add(...).
3. After the rollout, we compare the buffer’s stored hxs[t], cxs[t] against
   the PRE‑STEP states we recorded during rollout.

How this enforces the invariant:
--------------------------------
If the buffer were storing POST‑STEP states (h_{t+1}, c_{t+1}) or any other
misaligned state, the comparison would fail immediately. This guarantees that
the buffer’s hidden-state storage matches the exact semantics required by:

    • PPO’s recurrent evaluation
    • TBPTT chunking
    • LSTM state‑flow validation
    • Masked diagnostics alignment

This test protects the entire recurrent pipeline from silent, catastrophic
state‑flow bugs.
"""

import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import LSTMGates, RolloutStep


def test_rollout_buffer_pre_step_alignment():
    """
    PRE‑STEP Hidden State Alignment Test
    ------------------------------------
    This test validates the most critical invariant of the RecurrentRolloutBuffer:

        The buffer must store the PRE‑STEP LSTM state (h_t, c_t) used to produce
        action[t].

    Why this invariant matters:
    ---------------------------
    The entire recurrent PPO pipeline depends on PRE‑STEP state correctness:

    • Rollout:
        The policy uses (h_t, c_t) to compute logits[t], values[t], and the
        POST‑STEP state (h_{t+1}, c_{t+1}).

    • PPO Training:
        evaluate_actions_sequence() must reproduce rollout-time behavior exactly.
        This is only possible if the buffer provides the true PRE‑STEP states.

    • TBPTT:
        Each chunk must begin from the correct PRE‑STEP state; otherwise hidden
        state drift accumulates and training becomes nondeterministic.

    • Diagnostics:
        Drift, saturation, and entropy are defined over PRE‑STEP hidden states.
        Storing POST‑STEP states would silently corrupt all diagnostics.

    What this test checks:
    ----------------------
    1. Before each forward_step, we record the PRE‑STEP state (h_t, c_t).
    2. We store this state in the rollout buffer via buf.add().
    3. After the rollout, we assert that buf.hxs[t] and buf.cxs[t] exactly match
    the recorded PRE‑STEP states for every timestep.

    How this enforces the invariant:
    --------------------------------
    If the buffer were storing POST‑STEP states (h_{t+1}, c_{t+1}), or any other
    misaligned state, the comparison would fail immediately. This guarantees that
    the buffer’s hidden-state storage matches the semantics required for:

        • PPO recurrent evaluation
        • TBPTT chunking
        • LSTM state‑flow validation
        • Masked diagnostics alignment

    If this invariant breaks:
    -------------------------
    PPO logprobs/values misalign, TBPTT unrolls from the wrong state, diagnostics
    become meaningless, and training becomes nondeterministic.
    """

    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device
    buf = trainer.buffer

    T = buf.cfg.rollout_steps
    B = buf.cfg.num_envs
    H = buf.cfg.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    obs = torch.randn(T, B, obs_dim, device=device)

    # Initial PRE‑STEP state
    h = torch.randn(B, H, device=device)
    c = torch.randn(B, H, device=device)

    buf.reset()

    pre_h_list = []
    pre_c_list = []

    for t in range(T):
        pre_h = h.clone()
        pre_c = c.clone()

        logits, value, h_new, c_new, _ = policy.forward_step(obs[t], h, c)

        dummy_gates = LSTMGates(
            i_gates=torch.zeros(B, 1, H, device=device),
            f_gates=torch.zeros(B, 1, H, device=device),
            g_gates=torch.zeros(B, 1, H, device=device),
            o_gates=torch.zeros(B, 1, H, device=device),
            c_gates=torch.zeros(B, 1, H, device=device),
            h_gates=torch.zeros(B, 1, H, device=device),
        )

        step = RolloutStep(
            obs=obs[t],
            actions=torch.zeros(B, device=device),
            rewards=torch.zeros(B, device=device),
            values=value.squeeze(-1).detach(),  # FIXED SHAPE (B,)
            logprobs=torch.zeros(B, device=device),
            terminated=torch.zeros(B, dtype=torch.bool, device=device),
            truncated=torch.zeros(B, dtype=torch.bool, device=device),
            hxs=pre_h,
            cxs=pre_c,
            gates=dummy_gates,
        )
        buf.add(step)

        pre_h_list.append(pre_h)
        pre_c_list.append(pre_c)

        h, c = h_new.detach(), c_new.detach()

    pre_h_stack = torch.stack(pre_h_list, dim=0)
    pre_c_stack = torch.stack(pre_c_list, dim=0)

    assert torch.allclose(buf.hxs, pre_h_stack, atol=1e-6)
    assert torch.allclose(buf.cxs, pre_c_stack, atol=1e-6)
