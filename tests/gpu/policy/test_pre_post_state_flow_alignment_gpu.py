"""
PRE‑STEP / POST‑STEP State‑Flow Equivalence Test
------------------------------------------------

Rationale
---------
This test verifies the most fundamental invariant of the recurrent PPO
architecture: **the policy must produce identical LSTM state‑flow whether it is
run in full‑sequence mode or step‑by‑step mode**.

Every timestep t has two distinct LSTM states:

    PRE‑STEP:   (h_t,   c_t)   — state *before* consuming obs[t]
    POST‑STEP:  (h_{t+1}, c_{t+1}) — state *after* consuming obs[t]

During rollout:
    PRE‑STEP states are stored in the rollout buffer.
    POST‑STEP states become the next timestep’s PRE‑STEP.

During training:
    evaluate_actions_sequence() must reproduce the exact same PRE/POST
    transitions as rollout‑time step‑mode execution.

Why this matters
----------------
If PRE/POST alignment is broken:

    • TBPTT chunking begins from the wrong hidden state
    • PPO logprobs/values become misaligned with rollout behavior
    • recurrent diagnostics (drift, saturation, entropy) become meaningless
    • replay and rollout diverge
    • CPU/GPU equivalence breaks
    • training becomes nondeterministic

What this test enforces
-----------------------
1. Full‑sequence unroll produces:
       full.pre_hxs[t]  ==  h_t
       full.new_hxs[t]  ==  h_{t+1}

2. Step‑mode unroll reconstructs:
       pre_h_list[t]    ==  h_t
       post_h_list[t]   ==  h_{t+1}

3. PRE‑STEP and POST‑STEP sequences match exactly:
       full.pre_hxs  ==  pre_h_list
       full.new_hxs  ==  post_h_list

This guarantees that:
    • rollout → buffer → trainer → policy is perfectly aligned
    • TBPTT chunk boundaries are correct
    • recurrent PPO behaves deterministically across devices
"""

import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyEvalInput, PolicyInput

pytestmark = pytest.mark.gpu


def test_pre_post_state_flow_alignment_gpu(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    device = torch.device("cuda")

    # trainer_state stays on CPU — it has no .to()
    policy = LSTMPPOPolicy(trainer_state).to(device)

    B, T = 3, 7
    H = trainer_state.cfg.lstm.lstm_hidden_size

    policy = LSTMPPOPolicy(trainer_state).to(device)

    obs = torch.randn(T, B, trainer_state.env_info.flat_obs_dim, device=device)
    actions = torch.randint(0, trainer_state.env_info.action_dim, (T, B), device=device)
    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    # Full sequence
    full = policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=obs,
            hxs=h0,
            cxs=c0,
            actions=actions,
        )
    )

    # Step-mode reconstruction
    h_step = h0.clone()
    c_step = c0.clone()
    pre_h_list, post_h_list = [], []
    pre_c_list, post_c_list = [], []

    for t in range(T):
        pre_h_list.append(h_step)
        pre_c_list.append(c_step)

        out = policy.forward(
            PolicyInput(
                obs=obs[t],
                hxs=h_step,
                cxs=c_step,
            )
        )

        eval_step = policy.evaluate_actions(
            out,
            actions[t],
            pre_h=h_step,
            pre_c=c_step,
        )

        h_step = eval_step.new_hxs.squeeze(0)
        c_step = eval_step.new_cxs.squeeze(0)

        post_h_list.append(h_step)
        post_c_list.append(c_step)

    pre_h = torch.stack(pre_h_list, dim=0)
    post_h = torch.stack(post_h_list, dim=0)
    pre_c = torch.stack(pre_c_list, dim=0)
    post_c = torch.stack(post_c_list, dim=0)

    assert torch.allclose(full.pre_hxs, pre_h, atol=1e-6)
    assert torch.allclose(full.new_hxs, post_h, atol=1e-6)
    assert torch.allclose(full.pre_cxs, pre_c, atol=1e-6)
    assert torch.allclose(full.new_cxs, post_c, atol=1e-6)
