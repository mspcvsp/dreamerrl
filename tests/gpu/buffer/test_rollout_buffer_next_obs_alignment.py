"""
next_obs / next_rewards Alignment Invariant
-------------------------------------------

This test validates the auxiliary‑prediction alignment logic inside
RecurrentRolloutBuffer.get_recurrent_minibatches().

Invariant:
    next_obs[t]     = obs[t+1]
    next_rewards[t] = rewards[t]

with the final timestep padded (and masked out).

Why this matters:
-----------------
Auxiliary prediction (next‑observation and next‑reward prediction) provides
dense supervision that stabilizes LSTM training. Misalignment here silently
breaks:
    • auxiliary losses
    • PPO recurrent evaluation
    • TBPTT chunking
    • diagnostics alignment

What this test checks:
----------------------
1. next_obs is obs shifted by one timestep.
2. The final timestep is padded with zeros.
3. next_rewards matches rewards exactly.
4. Shapes are correct: (T, B, ...) for all tensors.
5. Alignment is preserved after environment‑major minibatch slicing.
"""

import torch

from dreamerrl.trainer import LSTMPPOTrainer
from dreamerrl.types import LSTMGates, RolloutStep


def test_rollout_buffer_next_obs_alignment():
    trainer = LSTMPPOTrainer.for_validation()
    buf = trainer.buffer
    device = trainer.device

    T = buf.cfg.rollout_steps
    B = buf.cfg.num_envs
    H = buf.cfg.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    buf.reset()

    # Create synthetic rollout data
    obs = torch.randn(T, B, obs_dim, device=device)
    rewards = torch.randn(T, B, device=device)

    h = torch.zeros(B, H, device=device)
    c = torch.zeros(B, H, device=device)

    for t in range(T):
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
            rewards=rewards[t],
            values=torch.zeros(B, device=device),
            logprobs=torch.zeros(B, device=device),
            terminated=torch.zeros(B, dtype=torch.bool, device=device),
            truncated=torch.zeros(B, dtype=torch.bool, device=device),
            hxs=h,
            cxs=c,
            gates=dummy_gates,
        )
        buf.add(step)

    # Extract minibatch
    batch = next(buf.get_recurrent_minibatches())

    # Shapes
    assert batch.obs.shape == (T, B, obs_dim)
    assert batch.next_obs.shape == (T, B, obs_dim)
    assert batch.next_rewards.shape == (T, B)

    # next_obs[t] = obs[t+1] for t < T-1
    assert torch.allclose(batch.next_obs[:-1], batch.obs[1:])

    # final timestep padded with zeros
    assert torch.all(batch.next_obs[-1] == 0)

    # next_rewards[t] = rewards[t]
    assert torch.allclose(batch.next_rewards, rewards)
