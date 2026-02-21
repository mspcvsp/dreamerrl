"""
Mask Correctness Invariant
--------------------------

This test validates the rollout buffer’s mask construction:

    mask[t, b] = 1.0  if timestep t is valid
                 0.0  if terminated[t] or truncated[t] is True

Why this matters:
-----------------
Masks are used throughout PPO and TBPTT to:
    • zero out invalid timesteps
    • prevent gradients from flowing across episode boundaries
    • ensure GAE bootstrapping is correct
    • ensure drift/saturation/entropy diagnostics ignore invalid steps

If mask construction is wrong, PPO silently trains on invalid data and
recurrent state-flow becomes nondeterministic.

What this test checks:
----------------------
1. terminated → mask = 0
2. truncated  → mask = 0
3. neither    → mask = 1
4. mask shape is (T, B)
5. mask updates correctly across multiple buffer.add() calls
"""

import torch

from dreamerrl.trainer import LSTMPPOTrainer
from dreamerrl.types import LSTMGates, RolloutStep


def test_rollout_buffer_mask_correctness():
    trainer = LSTMPPOTrainer.for_validation()
    buf = trainer.buffer
    device = trainer.device

    T = buf.cfg.rollout_steps
    B = buf.cfg.num_envs
    H = buf.cfg.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    buf.reset()

    # Synthetic rollout
    obs = torch.randn(B, obs_dim, device=device)
    h = torch.zeros(B, H, device=device)
    c = torch.zeros(B, H, device=device)

    for t in range(T):
        # Alternate terminated/truncated flags
        terminated = torch.zeros(B, dtype=torch.bool, device=device)
        truncated = torch.zeros(B, dtype=torch.bool, device=device)

        if t % 3 == 0:
            terminated[:] = True
        elif t % 3 == 1:
            truncated[:] = True

        dummy_gates = LSTMGates(
            i_gates=torch.zeros(B, 1, H, device=device),
            f_gates=torch.zeros(B, 1, H, device=device),
            g_gates=torch.zeros(B, 1, H, device=device),
            o_gates=torch.zeros(B, 1, H, device=device),
            c_gates=torch.zeros(B, 1, H, device=device),
            h_gates=torch.zeros(B, 1, H, device=device),
        )

        step = RolloutStep(
            obs=obs,
            actions=torch.zeros(B, device=device),
            rewards=torch.zeros(B, device=device),
            values=torch.zeros(B, device=device),
            logprobs=torch.zeros(B, device=device),
            terminated=terminated,
            truncated=truncated,
            hxs=h,
            cxs=c,
            gates=dummy_gates,
        )
        buf.add(step)

    mask = buf.mask  # (T, B)

    # Check shape
    assert mask.shape == (T, B)

    for t in range(T):
        if t % 3 == 0:
            assert torch.all(mask[t] == 0.0)  # terminated
        elif t % 3 == 1:
            assert torch.all(mask[t] == 0.0)  # truncated
        else:
            assert torch.all(mask[t] == 1.0)  # valid
