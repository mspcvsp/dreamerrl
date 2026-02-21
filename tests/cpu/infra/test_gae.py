"""
GAE Invariant Test
------------------
This test verifies the core mathematical invariants of Generalized Advantage
Estimation under the simplest possible conditions (zero value function, no
terminations, linear rewards).

Why this matters:
-----------------
GAE is one of PPO’s most fragile components. Small regressions in the
recursion, normalization order, or dtype handling silently destabilize
training. This test ensures that:

• reward normalization produces mean≈0 and std≈1
• return/advantage tensors have correct (T, B) shapes
• returns and advantages are non‑differentiable targets
• advantages and returns preserve perfect correlation after normalization

If any of these invariants fail, PPO credit assignment becomes misaligned and
training becomes unstable. Never modify GAE logic without re‑running this test.
"""

import torch

from dreamerrl.buffer import RecurrentRolloutBuffer
from dreamerrl.trainer_state import TrainerState


def test_gae_computation_basic(trainer_state: TrainerState):
    cfg = trainer_state.cfg
    buf_cfg = cfg.buffer_config
    device = torch.device("cpu")

    buf = RecurrentRolloutBuffer(trainer_state, device)

    T = buf_cfg.rollout_steps
    B = buf_cfg.num_envs

    # Non-constant rewards so normalization is meaningful
    rewards = torch.linspace(0, 1, T).unsqueeze(1).repeat(1, B)
    buf.rewards.copy_(rewards)

    buf.values.zero_()
    buf.terminated.zero_()
    buf.truncated.zero_()

    last_value = torch.zeros(B)

    buf.compute_returns_and_advantages(last_value)

    # Reward normalization
    assert abs(buf.rewards.mean().item()) < 1e-6
    assert abs(buf.rewards.std(unbiased=False).item() - 1.0) < 1e-6

    # Shapes
    assert buf.advantages.shape == (T, B)
    assert buf.returns.shape == (T, B)

    # Returns normalization
    assert abs(buf.returns.mean().item()) < 1e-6
    assert abs(buf.returns.std(unbiased=False).item() - 1.0) < 1e-6

    # Advantages must not require gradients
    assert buf.advantages.requires_grad is False
    assert buf.returns.requires_grad is False

    # Advantages and returns should be perfectly correlated
    flat_adv = buf.advantages.flatten()
    flat_ret = buf.returns.flatten()

    corr = torch.corrcoef(torch.stack([flat_adv, flat_ret]))[0, 1]
    assert corr > 0.999
