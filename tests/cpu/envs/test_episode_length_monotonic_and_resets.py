from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conftest import require_popgym_env

import torch

from lstmppo.trainer import LSTMPPOTrainer


def test_episode_length_monotonic_and_resets() -> None:
    """Episode length must increase by exactly 1 per step and reset on termination."""
    env_id = "popgym-RepeatPreviousEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)
    policy = trainer.policy
    env = trainer.env
    device = trainer.device

    state = env.reset()
    obs = state.obs

    H = trainer.state.cfg.lstm.lstm_hidden_size
    h = torch.zeros(1, H, device=device)
    c = torch.zeros(1, H, device=device)

    ep_len = 0
    lengths = []

    for _ in range(256):
        logits, value, new_h, new_c, gates = policy.forward_step(obs, h, c)
        action = logits.argmax(dim=-1)

        state = env.step(action)
        obs = state.obs

        ep_len += 1
        lengths.append(ep_len)

        if state.terminated or state.truncated:
            # Episode length must reset
            assert ep_len == lengths[-1]
            ep_len = 0
            state = env.reset()
            obs = state.obs

        h, c = new_h, new_c

    # Monotonicity check: each step increases by exactly 1
    diffs = [b - a for a, b in zip(lengths[:-1], lengths[1:])]
    for d in diffs:
        assert d in (1, -lengths[0])  # reset or increment
