import pytest
import torch

from lstmppo.trainer import LSTMPPOTrainer


def require_popgym_env(env_id: str) -> None:
    import gymnasium as gym

    registered = [e.id for e in gym.envs.registry.values()]
    if env_id not in registered:
        pytest.skip(f"PopGym environment not installed: {env_id}")


# ---------------------------------------------------------------------------
# Invariant 1:
# Observation normalization must keep logits finite and bounded.
# This mirrors CAGE‑2, where observations are continuous and normalized.
# ---------------------------------------------------------------------------


def test_obs_normalization_stable() -> None:
    env_id = "popgym-PositionOnlyCartPoleEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)
    policy = trainer.policy
    device = trainer.device

    # Extreme observation to stress normalization
    obs = torch.tensor([[1000.0, -1000.0]], device=device)

    H = trainer.state.cfg.lstm.lstm_hidden_size
    h = torch.zeros(1, H, device=device)
    c = torch.zeros(1, H, device=device)

    logits, value, new_h, new_c, gates = policy.forward_step(obs, h, c)

    assert torch.isfinite(logits).all()
    assert logits.abs().mean() < 50.0


# ---------------------------------------------------------------------------
# Invariant 2:
# Hidden state must evolve over time (not stuck at zero).
# This mirrors CAGE‑2, where the LSTM must track temporal structure.
# ---------------------------------------------------------------------------


def test_hidden_state_evolves() -> None:
    env_id = "popgym-RepeatPreviousEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)
    policy = trainer.policy
    env = trainer.env
    device = trainer.device

    state = env.reset()
    obs = state.obs

    H = trainer.state.cfg.lstm.lstm_hidden_size
    h0 = torch.zeros(1, H, device=device)
    c0 = torch.zeros(1, H, device=device)
    h = h0.clone()
    c = c0.clone()

    done = False
    steps = 0

    while not done and steps < 128:
        logits, value, new_h, new_c, gates = policy.forward_step(obs, h, c)
        action = logits.argmax(dim=-1)

        state = env.step(action)
        obs = state.obs
        done = state.terminated or state.truncated

        h, c = new_h, new_c
        steps += 1

    # LSTM must move away from zero state
    assert not torch.allclose(h, h0)
    assert torch.isfinite(h).all()
    assert torch.isfinite(c).all()


# ---------------------------------------------------------------------------
# Invariant 3:
# Long‑horizon rollouts must keep hidden state finite and stable.
# This mirrors CAGE‑2, where episodes are long and LSTM stability matters.
# ---------------------------------------------------------------------------


def test_long_rollout_hidden_state_stability() -> None:
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

    steps = 0
    max_steps = 512  # long enough to stress LSTM stability

    for _ in range(max_steps):
        logits, value, new_h, new_c, gates = policy.forward_step(obs, h, c)
        action = logits.argmax(dim=-1)

        state = env.step(action)
        obs = state.obs

        h, c = new_h, new_c
        steps += 1

    # Hidden state must remain finite and non‑exploding
    assert torch.isfinite(h).all()
    assert torch.isfinite(c).all()
    assert h.abs().mean() < 100.0
