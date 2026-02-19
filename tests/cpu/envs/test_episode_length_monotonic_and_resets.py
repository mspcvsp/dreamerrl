import torch

from lstmppo.trainer import LSTMPPOTrainer


def test_episode_length_monotonic_and_resets(require_popgym_env) -> None:
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
    prev_ep_len = 0

    for _ in range(256):
        logits, value, new_h, new_c, gates = policy.forward_step(obs, h, c)
        action = logits.argmax(dim=-1)

        state = env.step(action)
        obs = state.obs

        ep_len += 1

        # Invariant 1: monotonic increase
        assert ep_len == prev_ep_len + 1

        if state.terminated or state.truncated:
            # Invariant 2: reset to zero
            ep_len = 0

        prev_ep_len = ep_len
        h, c = new_h, new_c
