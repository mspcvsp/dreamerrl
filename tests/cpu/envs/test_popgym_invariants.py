import torch

from lstmppo.trainer import LSTMPPOTrainer


def test_position_only_cartpole_obs_normalization_stable():
    trainer = LSTMPPOTrainer.for_validation(env_id="PopGym-PositionOnlyCartPole-v0")
    policy = trainer.policy
    device = trainer.device

    # Large-magnitude raw obs should not explode encoder outputs
    raw_obs = torch.tensor([[1000.0, -1000.0]], device=device)  # shape (1, D)
    h0 = policy.initial_hxs(1, device=device)
    c0 = policy.initial_cxs(1, device=device)

    out = policy.forward_step(raw_obs, h0, c0)
    logits = out.logits

    assert torch.isfinite(logits).all()
    assert logits.abs().mean() < 50.0


def test_keycorridor_hidden_state_resets_on_done():
    trainer = LSTMPPOTrainer.for_validation(env_id="PopGym-KeyCorridorEasy-v0")
    policy = trainer.policy
    device = trainer.device

    B = 1
    H = trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    obs = torch.randn(4, B, obs_dim, device=device)
    h = torch.zeros(B, H, device=device)
    c = torch.zeros(B, H, device=device)

    # Step a few times to get non-zero hidden state
    out = policy.forward_sequence(obs[:2], h, c)
    h_mid = out.new_hxs[-1]
    c_mid = out.new_cxs[-1]

    assert h_mid.abs().sum() > 0

    # Simulate episode reset: hidden state must reset to zeros
    h_reset = torch.zeros_like(h_mid)
    c_reset = torch.zeros_like(c_mid)

    out_after = policy.forward_sequence(obs[2:], h_reset, c_reset)
    h_after = out_after.hxs[0]

    assert torch.allclose(h_after, h_reset, atol=1e-6)


def test_truncated_episodes_do_not_propagate_hidden_state():
    trainer = LSTMPPOTrainer.for_validation(env_id="PopGym-LabyrinthEasy-v0")
    policy = trainer.policy
    device = trainer.device

    B = 1
    H = trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    obs = torch.randn(6, B, obs_dim, device=device)
    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    # First "episode" segment
    out_1 = policy.forward_sequence(obs[:3], h0, c0)
    h_mid = out_1.new_hxs[-1]
    c_mid = out_1.new_cxs[-1]

    # Simulate truncation: environment ends due to time limit
    # Next rollout must start from zeros, not from h_mid/c_mid
    h_next = torch.zeros_like(h_mid)
    c_next = torch.zeros_like(c_mid)

    out_2 = policy.forward_sequence(obs[3:], h_next, c_next)
    h_start_next = out_2.hxs[0]

    assert torch.allclose(h_start_next, h_next, atol=1e-6)
