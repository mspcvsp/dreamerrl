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

    # done[t] = 1 means "episode ended after step t (transition t→t+1)"
    done = torch.zeros(T, B, device=device)
    done[2] = 1  # episode ends after t=2 → reset applied at t=3

    # Use non-zero initial state so zeroing is obvious
    h0 = torch.randn(B, H, device=device)
    c0 = torch.randn(B, H, device=device)

    out = policy.forward_sequence(obs, h0, c0, done=done)

    h = out.hn  # (T, B, H)
    c = out.cn  # (T, B, H)

    # 1. Hidden states must be zeroed at the reset timestep (t=3)
    assert torch.allclose(h[3], torch.zeros_like(h[3]))
    assert torch.allclose(c[3], torch.zeros_like(c[3]))

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
