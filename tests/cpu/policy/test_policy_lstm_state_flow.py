import torch

from tests.helpers.fake_policy import make_fake_policy


def test_lstm_pre_step_to_post_step_state_flow():
    """
    forward_step must produce the same (h_{t+1}, c_{t+1}, logits, value)
    as the single-timestep slice of forward_sequence.
    """
    policy = make_fake_policy(
        rollout_steps=4,
        num_envs=2,
        obs_dim=4,
        hidden_size=4,
    )

    T, B, D, H = 4, 2, 4, 4

    # Build a fake sequence
    obs = torch.randn(T, B, D)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)

    # Sequence path
    seq_out = policy.forward_sequence(obs, h0, c0)
    logits_seq_t0 = seq_out.logits[0]  # (B, A)
    value_seq_t0 = seq_out.value[0]  # (B,) or (B,1)
    h1_seq = seq_out.hn[0]  # (B, H)
    c1_seq = seq_out.cn[0]  # (B, H)

    # Single-step path
    logits_step, value_step, h1_step, c1_step, _ = policy.forward_step(obs[0], h0, c0)

    # Bit-exact comparisons
    assert torch.allclose(logits_step, logits_seq_t0)
    assert torch.allclose(value_step, value_seq_t0)
    assert torch.allclose(h1_step, h1_seq)
    assert torch.allclose(c1_step, c1_seq)
