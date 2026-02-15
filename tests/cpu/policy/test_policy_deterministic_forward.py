import torch

from tests.helpers.fake_policy import make_fake_policy


def test_forward_step_is_bit_exact():
    """
    Ensures LSTMPPOPolicy.forward_step is bit-exact deterministic:
    same obs + same (h_t, c_t) → identical outputs every time.
    """

    policy = make_fake_policy(
        rollout_steps=4,
        num_envs=2,
        obs_dim=4,
        hidden_size=4,
    )

    B, D, H = 2, 4, 4

    obs = torch.randn(B, D)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)

    # Run twice
    logits1, value1, h1, c1, gates1 = policy.forward_step(obs, h0, c0)
    logits2, value2, h2, c2, gates2 = policy.forward_step(obs, h0, c0)

    # Bit-exact checks
    assert torch.allclose(logits1, logits2)
    assert torch.allclose(value1, value2)
    assert torch.allclose(h1, h2)
    assert torch.allclose(c1, c2)

    # Gates is a tuple (i,f,g,o)
    for g1, g2 in zip(gates1, gates2):
        assert torch.allclose(g1, g2)
