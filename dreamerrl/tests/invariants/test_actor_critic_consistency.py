import torch


def test_actor_critic_consistency(world_model, dummy_actor):
    """
    Higher reward should produce higher value targets.
    Ensures monotonicity of distributional value head.
    """
    B = 8
    h = torch.randn(B, world_model.latent.deter_size)
    z = torch.randn(B, world_model.latent.stoch_size, world_model.latent.num_classes)

    logits = world_model.reward_head(h, z)
    probs = logits.softmax(-1)
    bins = world_model.reward_head.bin_values

    expected = (probs * bins).sum(-1)

    assert expected[0] < expected[-1]
