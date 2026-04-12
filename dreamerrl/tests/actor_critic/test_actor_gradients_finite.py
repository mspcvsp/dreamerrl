import torch


def test_actor_gradients_finite(actor, world_model):
    B = 4
    h = torch.randn(B, world_model.deter_size)
    z = torch.randn(B, world_model.stoch_size)

    logits = actor(h, z)
    loss = logits.sum()
    loss.backward()

    for p in actor.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
