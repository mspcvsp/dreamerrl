import torch


def test_world_model_gradients_finite(world_model, fake_batch):
    loss, _ = world_model.training_step(fake_batch)
    loss.backward()

    for p in world_model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
