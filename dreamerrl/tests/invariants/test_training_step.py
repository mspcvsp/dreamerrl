import torch

from dreamerrl.training.core.world_model_update import world_model_training_step


def test_world_model_gradients_finite(dummy_batch, world_model):
    loss = world_model_training_step(world_model, dummy_batch)
    loss.backward()

    for p in world_model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
