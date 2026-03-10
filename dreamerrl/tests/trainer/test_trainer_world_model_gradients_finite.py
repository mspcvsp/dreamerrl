import torch


def test_trainer_world_model_gradients_finite(test_trainer):
    out = test_trainer.world_model_training_step(batch_size=4, seq_len=5)
    loss = out["loss"]

    loss.backward()

    for p in test_trainer.world.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Gradient contains NaN or Inf"
