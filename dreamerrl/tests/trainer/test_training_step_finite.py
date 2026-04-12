import torch


def test_training_step_finite(test_trainer):
    out = test_trainer.training_step(batch_size=4, seq_len=5)
    assert torch.isfinite(out["world_model_loss"])
    assert torch.isfinite(out["actor_loss"])
    assert torch.isfinite(out["critic_loss"])
