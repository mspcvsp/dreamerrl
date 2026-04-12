import torch


def test_world_model_update_finite(test_trainer):
    out = test_trainer.world_model_update(batch_size=4, seq_len=5)
    assert torch.isfinite(out["loss"])
