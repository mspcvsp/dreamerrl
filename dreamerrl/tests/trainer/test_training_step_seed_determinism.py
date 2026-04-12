import torch


def test_training_step_seed_determinism(test_trainer):
    torch.manual_seed(0)
    out1 = test_trainer.training_step(batch_size=4, seq_len=5)

    torch.manual_seed(0)
    out2 = test_trainer.training_step(batch_size=4, seq_len=5)

    assert torch.allclose(out1["world_model_loss"], out2["world_model_loss"])
    assert torch.allclose(out1["actor_loss"], out2["actor_loss"])
    assert torch.allclose(out1["critic_loss"], out2["critic_loss"])
