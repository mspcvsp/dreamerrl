import torch

from dreamerrl.training.test_trainer import _TestDreamerTrainer


def test_trainer_world_model_training_step_deterministic(make_world_model, replay_buffer_factory, device):
    torch.manual_seed(0)
    rb1 = replay_buffer_factory()
    wm1 = make_world_model()
    trainer1 = _TestDreamerTrainer(wm1, None, None, rb1, device)

    torch.manual_seed(0)
    rb2 = replay_buffer_factory()
    wm2 = make_world_model()
    trainer2 = _TestDreamerTrainer(wm2, None, None, rb2, device)

    out1 = trainer1.world_model_training_step(batch_size=4, seq_len=5)
    out2 = trainer2.world_model_training_step(batch_size=4, seq_len=5)

    torch.testing.assert_close(out1["loss"], out2["loss"])
