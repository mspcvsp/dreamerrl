# tests/trainer/test_trainer_world_model_training_step_cpu_gpu_equivalence.py

import torch

from dreamerrl.training.test_trainer import _TestDreamerTrainer


def test_trainer_world_model_training_step_cpu_gpu_equivalence(make_world_model, replay_buffer_factory):
    if not torch.cuda.is_available():
        return

    rb_gpu = replay_buffer_factory()

    # CPU trainer
    torch.manual_seed(0)
    wm_cpu = make_world_model().cpu()
    rb_cpu = replay_buffer_factory()
    trainer_cpu = _TestDreamerTrainer(wm_cpu, None, None, rb_cpu, torch.device("cpu"))

    # GPU trainer
    torch.manual_seed(0)
    wm_gpu = make_world_model().cuda()
    rb_gpu = replay_buffer_factory()
    trainer_gpu = _TestDreamerTrainer(wm_gpu, None, None, rb_gpu, torch.device("cuda"))

    # Sync weights
    wm_gpu.load_state_dict(wm_cpu.state_dict())

    # Compute losses
    out_cpu = trainer_cpu.world_model_training_step(batch_size=4, seq_len=5)
    out_gpu = trainer_gpu.world_model_training_step(batch_size=4, seq_len=5)

    torch.testing.assert_close(out_cpu["loss"], out_gpu["loss"].cpu(), atol=1e-5, rtol=1e-5)
