import torch


def test_world_model_training_step_cpu_gpu_equivalence(make_world_model, fake_batch):
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    wm_cpu = make_world_model().cpu()
    wm_gpu = make_world_model().cuda()
    wm_gpu.load_state_dict(wm_cpu.state_dict())

    batch_cpu = {k: v.cpu() for k, v in fake_batch.items()}
    batch_gpu = {k: v.cuda() for k, v in fake_batch.items()}

    loss_cpu, _ = wm_cpu.training_step(batch_cpu)
    loss_gpu, _ = wm_gpu.training_step(batch_gpu)

    torch.testing.assert_close(loss_cpu, loss_gpu.cpu(), atol=1e-5, rtol=1e-5)
