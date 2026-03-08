import torch


def test_world_model_cpu_gpu_equivalence(make_world_model, fake_obs):
    if not torch.cuda.is_available():
        return

    # CPU model
    torch.manual_seed(0)
    wm_cpu = make_world_model().cpu()

    # GPU model
    torch.manual_seed(0)
    wm_gpu = make_world_model().cuda()

    wm_gpu.load_state_dict(wm_cpu.state_dict())

    state_cpu = wm_cpu.init_state(batch_size=fake_obs.shape[0])
    state_gpu = wm_gpu.init_state(batch_size=fake_obs.shape[0])

    out_cpu = wm_cpu.observe_step(state_cpu, fake_obs)
    out_gpu = wm_gpu.observe_step(state_gpu, fake_obs.cuda())

    # Move GPU outputs to CPU
    out_gpu = {
        "state": out_gpu["state"].to("cpu"),
        "recon": out_gpu["recon"].cpu(),
        "reward_pred": out_gpu["reward_pred"].cpu(),
    }

    torch.testing.assert_close(out_cpu["state"].h, out_gpu["state"].h)
    torch.testing.assert_close(out_cpu["state"].z, out_gpu["state"].z)
    torch.testing.assert_close(out_cpu["recon"], out_gpu["recon"])
    torch.testing.assert_close(out_cpu["reward_pred"], out_gpu["reward_pred"])
