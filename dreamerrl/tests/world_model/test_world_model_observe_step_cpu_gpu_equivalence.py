import torch


def test_world_model_observe_step_cpu_gpu_equivalence(make_world_model, fake_obs, state_to_cpu):
    if not torch.cuda.is_available():
        return

    # CPU model
    torch.manual_seed(0)
    wm_cpu = make_world_model().cpu()

    # GPU model
    torch.manual_seed(0)
    wm_gpu = make_world_model().cuda()

    # Ensure identical weights
    wm_gpu.load_state_dict(wm_cpu.state_dict())

    # Init states on correct device
    state_cpu = wm_cpu.init_state(batch_size=fake_obs.shape[0])
    state_gpu = wm_gpu.init_state(batch_size=fake_obs.shape[0])

    # Run observe_step
    out_cpu = wm_cpu.observe_step(state_cpu, fake_obs.cpu())
    out_gpu = wm_gpu.observe_step(state_gpu, fake_obs.cuda())

    # Convert GPU outputs to CPU for comparison
    out_gpu_cpu = {
        "state": state_to_cpu(out_gpu["state"]),
        "recon": out_gpu["recon"].cpu(),
        "reward_pred": out_gpu["reward_pred"].cpu(),
    }

    # Compare
    torch.testing.assert_close(out_cpu["state"].h, out_gpu_cpu["state"].h)
    torch.testing.assert_close(out_cpu["state"].z, out_gpu_cpu["state"].z)
    torch.testing.assert_close(out_cpu["recon"], out_gpu_cpu["recon"])
    torch.testing.assert_close(out_cpu["reward_pred"], out_gpu_cpu["reward_pred"])
