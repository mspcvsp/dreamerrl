import torch


def test_rollout_cpu_gpu_equivalence(world_model, imagine_input):
    wm_cpu = world_model.to("cpu")
    wm_gpu = world_model.to("cuda")

    with torch.no_grad():
        out_cpu = wm_cpu.imagination_rollout(imagine_input.to("cpu"), horizon=5)
        out_gpu = wm_gpu.imagination_rollout(imagine_input.to("cuda"), horizon=5)

    assert len(out_cpu) == len(out_gpu) == 5
    for s_cpu, s_gpu in zip(out_cpu, out_gpu):
        assert torch.allclose(s_cpu.h.cpu(), s_gpu.h.cpu(), atol=1e-5)
        assert torch.allclose(s_cpu.z.cpu(), s_gpu.z.cpu(), atol=1e-5)
