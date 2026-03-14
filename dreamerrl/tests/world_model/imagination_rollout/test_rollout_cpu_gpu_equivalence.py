import torch


def test_rollout_cpu_gpu_equivalence(world_model, imagine_input):
    wm_cpu = world_model.to("cpu")
    wm_gpu = world_model.to("cuda")

    with torch.no_grad():
        out_cpu = wm_cpu.imagination_rollout(imagine_input.to("cpu"), horizon=5)
        out_gpu = wm_gpu.imagination_rollout(imagine_input.to("cuda"), horizon=5)

    for t in range(5):
        for k in out_cpu[t]:
            assert torch.allclose(out_cpu[t][k].cpu(), out_gpu[t][k].cpu(), atol=1e-5)
