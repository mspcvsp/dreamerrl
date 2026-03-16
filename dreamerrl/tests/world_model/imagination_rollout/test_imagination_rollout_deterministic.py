import torch


def test_imagination_rollout_deterministic(world_model, imagine_input):
    wm = world_model.to("cpu")

    with torch.no_grad():
        out1 = wm.imagination_rollout(imagine_input, horizon=5)
        out2 = wm.imagination_rollout(imagine_input, horizon=5)

    assert len(out1) == len(out2) == 5
    for s1, s2 in zip(out1, out2):
        assert torch.allclose(s1.h, s2.h)
        assert torch.allclose(s1.z, s2.z)
