import torch


def test_rollout_matches_repeated_imagine_step(world_model, imagine_input):
    wm = world_model.to("cpu")
    h0 = imagine_input

    with torch.no_grad():
        rollout = wm.imagination_rollout(h0, horizon=5)
        manual = []
        h = h0
        for _ in range(5):
            h = wm.imagine_step(h)
            manual.append(h)

    for s_rollout, s_manual in zip(rollout, manual):
        assert torch.allclose(s_rollout.h, s_manual.h)
        assert torch.allclose(s_rollout.z, s_manual.z)
