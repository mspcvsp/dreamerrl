import torch


def test_imagination_rollout_shapes(world_model, imagine_input):
    wm = world_model.to("cpu")
    H = 5

    with torch.no_grad():
        rollout = wm.imagination_rollout(imagine_input, horizon=H)

    assert len(rollout) == H
    for s in rollout:
        assert isinstance(s, type(imagine_input))
        assert s.h.shape == imagine_input.h.shape
        assert s.z.shape == imagine_input.z.shape
