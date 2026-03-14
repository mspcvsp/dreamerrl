import torch


def test_observe_then_imagine_consistency(world_model, obs_input):
    wm = world_model.to("cpu")

    with torch.no_grad():
        state = wm.observe_step(obs_input)
        next_state = wm.imagine_step(state)

    assert next_state["h"].shape == state["h"].shape
    assert torch.isfinite(next_state["h"]).all()
