import torch


def test_observe_then_imagine_consistency(world_model, obs_input):
    wm = world_model.to("cpu")

    with torch.no_grad():
        out = wm.observe_step(obs_input)  # dict
        state = out["state"]  # WorldModelState
        next_state = wm.imagine_step(state)  # WorldModelState

    assert isinstance(state, type(next_state))
    assert state.h.shape == next_state.h.shape
    assert state.z.shape == next_state.z.shape
