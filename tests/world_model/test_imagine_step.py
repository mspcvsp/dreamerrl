def test_imagine_step_shapes(world_model, device):
    B = 4

    state = world_model.init_state(B)
    next_state = world_model.imagine_step(state)

    assert next_state.h.shape == (B, world_model.deter_size)
    assert next_state.z.shape == (B, world_model.stoch_size)
