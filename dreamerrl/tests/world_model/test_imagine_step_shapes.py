def test_imagine_step_shapes(make_world_model):
    wm = make_world_model()
    B = 4
    state = wm.init_state(B)

    next_state = wm.imagine_step(state)

    assert next_state.h.shape == (B, wm.deter_size)
    assert next_state.z.shape == (B, wm.stoch_size)
