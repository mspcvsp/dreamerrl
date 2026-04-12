import torch


def test_imagination_seed_determinism(world_model, actor, critic):
    B, T = 3, 5
    state = world_model.init_state(B)

    torch.manual_seed(0)
    out1 = world_model.imagine_trajectory_for_training(actor, critic, state, T)

    torch.manual_seed(0)
    out2 = world_model.imagine_trajectory_for_training(actor, critic, state, T)

    assert torch.allclose(out1["h"], out2["h"])
    assert torch.allclose(out1["z"], out2["z"])
    assert torch.allclose(out1["reward"], out2["reward"])
