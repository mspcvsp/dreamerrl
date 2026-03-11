import torch


def test_imagination_rollout_deterministic(world_model):
    B, horizon = 4, 6
    state0 = world_model.init_state(B)

    roll1 = world_model.imagination_rollout(state0, horizon=horizon)
    roll2 = world_model.imagination_rollout(state0, horizon=horizon)

    torch.testing.assert_close(roll1["state"].h, roll2["state"].h)
    torch.testing.assert_close(roll1["state"].z, roll2["state"].z)
    torch.testing.assert_close(roll1["reward_pred"], roll2["reward_pred"])
