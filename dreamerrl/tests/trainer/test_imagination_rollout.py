import torch

from dreamerrl.training.core import imagine_trajectory_for_training


def test_training_imagination_rollout(world_model, actor, critic):
    B, H = 3, 5
    deter = world_model.deter_size
    stoch = world_model.stoch_size

    state = world_model.init_state(B)

    traj = imagine_trajectory_for_training(
        world_model=world_model,
        actor=actor,
        critic=critic,
        state=state,
        horizon=H,
    )

    h = traj["h"]
    z = traj["z"]
    reward = traj["reward"]
    value = traj["value"]
    action = traj["action"]

    # Shapes
    assert h.shape == (H, B, deter)
    assert z.shape == (H, B, stoch)
    assert reward.shape == (H, B)
    assert value.shape == (H, B)
    assert action.shape == (H, B)

    # Finite values
    assert torch.isfinite(h).all()
    assert torch.isfinite(z).all()
    assert torch.isfinite(reward).all()
    assert torch.isfinite(value).all()
