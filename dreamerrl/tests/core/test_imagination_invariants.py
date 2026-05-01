import torch

from dreamerrl.training.core import imagine_trajectory_for_testing


def test_imagination_invariants(world_model):
    B, H = 4, 6

    state = world_model.init_state(B)

    traj = imagine_trajectory_for_testing(
        world_model=world_model,
        state=state,
        horizon=H,
    )

    h = traj["h"]
    z = traj["z"]
    reward = traj["reward"]

    # Basic shape invariants
    assert h.shape == (H, B, world_model.deter_size)
    assert z.shape == (H, B, world_model.stoch_size * world_model.num_classes)  # shape for discrete latent
    assert reward.shape == (H, B)

    # Finite values
    assert torch.isfinite(h).all()
    assert torch.isfinite(z).all()
    assert torch.isfinite(reward).all()
