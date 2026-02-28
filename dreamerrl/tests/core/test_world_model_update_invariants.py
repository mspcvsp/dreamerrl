import torch

from dreamerrl.training.core import world_model_training_step


def test_world_model_update_invariants(world_model, device):
    B, L = 3, 5
    obs_dim = world_model.flat_obs_dim

    batch = {
        "state": torch.randn(B, L, obs_dim, device=device),
        "reward": torch.randn(B, L, device=device),
    }

    loss = world_model_training_step(world_model, batch, kl_scale=1.0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
