import torch

from dreamerrl.models.world_model import WorldModelState


def test_observe_step_zero_latent_stability(world_model, obs_batch):
    """
    Ensures observe_step behaves correctly when the stochastic latent z is zero.
    This is the Dreamer-Lite / deterministic latent mode stability test.
    """

    wm = world_model.to(torch.device("cpu"))

    batch_size = obs_batch["obs"].shape[0]
    latent_dim = wm.latent_dim

    zero_state = WorldModelState(
        h=torch.zeros(batch_size, wm.deter_size, device="cpu"),
        z=torch.zeros(batch_size, latent_dim, device="cpu"),
    )

    zero_state_before = zero_state.clone()

    with torch.no_grad():
        out = wm.observe_step(zero_state, obs_batch["obs"])

    state_after = out["state"]

    # Invariant: input zero_state is not mutated
    assert torch.allclose(zero_state.h, zero_state_before.h)
    assert torch.allclose(zero_state.z, zero_state_before.z)

    # Invariant: resulting state has correct shapes
    assert state_after.h.shape == (batch_size, wm.deter_size)
    assert state_after.z.shape == (batch_size, latent_dim)
