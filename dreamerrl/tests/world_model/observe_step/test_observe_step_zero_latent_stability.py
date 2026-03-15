import torch


def test_observe_step_zero_latent_stability(world_model, obs_batch):
    """
    Ensures observe_step behaves correctly when the stochastic latent z is zero.
    This is the Dreamer-Lite / deterministic latent mode stability test.
    """

    wm = world_model.to(torch.device("cpu"))

    # Construct a zero-latent state matching the model's latent dimension
    batch_size = obs_batch["obs"].shape[0]
    latent_dim = wm.latent_dim  # adjust if your model uses a different attribute

    zero_state = {
        "h": torch.zeros(batch_size, latent_dim, device="cpu"),
        "z": torch.zeros(batch_size, latent_dim, device="cpu"),
    }

    # Clone for mutation check
    zero_state_before = {
        "h": zero_state["h"].clone(),
        "z": zero_state["z"].clone(),
    }

    with torch.no_grad():
        out = wm.observe_step(zero_state, obs_batch["obs"])

    # --- Invariants ---

    # 1. Output must be finite
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} contains non-finite values"

    # 2. Shapes must match input latent shapes
    assert out["h"].shape == zero_state["h"].shape
    assert out["z"].shape == zero_state["z"].shape

    # 3. No mutation of input zero-state
    assert torch.allclose(zero_state["h"], zero_state_before["h"])
    assert torch.allclose(zero_state["z"], zero_state_before["z"])

    # 4. Deterministic under zero-latent conditions
    with torch.no_grad():
        out2 = wm.observe_step(zero_state, obs_batch["obs"])

    for k in out:
        assert torch.allclose(out[k], out2[k]), f"{k} is not deterministic under zero-latent mode"
