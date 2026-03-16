import torch

from dreamerrl.training.core import imagination_rollout


def test_trainer_imagination_rollout(world_model):
    B, H = 3, 5
    deter = world_model.deter_size
    stoch = world_model.stoch_size

    state = world_model.init_state(B)

    out = imagination_rollout(
        world_model=world_model,
        actor=None,
        critic=None,
        state=state,
        horizon=H,
        with_values=False,
        with_actions=False,
        no_grad=True,
    )

    # --- Assert non-None first ---
    assert out["h"] is not None
    assert out["z"] is not None

    h = out["h"]
    z = out["z"]

    # --- Shape checks ---
    assert h.shape == (H, B, deter)
    assert z.shape == (H, B, stoch)

    # --- Optional fields ---
    assert out["value"] is None
    assert out["action"] is None

    # --- Finite checks ---
    assert torch.isfinite(h).all()
    assert torch.isfinite(z).all()
