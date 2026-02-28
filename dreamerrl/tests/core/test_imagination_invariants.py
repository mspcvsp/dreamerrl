import torch

from dreamerrl.training.core import imagination_rollout


def test_imagination_rollout_invariants(world_model):
    B, H = 4, 6
    deter = world_model.deter_size
    stoch = world_model.stoch_size

    state = world_model.init_state(B)
    out = imagination_rollout(world_model, actor=None, critic=None, state=state, horizon=H)

    assert out["h"].shape == (H, B, deter)
    assert out["z"].shape == (H, B, stoch)
    assert out["value"] is None
    assert out["action"] is None
    assert torch.isfinite(out["h"]).all()
    assert torch.isfinite(out["z"]).all()
