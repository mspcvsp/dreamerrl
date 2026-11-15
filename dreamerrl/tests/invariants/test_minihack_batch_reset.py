import pytest
import torch


@pytest.mark.invariants
@pytest.mark.minihack_invariants
def test_per_env_reset(minihack_env):
    """
    Ensures only terminated envs are reset and others continue.
    We do NOT rely on glyph equality, because MiniHack can produce
    identical glyph grids across resets with the same seed.
    Instead we check:
      - is_first is re-raised only for terminated envs
      - prev_action is cleared only for terminated envs
    """
    minihack_env.reset()

    actions = torch.zeros(minihack_env.batch_size, dtype=torch.long)
    out = None
    for _ in range(200):  # enough steps to force at least one termination
        out = minihack_env.step(actions)
        if out["is_last"].any():
            break

    assert out is not None
    assert out["is_last"].any()

    last_mask = out["is_last"]
    assert torch.equal(out["is_first"], last_mask)
    assert (out["prev_action"][last_mask] == 0).all()
    out2 = minihack_env.step(actions)
    assert not out2["is_first"].any()
    assert (out2["prev_action"].sum(-1) == 1).all()
