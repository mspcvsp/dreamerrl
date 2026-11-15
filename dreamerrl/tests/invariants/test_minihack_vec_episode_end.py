import pytest
import torch


@pytest.mark.minihack_invariants
def test_minihack_vec_episode_end(minihack_vec_env):
    out = None

    for _ in range(60):
        out = minihack_vec_env.step(
            torch.zeros(
                minihack_vec_env.batch_size,
                dtype=torch.long,
            )
        )

        if out["is_last"].any():
            break

    assert out is not None

    ended = out["is_last"]

    assert ended.any()

    # SAME_STEP resets return the next initial observation immediately.
    assert torch.equal(out["is_first"], ended)
    assert (~out["is_terminal"] | ended).all()
    assert "transition_state" in out
    next_out = minihack_vec_env.step(torch.zeros(minihack_vec_env.batch_size, dtype=torch.long))
    assert not next_out["is_first"].any()
