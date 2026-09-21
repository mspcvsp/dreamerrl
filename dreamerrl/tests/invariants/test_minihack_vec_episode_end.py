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

    # Dreamer requires terminal == last
    assert torch.equal(
        out["is_last"],
        out["is_terminal"],
    )

    # Reset envs should re-raise is_first on next step
    next_out = minihack_vec_env.step(
        torch.zeros(
            minihack_vec_env.batch_size,
            dtype=torch.long,
        )
    )

    for i in range(minihack_vec_env.batch_size):
        if bool(ended[i]):
            assert next_out["is_first"][i]
