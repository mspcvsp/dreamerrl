import torch


def test_minihack_time_limit(minihack_vec_env):
    for step in range(60):
        out = minihack_vec_env.step(torch.zeros(minihack_vec_env.batch_size, dtype=torch.long))

        if out["is_last"].any():
            print("termination at step", step)
            break

    assert out["is_last"].any()
