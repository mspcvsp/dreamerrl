import pytest
import torch

from dreamerrl.env.env_factory import make_env
from dreamerrl.utils.types import EnvironmentConfig


@pytest.mark.smoke
def smoke_test_minihack():
    cfg = EnvironmentConfig(
        env_id="MiniHack-Room-5x5-v0",
        num_envs=4,
        max_episode_steps=50,
        seed=123,
        deterministic=True,
        parallel=False,
    )

    env = make_env(cfg, device=torch.device("cpu"))

    print("\n=== RESET ===")
    out = env.reset()
    print("state shape:", out["state"].shape)
    print("reward shape:", out["reward"].shape)
    print("is_first:", out["is_first"])
    print("is_last:", out["is_last"])
    print("is_terminal:", out["is_terminal"])
    print("prev_action shape:", out["prev_action"].shape)

    # Check obs_dim consistency
    assert out["state"].shape == (cfg.num_envs, env.obs_dim), "obs_dim mismatch"

    print("\n=== STEP LOOP ===")
    for t in range(5):
        actions = torch.randint(low=0, high=env.action_dim, size=(cfg.num_envs,))
        out = env.step(actions)

        print(f"\nStep {t}")
        print("state shape:", out["state"].shape)
        print("reward:", out["reward"])
        print("is_first:", out["is_first"])
        print("is_last:", out["is_last"])
        print("is_terminal:", out["is_terminal"])
        print("prev_action shape:", out["prev_action"].shape)

        # Validate shapes
        assert out["state"].shape == (cfg.num_envs, env.obs_dim)
        assert out["prev_action"].shape == (cfg.num_envs, env.action_dim)

    print("\n=== PASSED BASIC SHAPE TESTS ===")


if __name__ == "__main__":
    smoke_test_minihack()
