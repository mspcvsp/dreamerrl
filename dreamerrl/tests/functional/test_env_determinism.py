import gymnasium as gym
import numpy as np
import popgym  # noqa: F401  # ensures PopGym registers its envs


def test_env_determinism():
    env_id = "popgym-RepeatFirstEasy-v0"
    seed = 0
    steps = 200

    traj1 = rollout(env_id, seed, steps)
    traj2 = rollout(env_id, seed, steps)

    assert all(np.array_equal(a[0], b[0]) for a, b in zip(traj1, traj2)), (
        "Trajectories differ between runs with the same seed"
    )


def rollout(env_id, seed, steps=200):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    traj = []
    for t in range(steps):
        action = 0
        obs, r, term, trunc, _ = env.step(action)
        traj.append((np.array(obs, copy=True), r, term, trunc))

        if term or trunc:
            obs, _ = env.reset(seed=seed)
    return traj
