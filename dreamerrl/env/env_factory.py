# dreamerrl/env/env_factory.py

from __future__ import annotations

from dreamerrl.utils.types import EnvironmentConfig


def make_env(cfg: EnvironmentConfig, device, probe=None):
    """
    Unified Dreamer-V3 environment factory.

    Selects the correct environment wrapper based on env_id prefix:
        popgym-*
        minigrid-*
        minihack-*
        cage2-*

    All wrappers implement the Dreamer-native EnvInterface:
        reset() -> dict(state, reward, is_first, is_last, is_terminal, prev_action)
        step(action) -> same dict
        batch_size, obs_dim, action_dim
    """

    env_id = cfg.env_id.lower()

    # ---------------------------------------------------------
    # PopGym
    # ---------------------------------------------------------
    if env_id.startswith("popgym"):
        from dreamerrl.env.popgym.popgym_parallel_env import PopGymParallelEnv
        from dreamerrl.env.popgym.popgym_wrappers import PopGymVecEnv

        if cfg.parallel:
            return PopGymParallelEnv(cfg, device=device)
        return PopGymVecEnv(cfg, device=device, probe=probe)

    # ---------------------------------------------------------
    # Minigrid
    # ---------------------------------------------------------
    if env_id.startswith("minigrid"):
        from dreamerrl.env.minigrid.minigrid_parallel_env import MinigridParallelEnv
        from dreamerrl.env.minigrid.minigrid_wrappers import MinigridVecEnv

        if cfg.parallel:
            return MinigridParallelEnv(cfg, device=device)
        return MinigridVecEnv(cfg, device=device)

    # ---------------------------------------------------------
    # MiniHack
    # ---------------------------------------------------------
    if env_id.startswith("minihack"):
        from dreamerrl.env.minihack.minihack_parallel_env import MiniHackParallelEnv
        from dreamerrl.env.minihack.minihack_wrappers import MiniHackVecEnv

        if cfg.parallel:
            return MiniHackParallelEnv(cfg, device=device)
        return MiniHackVecEnv(cfg, device=device)

    if env_id.startswith("cage2"):
        from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv

        return Cage2BlueEnv(cfg, device=device, probe=probe)

    raise ValueError(f"Unknown environment family for env_id={cfg.env_id}")
