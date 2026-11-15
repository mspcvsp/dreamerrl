"""Async PopGym adapter sharing the synchronous transition implementation."""

from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from dreamerrl.env.popgym.popgym_wrappers import PopGymVecEnv, make_env


class PopGymParallelEnv(PopGymVecEnv):
    def _build_vector(self, cfg):
        return AsyncVectorEnv([make_env(cfg, i) for i in range(cfg.num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)
