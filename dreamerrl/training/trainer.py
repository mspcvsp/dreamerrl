from __future__ import annotations

import torch
from gymnasium.spaces import Discrete

from dreamerrl.env.popgym.popgym_wrappers import PopGymVecEnv
from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.replay_buffer.replay_buffer import DreamerReplayBuffer
from dreamerrl.training.core import actor_critic_update, world_model_training_step
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


class DreamerTrainer:
    def __init__(self, cfg: DreamerConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if cfg.train.cuda and torch.cuda.is_available() else "cpu")

        self.env = PopGymVecEnv(cfg.env.env_id, cfg.env.num_envs, device=self.device)
        obs_space = self.env.venv.single_observation_space

        action_space = self.env.venv.single_action_space
        assert isinstance(action_space, Discrete)
        action_dim: int = int(action_space.n)

        latent = LatentConfig(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            num_classes=cfg.world.num_classes,
        )

        net_world = NetworkConfig(
            hidden_size=cfg.world.hidden_size,
            action_dim=action_dim,
            value_bins=cfg.world.value_bins,
        )

        self.world = WorldModel(
            obs_space=obs_space,
            action_dim=action_dim,
            latent=latent,
            net=net_world,
            free_bits=cfg.world.free_bits,
            device=self.device,
        )

        self.world_state = self.world.init_state(self.env.batch_size)

        net_actor = NetworkConfig(hidden_size=cfg.ac.actor_hidden, action_dim=action_dim)
        net_critic = NetworkConfig(hidden_size=cfg.ac.critic_hidden, value_bins=cfg.world.value_bins)

        self.actor = Actor(latent=latent, net=net_actor).to(self.device)
        self.critic = ValueHead(latent=latent, net=net_critic).to(self.device)

        flat_obs_dim = self.world.flat_obs_dim
        self.replay = DreamerReplayBuffer(
            num_envs=cfg.env.num_envs,
            obs_dim=flat_obs_dim,
            capacity_episodes=cfg.train.replay_capacity,
            device=self.device,
        )

        self.model_opt = torch.optim.Adam(self.world.parameters(), lr=cfg.train.model_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.train.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.train.critic_lr)

        # schedulers, logging, seeding unchanged...
