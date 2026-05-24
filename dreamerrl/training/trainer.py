from __future__ import annotations

import math
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
import wandb
from gymnasium.spaces import Discrete

from dreamerrl.env.popgym.popgym_wrappers import PopGymVecEnv
from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.replay_buffer.replay_buffer import ReplayBuffer
from dreamerrl.training.core import actor_critic_update, world_model_training_step
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import DreamerConfig, LatentConfig, LRScheduleConfig, NetworkConfig


class CosineWarmupScheduler:
    """Single shared LR schedule for world, actor, critic (Dreamer‑V3 requirement)."""

    def __init__(self, cfg: LRScheduleConfig):
        self.cfg = cfg

    def __call__(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.base_lr * (step / self.cfg.warmup_steps)

        progress = (step - self.cfg.warmup_steps) / max(1, self.cfg.total_steps - self.cfg.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        min_lr = self.cfg.base_lr * self.cfg.lr_floor
        return min_lr + (self.cfg.base_lr - min_lr) * cosine


class DreamerTrainer:
    """
    Dreamer‑V3 trainer.

    Wiring summary:
      • WorldModel: factored discrete latent RSSM + symlog reward/continue heads
      • Actor: policy over discrete actions from (h, z)
      • Critic: distributional value head over symlog bins
      • ReplayBuffer: stores raw env transitions, samples (B, L, ·) sequences
    """

    def __init__(self, cfg: DreamerConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if cfg.train.cuda and torch.cuda.is_available() else "cpu")

        # -----------------------------------------------------
        # Environment
        # -----------------------------------------------------
        self.env = PopGymVecEnv(cfg.env.env_id, cfg.env.num_envs, device=self.device)
        obs_space = self.env.venv.single_observation_space

        action_space = self.env.venv.single_action_space
        assert isinstance(action_space, Discrete)
        self.action_dim: int = int(action_space.n)

        # -----------------------------------------------------
        # Latent + Network configs
        # -----------------------------------------------------
        latent = LatentConfig(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            num_classes=cfg.world.num_classes,
        )

        net_world = NetworkConfig(
            hidden_size=cfg.world.hidden_size,
            action_dim=self.action_dim,
            value_bins=cfg.world.value_bins,
        )

        # -----------------------------------------------------
        # World Model
        # -----------------------------------------------------
        self.world = WorldModel(
            obs_space=obs_space,
            latent=latent,
            net=net_world,
            free_nats=cfg.world.free_nats,
            device=self.device,
        )
        self.world_state = self.world.init_state(self.env.batch_size)

        # -----------------------------------------------------
        # Actor + Critic
        # -----------------------------------------------------
        net_actor = NetworkConfig(
            hidden_size=cfg.ac.actor_hidden,
            action_dim=self.action_dim,
        )

        net_critic = NetworkConfig(
            hidden_size=cfg.ac.critic_hidden,
            value_bins=cfg.world.value_bins,
        )

        self.actor = Actor(latent=latent, net=net_actor).to(self.device)
        self.critic = ValueHead(latent=latent, net=net_critic).to(self.device)

        # -----------------------------------------------------
        # Replay Buffer
        # -----------------------------------------------------
        flat_obs_dim = self.world.flat_obs_dim
        self.replay = ReplayBuffer(
            capacity=cfg.train.replay_capacity,
            obs_dim=flat_obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            seq_len=cfg.train.seq_len,
        )

        # -----------------------------------------------------
        # Optimizers
        # -----------------------------------------------------
        self.model_opt = torch.optim.Adam(self.world.parameters(), lr=cfg.train.model_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.train.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.train.critic_lr)

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------
        wandb.init(project="dreamer_v3", config=cfg.__dict__)

        # -----------------------------------------------------
        # Seeding
        # -----------------------------------------------------
        set_global_seeds(cfg.train.seed)

        self.env_state: Dict[str, Any] = self.env.reset()
        self.total_env_steps: int = 0

    @property
    def global_step(self) -> int:
        return self.total_env_steps

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    def train(self, total_updates: int) -> None:
        lr_cfg = LRScheduleConfig(
            base_lr=self.cfg.train.model_lr,
            warmup_steps=self.cfg.train.warmup_steps,
            total_steps=total_updates,
            lr_floor=0.1,
        )
        lr_schedule = CosineWarmupScheduler(lr_cfg)

        for update_idx in range(total_updates):
            t0 = time.time()

            lr = lr_schedule(update_idx)
            for pg in self.model_opt.param_groups:
                pg["lr"] = lr
            for pg in self.actor_opt.param_groups:
                pg["lr"] = lr
            for pg in self.critic_opt.param_groups:
                pg["lr"] = lr

            self.collect_env_steps()

            batch = self.replay.sample(batch_size=self.cfg.train.batch_size)

            model_loss = self.update_world_model(batch, update_idx)
            actor_loss, critic_loss = self.update_actor_critic(batch, update_idx)

            wandb.log(
                {
                    "loss/model": model_loss,
                    "loss/actor": actor_loss,
                    "loss/critic": critic_loss,
                    "time/update": time.time() - t0,
                },
                step=update_idx,
            )

    # -------------------------------------------------------------
    # Collect steps from environment
    # -------------------------------------------------------------
    def collect_env_steps(self) -> None:
        # 1. Choose discrete action
        if self.global_step < self.cfg.train.random_exploration_steps:
            actions_discrete = torch.randint(
                low=0,
                high=self.action_dim,
                size=(self.env.batch_size,),
                device=self.device,
            )
        else:
            actions_discrete, _ = self.actor.act(self.world_state)

        # 2. Step environment with discrete actions
        env_out = self.env.step(actions_discrete)

        # 3. One‑hot encode actions for RSSMCore
        actions_one_hot = F.one_hot(actions_discrete, num_classes=self.action_dim).float()

        # 4. Update latent state using observation + one‑hot action
        wm_out = self.world.observe_step(
            prev_state=self.world_state,
            obs=env_out["state"],
            action=actions_one_hot,
            reward=env_out["reward"],
            is_first=env_out["is_first"],
            is_last=env_out["is_last"],
            is_terminal=env_out["is_terminal"],
        )
        self.world_state = wm_out["post"]

        # 5. Store raw env transition in replay
        done = env_out["is_last"].float()

        self.replay.add(
            obs=env_out["state"],
            action=actions_discrete,
            reward=env_out["reward"],
            done=done,
        )

        self.env_state = env_out
        self.total_env_steps += self.env.batch_size

    # -------------------------------------------------------------
    # World Model Update
    # -------------------------------------------------------------
    def update_world_model(self, batch: Dict[str, torch.Tensor], update_idx: int) -> float:
        self.model_opt.zero_grad()

        loss = world_model_training_step(
            world_model=self.world,
            batch=batch,
            kl_scale=self.cfg.world.kl_scale,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world.parameters(), self.cfg.train.grad_clip)
        self.model_opt.step()

        return float(loss.item())

    # -------------------------------------------------------------
    # Actor + Critic Update
    # -------------------------------------------------------------
    def update_actor_critic(self, batch: Dict[str, torch.Tensor], update_idx: int):
        actor_loss, critic_loss = actor_critic_update(
            world_model=self.world,
            actor=self.actor,
            critic=self.critic,
            batch=batch,
            imagination_horizon=self.cfg.world.imagination_horizon,
            discount=self.cfg.ac.discount,
            lam=self.cfg.ac.lambda_,
        )

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.train.grad_clip)
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.train.grad_clip)
        self.critic_opt.step()

        return float(actor_loss.item()), float(critic_loss.item())
