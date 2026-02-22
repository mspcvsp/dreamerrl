from __future__ import annotations

import time
from typing import Any, Dict

import torch
from gymnasium.spaces import Discrete

import wandb
from dreamerrl.env.popgym.popgym_wrappers import PopGymVecEnv
from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.replay_buffer import DreamerReplayBuffer
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import DreamerConfig


class DreamerLRScheduler:
    def __init__(self, base_lr: float, warmup_steps: int):
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, update_idx: int) -> float:
        if update_idx < self.warmup_steps:
            return self.base_lr * (update_idx / self.warmup_steps)
        return self.base_lr


class DreamerTrainer:
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
        action_dim: int = int(action_space.n)

        # -----------------------------------------------------
        # World Model
        # -----------------------------------------------------
        self.world = WorldModel(
            obs_space=obs_space,
            action_dim=action_dim,
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            encoder_hidden=cfg.world.encoder_hidden,
            rssm_hidden=cfg.world.hidden_size,
            decoder_hidden=cfg.world.decoder_hidden,
            reward_hidden=cfg.world.reward_hidden,
            use_stochastic_latent=cfg.use_stochastic_latent,
            device=self.device,
        )

        # Latent state for online interaction (B = num_envs)
        self.world_state = self.world.init_state(self.env.batch_size)

        # -----------------------------------------------------
        # Actor + Critic
        # -----------------------------------------------------
        self.actor = Actor(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            hidden_size=cfg.ac.actor_hidden,
            action_dim=action_dim,
        ).to(self.device)

        self.critic = ValueHead(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            hidden_size=cfg.ac.critic_hidden,
        ).to(self.device)

        # -----------------------------------------------------
        # Replay Buffer
        # -----------------------------------------------------
        flat_obs_dim = self.world.flat_obs_dim
        self.replay = DreamerReplayBuffer(
            num_envs=cfg.env.num_envs,
            obs_dim=flat_obs_dim,
            capacity_episodes=cfg.train.replay_capacity,
            device=self.device,
        )

        # -----------------------------------------------------
        # Optimizers
        # -----------------------------------------------------
        self.model_opt = torch.optim.Adam(self.world.parameters(), lr=cfg.train.model_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.train.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.train.critic_lr)

        # -----------------------------------------------------
        # LR Schedulers (warmup only)
        # -----------------------------------------------------
        self.model_lr_sch = DreamerLRScheduler(cfg.train.model_lr, cfg.train.warmup_steps)
        self.actor_lr_sch = DreamerLRScheduler(cfg.train.actor_lr, cfg.train.warmup_steps)
        self.critic_lr_sch = DreamerLRScheduler(cfg.train.critic_lr, cfg.train.warmup_steps)

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------
        wandb.init(project="dreamer" + self.cfg.mode, config=cfg.__dict__)

        # -----------------------------------------------------
        # Seeding
        # -----------------------------------------------------
        set_global_seeds(cfg.train.seed)

        # Initial env state
        self.env_state: Dict[str, Any] = self.env.reset()

        # Global step counter (env steps, not updates)
        self.total_env_steps: int = 0

    # -------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------
    @property
    def global_step(self) -> int:
        return self.total_env_steps

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    def train(self, total_updates: int) -> None:
        for update_idx in range(total_updates):
            t0 = time.time()

            # 1. Collect environment steps
            self.collect_env_steps()

            # 2. Sample sequences from replay
            batch = self.replay.sample(
                batch_size=self.cfg.train.batch_size,
                seq_len=self.cfg.train.seq_len,
                device=self.device,
            )

            # 3. Update world model
            model_loss = self.update_world_model(batch, update_idx)

            # 4. Imagination rollout + actor/critic update
            actor_loss, critic_loss = self.update_actor_critic(batch, update_idx)

            # 5. Logging
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
        # Phase 1: random exploration
        if self.global_step < self.cfg.train.random_exploration_steps:
            actions = torch.randint(
                low=0,
                high=self.env.action_dim,
                size=(self.env.batch_size,),
                device=self.device,
            )
        else:
            actions, _ = self.actor.act(self.world_state)

        next_state = self.env.step(actions)

        # Update world model state with new observations
        out = self.world.observe_step(self.world_state, next_state["state"])
        self.world_state = out["state"]

        # Store in replay
        self.replay.add_batch(
            {
                "state": next_state["state"],
                "action": actions,
                "reward": next_state["reward"],
                "is_first": next_state["is_first"],
                "is_last": next_state["is_last"],
                "is_terminal": next_state["is_terminal"],
                "info": next_state["info"],
            }
        )

        self.total_env_steps += self.env.batch_size

    # -------------------------------------------------------------
    # World Model Update
    # -------------------------------------------------------------
    def update_world_model(self, batch: Dict[str, torch.Tensor], update_idx: int) -> float:
        lr = self.model_lr_sch(update_idx)
        for pg in self.model_opt.param_groups:
            pg["lr"] = lr

        self.model_opt.zero_grad()

        obs = batch["state"]  # (B, L, obs_dim)
        reward = batch["reward"]  # (B, L)

        B, L, _ = obs.shape

        state = self.world.init_state(B)

        kl_losses = []
        recon_losses = []
        reward_losses = []

        for t in range(L):
            out = self.world.observe_step(state, obs[:, t])
            state = out["state"]

            recon_losses.append(((out["recon"] - obs[:, t]) ** 2).mean())
            reward_losses.append(((out["reward_pred"].squeeze(-1) - reward[:, t]) ** 2).mean())
            kl_losses.append(out["kl"])

        model_loss = (
            torch.stack(recon_losses).mean()
            + torch.stack(reward_losses).mean()
            + self.cfg.world.kl_scale * torch.stack(kl_losses).mean()
        )

        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world.parameters(), self.cfg.train.grad_clip)
        self.model_opt.step()

        return float(model_loss.item())

    # -------------------------------------------------------------
    # Actor + Critic Update
    # -------------------------------------------------------------
    def update_actor_critic(self, batch: Dict[str, torch.Tensor], update_idx: int) -> tuple[float, float]:
        lr_actor = self.actor_lr_sch(update_idx)
        lr_critic = self.critic_lr_sch(update_idx)

        for pg in self.actor_opt.param_groups:
            pg["lr"] = lr_actor
        for pg in self.critic_opt.param_groups:
            pg["lr"] = lr_critic

        B = batch["state"].size(0)
        state = self.world.init_state(B)

        imagined_states = []
        for _ in range(self.cfg.world.imagination_horizon):
            state = self.world.imagine_step(state)
            imagined_states.append(state)

        rewards = torch.stack([self.world.predict_reward(s) for s in imagined_states])  # (T, B, 1)
        values = torch.stack([self.critic(s.h, s.z) for s in imagined_states])  # (T, B, 1)

        returns = self.lambda_return(
            rewards.squeeze(-1),
            values.squeeze(-1),
            self.cfg.ac.discount,
            self.cfg.ac.lambda_,
        )

        logits = torch.stack([self.actor(s.h, s.z) for s in imagined_states])  # (T, B, A)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)

        actor_loss = -(logp * returns.detach()).mean()
        critic_loss = (values.squeeze(-1) - returns.detach()).pow(2).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.train.grad_clip)
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.train.grad_clip)
        self.critic_opt.step()

        return float(actor_loss.item()), float(critic_loss.item())

    # -------------------------------------------------------------
    # λ-return
    # -------------------------------------------------------------
    def lambda_return(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        discount: float,
        lambda_: float,
    ) -> torch.Tensor:
        T, B = rewards.shape
        returns = torch.zeros_like(values)

        next_value = values[-1]  # (B,)
        for t in reversed(range(T)):
            delta = rewards[t] + discount * next_value - values[t]
            next_value = values[t] + lambda_ * delta
            returns[t] = next_value

        return returns
