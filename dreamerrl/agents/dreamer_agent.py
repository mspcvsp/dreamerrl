from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from dreamerrl.models.actor import Actor
from dreamerrl.models.decoder import ObsDecoder
from dreamerrl.models.init import init_weights
from dreamerrl.models.obs_encoder import build_obs_encoder, get_flat_obs_dim
from dreamerrl.models.posterior import Posterior
from dreamerrl.models.prior import Prior
from dreamerrl.models.reward_head import RewardHead
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model_core import RSSMCore
from dreamerrl.utils.types import DreamerConfig


# -------------------------------------------------------------
# Latent state container
# -------------------------------------------------------------
@dataclass
class DreamerState:
    h: torch.Tensor  # deterministic hidden state
    z: torch.Tensor  # stochastic latent state (zero for lite mode)


# -------------------------------------------------------------
# Dreamer Agent
# -------------------------------------------------------------
class DreamerAgent(nn.Module):
    """
    Dreamer agent with feature flags for:
    - Dreamer-Lite (deterministic RSSM, no KL, no overshooting)
    - Full Dreamer (stochastic RSSM, KL balancing, free nats, overshooting)
    """

    def __init__(self, cfg: DreamerConfig, obs_space: gym.Space, action_dim: int, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # ---------------------------------------------------------
        # World Model Components
        # ---------------------------------------------------------
        self.encoder = build_obs_encoder(obs_space, cfg.world.encoder_hidden).to(device)

        self.rssm = RSSMCore(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            hidden_size=cfg.world.hidden_size,
        ).to(device)

        self.prior = Prior(cfg.world.deter_size, cfg.world.stoch_size, cfg.world.hidden_size).to(device)
        self.posterior = Posterior(cfg.world.deter_size, cfg.world.stoch_size, cfg.world.hidden_size).to(device)

        self.decoder = ObsDecoder(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            hidden_size=cfg.world.decoder_hidden,
            obs_shape=get_flat_obs_dim(obs_space),
        ).to(device)

        self.reward_head = RewardHead(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
        ).to(device)

        # ---------------------------------------------------------
        # Actor + Critic
        # ---------------------------------------------------------
        self.actor = Actor(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            hidden_size=cfg.ac.actor_hidden,
            action_dim=action_dim,
        ).to(device)

        self.critic = ValueHead(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            hidden_size=cfg.ac.critic_hidden,
        ).to(device)

        self.apply(init_weights)

    # -------------------------------------------------------------
    # Initial latent state
    # -------------------------------------------------------------
    def initial_state(self, batch_size: int) -> DreamerState:
        h = torch.zeros(batch_size, self.cfg.world.deter_size, device=self.device)
        z = torch.zeros(batch_size, self.cfg.world.stoch_size, device=self.device)
        return DreamerState(h=h, z=z)

    # -------------------------------------------------------------
    # Observe real environment transitions
    # -------------------------------------------------------------
    def observe(self, obs: torch.Tensor, prev: DreamerState):
        embed = self.encoder(obs)

        # ------------------------------
        # Dreamer-Lite: deterministic only
        # ------------------------------
        if not self.cfg.use_stochastic_latent:
            # No posterior, no KL, no stochastic z
            h = self.rssm(prev.h, torch.zeros_like(prev.z))
            z = torch.zeros_like(prev.z)

            recon = self.decoder(h, z)
            reward_pred = self.reward_head(h, z)

            return DreamerState(h=h, z=z), {
                "prior": None,
                "post": None,
                "kl": torch.tensor(0.0, device=self.device),
                "recon": recon,
                "reward_pred": reward_pred,
            }

        # ------------------------------
        # Full Dreamer: stochastic RSSM
        # ------------------------------
        prior = self.prior(prev.h)
        post = self.posterior(prev.h, embed)

        # RSSM deterministic update
        h = self.rssm(prev.h, post["z"])
        z = post["z"]

        # Reconstruction + reward prediction
        recon = self.decoder(h, z)
        reward_pred = self.reward_head(h, z)

        # KL divergence
        kl = torch.distributions.kl.kl_divergence(
            torch.distributions.Normal(post["mean"], post["std"]),
            torch.distributions.Normal(prior["mean"], prior["std"]),
        )

        # Free nats
        if self.cfg.use_free_nats:
            kl = torch.clamp(kl, min=self.cfg.world.free_nats)

        # KL balancing
        if self.cfg.use_kl_balance:
            kl = self.cfg.world.kl_balance * kl

        return DreamerState(h=h, z=z), {
            "prior": prior,
            "post": post,
            "kl": kl.mean(),
            "recon": recon,
            "reward_pred": reward_pred,
        }

    # -------------------------------------------------------------
    # Act in environment
    # -------------------------------------------------------------
    @torch.no_grad()
    def act(self, state: DreamerState):
        logits = self.actor(state.h, state.z)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, {"logits": logits, "entropy": dist.entropy()}

    # -------------------------------------------------------------
    # Imagination rollout (latent rollout)
    # -------------------------------------------------------------
    def imagine(self, start: DreamerState, horizon: int):
        h, z = start.h, start.z

        latents, actions, rewards, values = [], [], [], []

        for _ in range(horizon):
            logits = self.actor(h, z)
            dist = Categorical(logits=logits)
            action = dist.sample()

            # Dreamer-Lite: deterministic latent rollout
            if not self.cfg.use_stochastic_latent:
                h = self.rssm(h, torch.zeros_like(z))
                z = torch.zeros_like(z)
            else:
                prior = self.prior(h)
                h = self.rssm(h, prior["z"])
                z = prior["z"]

            reward = self.reward_head(h, z)
            value = self.critic(h, z)

            latents.append((h, z))
            actions.append(action)
            rewards.append(reward)
            values.append(value)

        return {
            "latents": latents,
            "actions": actions,
            "rewards": torch.stack(rewards),
            "values": torch.stack(values),
        }

    # -------------------------------------------------------------
    # λ-return (Dreamer)
    # -------------------------------------------------------------
    def lambda_return(self, rewards, values, discount, lambda_):
        T = rewards.size(0)
        returns = torch.zeros_like(values)

        next_value = values[-1]
        for t in reversed(range(T)):
            delta = rewards[t] + discount * next_value - values[t]
            next_value = values[t] + lambda_ * delta
            returns[t] = next_value

        return returns

    # -------------------------------------------------------------
    # Training update (world model + actor + critic)
    # -------------------------------------------------------------
    def update(self, batch, optimizers):
        model_opt = optimizers["model"]
        actor_opt = optimizers["actor"]
        critic_opt = optimizers["critic"]

        # ---------------------------------------------------------
        # 1. World Model Loss
        # ---------------------------------------------------------
        model_opt.zero_grad()

        embed = self.encoder(batch["obs"])
        h = batch["h0"]
        z = batch["z0"]

        kls, recons, reward_preds = [], [], []

        for t in range(batch["obs"].size(1)):
            if not self.cfg.use_stochastic_latent:
                # Dreamer-Lite: deterministic
                h = self.rssm(h, torch.zeros_like(z))
                z = torch.zeros_like(z)
                kl = torch.tensor(0.0, device=self.device)
            else:
                prior = self.prior(h)
                post = self.posterior(h, embed[:, t])
                h = self.rssm(h, post["z"])
                z = post["z"]

                kl = torch.distributions.kl.kl_divergence(
                    torch.distributions.Normal(post["mean"], post["std"]),
                    torch.distributions.Normal(prior["mean"], prior["std"]),
                )

                if self.cfg.use_free_nats:
                    kl = torch.clamp(kl, min=self.cfg.world.free_nats)

                if self.cfg.use_kl_balance:
                    kl = self.cfg.world.kl_balance * kl

            recon = self.decoder(h, z)
            reward_pred = self.reward_head(h, z)

            kls.append(kl.mean())
            recons.append((recon - batch["obs"][:, t]).pow(2).mean())
            reward_preds.append((reward_pred - batch["reward"][:, t]).pow(2).mean())

        model_loss = (
            torch.stack(recons).mean()
            + torch.stack(reward_preds).mean()
            + self.cfg.world.kl_scale * torch.stack(kls).mean()
        )

        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.train.grad_clip)
        model_opt.step()

        # ---------------------------------------------------------
        # 2. Actor + Critic Loss (Imagination)
        # ---------------------------------------------------------
        actor_opt.zero_grad()
        critic_opt.zero_grad()

        imagined = self.imagine(
            start=DreamerState(h=batch["h0"], z=batch["z0"]),
            horizon=self.cfg.world.imagination_horizon,
        )

        rewards = imagined["rewards"]
        values = imagined["values"]

        # Dreamer-Lite: simple return
        if not self.cfg.use_value_bootstrap:
            returns = rewards.sum(dim=0)
        else:
            returns = self.lambda_return(rewards, values, self.cfg.ac.discount, self.cfg.ac.lambda_)

        actor_loss = -returns.mean()
        critic_loss = (values - returns.detach()).pow(2).mean()

        actor_loss.backward()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.train.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.train.grad_clip)

        actor_opt.step()
        critic_opt.step()

        return {
            "model_loss": model_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
        }
