from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.env.obs_utils import get_flat_obs_dim
from dreamerrl.models.actor import act_in_imagination
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig

from .base_obs_encoder import BaseEncoder
from .categorical_kl import structured_kl
from .continue_head import ContinueHead
from .observation_model import build_observation_model
from .posterior import Posterior
from .prior import Prior
from .reward_head import RewardModel
from .world_model_core import RSSMCore


@dataclass
class WorldModelState:
    """
    Dreamer‑V3 RSSM state:

        h_t: deterministic state (B, deter_size)
        z_t: factored discrete latent (B, K, C)
    """

    h: torch.Tensor
    z: torch.Tensor
    prior_stats: Optional[Dict[str, torch.Tensor]] = None
    post_stats: Optional[Dict[str, torch.Tensor]] = None

    def to(self, device: torch.device) -> "WorldModelState":
        return WorldModelState(
            h=self.h.to(device),
            z=self.z.to(device),
            prior_stats=None if self.prior_stats is None else {k: v.to(device) for k, v in self.prior_stats.items()},
            post_stats=None if self.post_stats is None else {k: v.to(device) for k, v in self.post_stats.items()},
        )

    def clone(self) -> "WorldModelState":
        return WorldModelState(
            h=self.h.clone(),
            z=self.z.clone(),
            prior_stats=None if self.prior_stats is None else {k: v.clone() for k, v in self.prior_stats.items()},
            post_stats=None if self.post_stats is None else {k: v.clone() for k, v in self.post_stats.items()},
        )

    def detach(self) -> "WorldModelState":
        return WorldModelState(
            h=self.h.detach(),
            z=self.z.detach(),
            prior_stats=None if self.prior_stats is None else {k: v.detach() for k, v in self.prior_stats.items()},
            post_stats=None if self.post_stats is None else {k: v.detach() for k, v in self.post_stats.items()},
        )


class WorldModel(nn.Module):
    """
    Dreamer‑V3 world model:

      • ObsEncoder: configured BaseEncoder with an output_size contract
      • RSSMCore: gated transition h_{t+1} = GRU(embed(z_t, a_t), h_t)
      • Prior / Posterior: factored discrete latents over K×C
      • ObsDecoder: reconstructs observations from (h, z)
      • RewardHead: distributional symlog reward
      • ContinueHead: Bernoulli continuation (episode continues vs terminates)
      • KL: structured KL_dyn / KL_rep with aggregate free‑nats
    """

    def __init__(
        self,
        *,
        obs_space: gym.Space,
        dreamer_cfg: DreamerConfig,
        latent: LatentConfig,
        net: NetworkConfig,
        device: Optional[torch.device] = None,
        probe=None,
    ):
        super().__init__()

        self.device = device or torch.device("cpu")
        self.cfg = dreamer_cfg.world
        self.latent = latent
        self.net_cfg = net
        self.probe = probe
        self.obs_space = obs_space
        self.flat_obs_dim = get_flat_obs_dim(obs_space)
        self.embed_size = net.hidden_size
        if not torch.isfinite(torch.tensor(self.cfg.reconstruction_scale)) or self.cfg.reconstruction_scale < 0:
            raise ValueError("reconstruction_scale must be finite and nonnegative")

        encoder = build_observation_model(
            "encoder", obs_space=obs_space, cfg=self.cfg, latent=latent, net=net, device=self.device
        )
        assert isinstance(encoder, BaseEncoder)
        self.encoder: BaseEncoder = encoder
        self.embed_size = self.encoder.output_size
        self.rssm: RSSMCore = RSSMCore(latent=latent, net=net).to(self.device)

        deterministic_latent_for_tests = dreamer_cfg.train.deterministic_env

        self.prior: Prior = Prior(
            latent=latent, net=net, deterministic_latent_for_tests=deterministic_latent_for_tests
        ).to(self.device)

        self.posterior: Posterior = Posterior(
            latent=latent,
            net=net,
            deterministic_latent_for_tests=deterministic_latent_for_tests,
            embed_size=self.embed_size,
        ).to(self.device)

        self.decoder = build_observation_model(
            "decoder", obs_space=obs_space, cfg=self.cfg, latent=latent, net=net, device=self.device
        )

        # Keep the historical reward_heads.main state_dict namespace.
        self.reward_heads = RewardModel(latent=latent, net=net).to(self.device)

        self.continue_head: ContinueHead = ContinueHead(latent=latent, net=net).to(self.device)

        # Environment-owned stateless callbacks; no generic auxiliary heads.
        self.aux_objectives = [] if self.cfg.disable_aux_losses else list(self.cfg.aux_objectives)
        names = [objective.name for objective in self.aux_objectives]
        if len(names) != len(set(names)):
            raise ValueError("Auxiliary loss names must be unique")
        for objective in self.aux_objectives:
            if not objective.name or not callable(objective.fn):
                raise ValueError("Auxiliary losses require a name and callable")
            if isinstance(objective.fn, nn.Module):
                raise ValueError("Auxiliary callbacks must be stateless; modules need explicit model registration")
            if not torch.isfinite(torch.tensor(objective.weight)) or objective.weight < 0:
                raise ValueError("Auxiliary weights must be finite and nonnegative")

        # Backward‑compatibility alias for invariants + actor/critic tests
        self.reward_head = self.reward_heads.main

    def init_state(self, batch_size: int) -> WorldModelState:
        device = next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.latent.deter_size, device=device)
        z0 = torch.zeros(batch_size, self.latent.num_classes, self.latent.stoch_size, device=device)

        return WorldModelState(h=h0, z=z0)

    def filter_step(self, prev_state, embed, action, is_first=None) -> WorldModelState:
        """Posterior filtering only: no prior, prediction heads, or training KL."""
        prev_state = self._ensure_state(prev_state)
        if is_first is not None:
            keep = (~is_first.bool()).to(prev_state.h.dtype).unsqueeze(-1)
            prev_state = WorldModelState(h=prev_state.h * keep, z=prev_state.z * keep.unsqueeze(-1))
            action = action * keep
        h = self.rssm(prev_state.h, prev_state.z, action)
        stats = {**self.posterior(h, embed), "h": h}
        return WorldModelState(h=h, z=stats["z"], post_stats=stats)

    def observe_step(
        self,
        prev_state: Any,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        is_first: Optional[torch.Tensor] = None,
        is_last: Optional[torch.Tensor] = None,
        is_terminal: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        One environment step:

          • encode obs_t
          • posterior q(z_t | h_t, embed_t)
          • prior    p(z_t | h_t)
          • deterministic transition h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
          • decode obs_t, reward_t, continue_t
          • compute structured KL with free‑nats
        """
        embed = self.encoder(obs)
        filtered = self.filter_step(prev_state, embed, action, is_first)
        h = filtered.h
        post_stats = filtered.post_stats
        assert post_stats is not None
        prior_stats = self.prior(h)
        z = post_stats["z"]
        post_stats = {**post_stats, "h": h}
        prior_stats = {**prior_stats, "h": h}
        post = WorldModelState(h=h, z=z, prior_stats=prior_stats, post_stats=post_stats)
        prior = WorldModelState(h=h, z=prior_stats["z"], prior_stats=prior_stats)

        recon = self.decoder(h, z)
        reward_main_logits = self.reward_heads.main(h, z)
        cont_logits = self.continue_head(h, z)

        kl_dict = structured_kl(q_probs=post_stats["probs"], p_probs=prior_stats["probs"], kl_cfg=self.cfg.kl_cfg)

        if self.probe:
            self.probe.wm_observe(embed, post_stats, prior_stats)
            self.probe.wm_kl(kl_dict)

        for key in ["raw_kl_dyn", "kl_dyn", "raw_kl_rep", "kl_rep", "kl_total"]:
            if not torch.isfinite(kl_dict[key]).all():
                raise ValueError(f"KL divergence {key} is not finite: {kl_dict[key]}")
            post_stats[key] = kl_dict[key]

        return {
            "post": post,
            "prior": prior,
            "post_stats": post_stats,
            "prior_stats": prior_stats,
            "recon": recon,
            "reward_logits": reward_main_logits,
            "reward_main_logits": reward_main_logits,
            "cont_logits": cont_logits,
            "kl": post_stats["kl_total"],
        }

    def imagine_step(
        self,
        prev: Any,
        actor: nn.Module,
        deterministic_imagination: bool = False,
        action: Optional[torch.Tensor] = None,
    ) -> WorldModelState:
        """
        Imagination step in latent space:

          • actor(h, z) → logits over actions
          • sample or argmax action
          • one‑hot encode action
          • RSSMCore transition h_{t+1}
          • prior p(z_{t+1} | h_{t+1})
        """
        prev_state = self._ensure_state(prev)

        logits = None
        if action is None:
            action_dim = self.net_cfg.action_dim
            if action_dim is None:
                raise ValueError("action_dim is required to sample imagined actions")
            logits = actor(prev_state.h, prev_state.z)
            a = act_in_imagination(logits, deterministic_imagination=deterministic_imagination)
            action = F.one_hot(a, num_classes=action_dim).float()

        h = self.rssm(prev_state.h, prev_state.z, action)
        prior = self.prior(h)

        """
        # Non-determinism in imagination shows up after the RSSM transition:
        - actor logits may diverge
        - sampled actions may diverge
        - RSSM transition may diverge
        - prior distribution may diverge
        """
        if self.probe and logits is not None:
            self.probe.wm_imagine(logits, action, h)

        if deterministic_imagination:
            idx = prior["probs"].argmax(dim=-1)
            z = F.one_hot(idx, num_classes=self.latent.stoch_size).float()
        else:
            z = prior["z"]

        return WorldModelState(h=h, z=z, prior_stats=prior, post_stats=None)

    def _ensure_state(self, s: Any) -> WorldModelState:
        if isinstance(s, WorldModelState):
            return s
        if isinstance(s, dict) and "state" in s:
            return s["state"]
        raise TypeError("State must be WorldModelState or dict with 'state'")

    def imagine_trajectory_for_training(self, actor, critic, start_state, horizon, deterministic_imagination=False):
        from dreamerrl.training.core.imagination import imagine_trajectory_for_training

        return imagine_trajectory_for_training(self, actor, critic, start_state, horizon, deterministic_imagination)

    def imagine_trajectory_for_testing(self, actor, start_state, horizon):
        from dreamerrl.training.core.imagination import imagine_trajectory_for_testing

        return imagine_trajectory_for_testing(self, actor, start_state, horizon)
