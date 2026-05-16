from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from dreamerrl.utils.types import KLConfig, LatentConfig, NetworkConfig

from .categorical_kl import structured_kl
from .continue_head import ContinueHead
from .decoder import ObsDecoder
from .obs_encoder import ObsEncoder, build_obs_encoder, get_flat_obs_dim
from .posterior import Posterior
from .prior import Prior
from .reward_head import RewardHead
from .world_model_core import RSSMCore


@dataclass
class WorldModelState:
    """
    Dreamer‑V3 RSSM state: (h, z), with z factored as (B, K, C).
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
    def __init__(
        self,
        *,
        obs_space: gym.Space,
        latent: LatentConfig,
        net: NetworkConfig,
        free_bits: float = 0.0,
        kl_cfg: Optional[KLConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device or torch.device("cpu")
        self.latent = latent
        self.net_cfg = net
        self.free_bits = free_bits

        self.kl_cfg = kl_cfg or KLConfig(
            max_kl=100.0,
            min_kl=-1e-6,
            require_nonzero=True,
        )

        self.obs_space = obs_space
        self.flat_obs_dim = get_flat_obs_dim(obs_space)
        self.embed_size = net.hidden_size

        self.encoder: ObsEncoder = build_obs_encoder(obs_space, embed_dim=self.embed_size).to(self.device)
        self.rssm: RSSMCore = RSSMCore(latent=latent, net=net).to(self.device)
        self.prior: Prior = Prior(latent=latent, net=net).to(self.device)
        self.posterior: Posterior = Posterior(latent=latent, net=net).to(self.device)
        self.decoder: ObsDecoder = ObsDecoder(latent=latent, net=net, output_dim=self.flat_obs_dim).to(self.device)
        self.reward_head: RewardHead = RewardHead(latent=latent, net=net).to(self.device)
        self.continue_head: ContinueHead = ContinueHead(latent=latent, net=net).to(self.device)

    def init_state(self, batch_size: int) -> WorldModelState:
        device = next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.latent.deter_size, device=device)
        z0 = torch.zeros(batch_size, self.latent.stoch_size, self.latent.num_classes, device=device)
        return WorldModelState(h=h0, z=z0)

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
        prev_state = self._ensure_state(prev_state)
        embed = self.encoder(obs)

        post_stats = self.posterior(prev_state.h, embed)
        prior_stats = self.prior(prev_state.h)

        z = post_stats["z"]

        h = self.rssm(prev_state.h, action)

        post_stats = {**post_stats, "h": h}
        prior_stats = {**prior_stats, "h": prev_state.h}

        post = WorldModelState(h=h, z=z, prior_stats=prior_stats, post_stats=post_stats)
        prior = WorldModelState(h=prev_state.h, z=prior_stats["z"], prior_stats=prior_stats, post_stats=None)

        recon = self.decoder(h, z)
        reward_logits = self.reward_head(h, z)
        cont_logits = self.continue_head(h, z)

        kl_dict = structured_kl(
            q_probs=post_stats["probs"],
            p_probs=prior_stats["probs"],
            free_bits=self.free_bits,
            kl_cfg=self.kl_cfg,
        )

        for key in ["kl_dyn", "kl_rep", "kl_total"]:
            if not torch.isfinite(kl_dict[key]).all():
                raise ValueError(f"KL divergence {key} is not finite: {kl_dict[key]}")
            post_stats[key] = kl_dict[key]

        return {
            "post": post,
            "prior": prior,
            "post_stats": post_stats,
            "prior_stats": prior_stats,
            "recon": recon,
            "reward_logits": reward_logits,
            "cont_logits": cont_logits,
            "kl": post_stats["kl_total"],
        }

    def imagine_step(
        self,
        prev: Any,
        actor: nn.Module,
        stochastic: bool = True,
    ) -> WorldModelState:
        prev_state = self._ensure_state(prev)

        logits = actor(prev_state.h, prev_state.z)
        dist = Categorical(logits=logits)

        if stochastic:
            a = dist.sample()
        else:
            a = logits.argmax(dim=-1)

        assert self.net_cfg.action_dim is not None, "action_dim must be specified in net config for imagine_step"
        action = F.one_hot(a, num_classes=self.net_cfg.action_dim).float()

        h = self.rssm(prev_state.h, action)

        prior = self.prior(h)

        if stochastic:
            z = prior["z"]
        else:
            idx = prior["probs"].argmax(dim=-1)
            z = F.one_hot(idx, num_classes=self.latent.num_classes).float()

        return WorldModelState(h=h, z=z, prior_stats=prior, post_stats=None)

    def _ensure_state(self, s: Any) -> WorldModelState:
        if isinstance(s, WorldModelState):
            return s
        if isinstance(s, dict) and "state" in s:
            return s["state"]
        raise TypeError("State must be WorldModelState or dict with 'state'")

    def imagine_trajectory_for_training(self, actor, critic, start_state, horizon):
        from dreamerrl.training.core.imagination import imagine_trajectory_for_training

        return imagine_trajectory_for_training(self, actor, critic, start_state, horizon)

    def imagine_trajectory_for_testing(self, actor, start_state, horizon):
        from dreamerrl.training.core.imagination import imagine_trajectory_for_testing

        return imagine_trajectory_for_testing(self, actor, start_state, horizon)
