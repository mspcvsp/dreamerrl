# dreamerrl/models/world_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn

from .continue_head import ContinueHead
from .decoder import ObsDecoder
from .obs_encoder import build_obs_encoder, get_flat_obs_dim
from .posterior import Posterior
from .prior import Prior
from .reward_head import RewardHead
from .world_model_core import RSSMCore


@dataclass
class WorldModelState:
    h: torch.Tensor
    z: torch.Tensor
    prior_stats: Optional[Dict[str, torch.Tensor]] = None
    post_stats: Optional[Dict[str, torch.Tensor]] = None

    def to(self, device):
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
    Clean Dreamer‑V3 world model.
    Contains only modules + observe_step + imagine_step.
    All training logic lives in training/core/world_model_update.py.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_dim: int,
        deter_size: int,
        stoch_size: int,
        encoder_hidden: int = 256,
        rssm_hidden: int = 256,
        decoder_hidden: int = 256,
        reward_hidden: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Deterministic initialization for CPU/GPU equivalence tests
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        build_device = torch.device("cpu")
        self.device = device or torch.device("cpu")
        self.deter_size = deter_size
        self.stoch_size = stoch_size

        # Encoder
        self.obs_space = obs_space
        self.flat_obs_dim = get_flat_obs_dim(obs_space)
        self.embed_size = encoder_hidden
        self.encoder = build_obs_encoder(obs_space, embed_dim=self.embed_size).to(build_device)

        # RSSM
        self.rssm = RSSMCore(deter_size, stoch_size, rssm_hidden).to(build_device)

        # Prior / Posterior
        self.prior = Prior(deter_size, stoch_size, rssm_hidden).to(build_device)
        self.posterior = Posterior(deter_size, stoch_size, rssm_hidden).to(build_device)

        # Decoder + Heads
        self.decoder = ObsDecoder(deter_size, stoch_size, decoder_hidden, self.flat_obs_dim).to(build_device)
        self.reward_head = RewardHead(deter_size, stoch_size, reward_hidden).to(build_device)
        self.continue_head = ContinueHead(deter_size, stoch_size, reward_hidden).to(build_device)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_state(self, batch_size: int) -> WorldModelState:
        device = next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.deter_size, device=device)
        z0 = torch.zeros(batch_size, self.stoch_size, device=device)
        return WorldModelState(h=h0, z=z0)

    def observe_step(self, *args, **kwargs):
        """
        Dreamer-Lite compatibility shim:
        Allow calling observe_step({"obs": ..., "action": ...})
        """
        # Case 1: Dreamer-Lite dict call
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]
            obs = batch["obs"]

            # Dreamer-Lite tests assume an implicit initial state
            prev_state = self.init_state(obs.size(0))

            # V3 observe_step ignores action
            return self._observe_step_v3(prev_state, obs)

        # Case 2: Normal V3 call
        return self._observe_step_v3(*args, **kwargs)

    # ------------------------------------------------------------------
    # Observe real environment transition
    # ------------------------------------------------------------------
    def _observe_step_v3(self, prev: Any, obs: torch.Tensor) -> Dict[str, Any]:
        prev_state = self._ensure_state(prev)
        embed = self.encoder(obs)

        post = self.posterior(prev_state.h, embed)
        prior = self.prior(prev_state.h)

        z = post["z"]
        h = self.rssm(prev_state.h, z)

        state = WorldModelState(h=h, z=z, prior_stats=prior, post_stats=post)

        recon = self.decoder(h, z)
        reward_logits = self.reward_head(h, z)
        cont_logits = self.continue_head(h, z).squeeze(-1)

        return {
            "state": state,
            "post_stats": post,
            "prior_stats": prior,
            "recon": recon,
            "reward_logits": reward_logits,
            "cont_logits": cont_logits,
            "kl": self.kl_divergence(post, prior),
        }

    # ------------------------------------------------------------------
    # Imagination step (latent rollout)
    # ------------------------------------------------------------------
    def imagine_step(self, prev: Any, stochastic: bool = True) -> WorldModelState:
        prev_state = self._ensure_state(prev)
        prior = self.prior(prev_state.h)

        z = prior["z"] if stochastic else prior["mean"]
        h = self.rssm(prev_state.h, z)

        return WorldModelState(h=h, z=z, prior_stats=prior, post_stats=None)

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------
    @staticmethod
    def kl_divergence(post, prior):
        mean_q, std_q = post["mean"], post["std"]
        mean_p, std_p = prior["mean"], prior["std"]

        var_q = std_q**2
        var_p = std_p**2

        kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5
        return kl.sum(dim=-1)

    def structured_kl(self, post, prior):
        kl_dyn = self.kl_divergence(
            {"mean": post["mean"].detach(), "std": post["std"].detach()},
            prior,
        )
        kl_rep = self.kl_divergence(
            post,
            {"mean": prior["mean"].detach(), "std": prior["std"].detach()},
        )
        return kl_dyn, kl_rep

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_state(self, s):
        if isinstance(s, WorldModelState):
            return s
        if isinstance(s, dict) and "state" in s:
            return s["state"]
        raise TypeError("State must be WorldModelState or dict with 'state'")

    def imagine_trajectory_for_training(self, actor, critic, start_state, horizon):
        from dreamerrl.training.core.imagination import imagine_trajectory_for_training

        return imagine_trajectory_for_training(
            world_model=self,
            actor=actor,
            critic=critic,
            state=start_state,
            horizon=horizon,
        )

    def imagine_trajectory_for_testing(self, start_state, horizon):
        from dreamerrl.training.core.imagination import imagine_trajectory_for_testing

        return imagine_trajectory_for_testing(
            world_model=self,
            state=start_state,
            horizon=horizon,
        )
