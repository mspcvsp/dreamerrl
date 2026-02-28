from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn

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


class WorldModel(nn.Module):
    """
    Dreamer world model with deterministic CPU/GPU‑equivalent initialization.
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
        use_stochastic_latent: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Deterministic initialization for CPU/GPU equivalence tests.
        # Trainer reseeds RNG after construction, so training remains stochastic.
        # ---------------------------------------------------------
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        # Always construct on CPU first
        build_device = torch.device("cpu")

        self.device = device or torch.device("cpu")
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.use_stochastic_latent = use_stochastic_latent

        # ---------------------------------------------------------
        # Observation encoder / decoder
        # ---------------------------------------------------------
        self.obs_space = obs_space
        self.flat_obs_dim = get_flat_obs_dim(obs_space)

        # Encoder: obs → embed
        self.encoder = build_obs_encoder(obs_space, embed_dim=encoder_hidden).to(build_device)

        # RSSM deterministic core
        self.rssm = RSSMCore(
            deter_size=deter_size,
            stoch_size=stoch_size,
            hidden_size=rssm_hidden,
        ).to(build_device)

        # Prior / Posterior (only used if use_stochastic_latent=True)
        if self.use_stochastic_latent:
            self.prior = Prior(
                deter_size=deter_size,
                stoch_size=stoch_size,
                hidden_size=rssm_hidden,
            ).to(build_device)

            self.posterior = Posterior(
                deter_size=deter_size,
                stoch_size=stoch_size,
                hidden_size=rssm_hidden,
            ).to(build_device)
        else:
            self.prior = None
            self.posterior = None

        # Decoder: (h,z) → reconstructed obs
        self.decoder = ObsDecoder(
            deter_size=deter_size,
            stoch_size=stoch_size,
            hidden_size=decoder_hidden,
            obs_shape=self.flat_obs_dim,
        ).to(build_device)

        # Reward head: (h,z) → scalar reward
        self.reward_head = RewardHead(
            deter_size=deter_size,
            stoch_size=stoch_size,
            hidden_size=reward_hidden,
        ).to(build_device)

        """
        DO NOT MOVE TO DEVICE HERE. The caller (trainer or test) will move the model. This guarantees CPU/GPU models
        start from identical weights.
        """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_state(self, batch_size: int) -> WorldModelState:
        h0 = torch.zeros(batch_size, self.deter_size, device=self.device)
        z0 = torch.zeros(batch_size, self.stoch_size, device=self.device)
        return WorldModelState(h=h0, z=z0)

    # ------------------------------------------------------------------
    # Single-step observation update
    # ------------------------------------------------------------------
    def observe_step(
        self,
        prev: WorldModelState,
        obs: torch.Tensor,
    ) -> Dict[str, Any]:
        embed = self.encoder(obs)

        if self.use_stochastic_latent:
            assert self.posterior is not None
            assert self.prior is not None

            post = self.posterior(prev.h, embed)
            z = post["z"]
            prior = self.prior(prev.h)
            h = self.rssm(prev.h, z)
            state = WorldModelState(h=h, z=z, prior_stats=prior, post_stats=post)
            kl = self.kl_divergence(post, prior)
        else:
            z = torch.zeros_like(prev.z)
            h = self.rssm(prev.h, z)
            state = WorldModelState(h=h, z=z)
            kl = torch.zeros((), device=self.device)

        recon = self.decoder(state.h, state.z)
        reward_pred = self.reward_head(state.h, state.z)

        return {
            "state": state,
            "recon": recon,
            "reward_pred": reward_pred,
            "kl": kl,
        }

    # ------------------------------------------------------------------
    # Imagination step
    # ------------------------------------------------------------------
    def imagine_step(self, prev: WorldModelState) -> WorldModelState:
        if self.use_stochastic_latent:
            assert self.prior is not None

            prior = self.prior(prev.h)
            z = prior["z"]
            h = self.rssm(prev.h, z)
            return WorldModelState(h=h, z=z, prior_stats=prior, post_stats=None)
        else:
            z = torch.zeros_like(prev.z)
            h = self.rssm(prev.h, z)
            return WorldModelState(h=h, z=z)

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------
    @staticmethod
    def kl_divergence(post: Dict[str, torch.Tensor], prior: Dict[str, torch.Tensor]) -> torch.Tensor:
        mean_q, std_q = post["mean"], post["std"]
        mean_p, std_p = prior["mean"], prior["std"]

        var_q = std_q**2
        var_p = std_p**2

        kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5
        return kl.sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def reconstruct(self, state: WorldModelState) -> torch.Tensor:
        return self.decoder(state.h, state.z)

    def predict_reward(self, state: WorldModelState) -> torch.Tensor:
        return self.reward_head(state.h, state.z)
