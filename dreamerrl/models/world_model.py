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
    """
    Latent state of the world model at a single timestep.
    Works for both Dreamer-Lite and full Dreamer.

    h: deterministic state (B, deter_size)
    z: stochastic state (B, stoch_size) -- zero in Dreamer-Lite
    prior_stats / post_stats: optional dicts with mean/std/z
    """

    h: torch.Tensor
    z: torch.Tensor
    prior_stats: Optional[Dict[str, torch.Tensor]] = None
    post_stats: Optional[Dict[str, torch.Tensor]] = None


class WorldModel(nn.Module):
    """
    Full Dreamer world model:

        obs  → encoder → embed
        (h,z) + embed → posterior → z_t
        (h,z)         → prior     → ẑ_t
        (h,z) → RSSMCore → h_t
        (h,z) → ObsDecoder → obŝ
        (h,z) → RewardHead → r̂

    Flags:
    - use_stochastic_latent = False → Dreamer-Lite (z ≡ 0, no KL)
    - use_stochastic_latent = True  → full Dreamer (prior/posterior, KL)
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

        """
        Deterministic initialization for CPU/GPU equivalence tests. This does NOT affect training randomness because
        DreamerTrainer calls set_global_seeds(cfg.train.seed) *after* model construction.
        """
        torch.manual_seed(0)

        self.device = device or torch.device("cpu")
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.use_stochastic_latent = use_stochastic_latent

        # ---------------------------------------------------------
        # Observation shape / encoder / decoder
        # ---------------------------------------------------------
        self.obs_space = obs_space
        self.flat_obs_dim = get_flat_obs_dim(obs_space)

        # Encoder: obs → embed
        self.encoder = build_obs_encoder(obs_space, embed_dim=encoder_hidden)

        # RSSM deterministic core
        self.rssm = RSSMCore(
            deter_size=deter_size,
            stoch_size=stoch_size,
            hidden_size=rssm_hidden,
        )

        # Prior / Posterior (only used if use_stochastic_latent=True)
        if self.use_stochastic_latent:
            self.prior = Prior(deter_size=deter_size, stoch_size=stoch_size, hidden_size=rssm_hidden)
            self.posterior = Posterior(deter_size=deter_size, stoch_size=stoch_size, hidden_size=rssm_hidden)
        else:
            self.prior = None
            self.posterior = None

        # Decoder: (h,z) → reconstructed obs
        self.decoder = ObsDecoder(
            deter_size=deter_size,
            stoch_size=stoch_size,
            hidden_size=decoder_hidden,
            obs_shape=self.flat_obs_dim,
        )

        # Reward head: (h,z) → scalar reward
        self.reward_head = RewardHead(
            deter_size=deter_size,
            stoch_size=stoch_size,
            hidden_size=reward_hidden,
        )

        self.to(self.device)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_state(self, batch_size: int) -> WorldModelState:
        h0 = torch.zeros(batch_size, self.deter_size, device=self.device)
        z0 = torch.zeros(batch_size, self.stoch_size, device=self.device)
        return WorldModelState(h=h0, z=z0)

    # ------------------------------------------------------------------
    # Single-step observation update: (h,z) + obs_t → new state, recon, reward, KL
    # ------------------------------------------------------------------
    def observe_step(
        self,
        prev: WorldModelState,
        obs: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        prev: WorldModelState at t-1
        obs:  (B, flat_obs_dim) already flattened by env wrapper

        Returns dict with:
            - state: WorldModelState at t
            - recon: reconstructed obs (B, flat_obs_dim)
            - reward_pred: predicted reward (B, 1)
            - kl: KL divergence (scalar tensor) or 0 for Dreamer-Lite
        """
        # Encode observation
        embed = self.encoder(obs)

        if self.use_stochastic_latent:
            assert self.posterior is not None and self.prior is not None, (
                "Posterior and Prior must be defined for stochastic latent"
            )

            # Posterior conditioned on h_{t-1}, embed_t
            post = self.posterior(prev.h, embed)
            z = post["z"]

            # Prior from h_{t-1}
            prior = self.prior(prev.h)

            # Deterministic transition
            h = self.rssm(prev.h, z)

            state = WorldModelState(h=h, z=z, prior_stats=prior, post_stats=post)

            # KL between posterior and prior
            kl = self.kl_divergence(post, prior)
        else:
            # Dreamer-Lite: no stochastic latent, z ≡ 0, no KL
            z = torch.zeros_like(prev.z)
            h = self.rssm(prev.h, z)
            state = WorldModelState(h=h, z=z)
            kl = torch.zeros((), device=self.device)

        # Reconstruction and reward prediction
        recon = self.decoder(state.h, state.z)
        reward_pred = self.reward_head(state.h, state.z)

        return {
            "state": state,
            "recon": recon,
            "reward_pred": reward_pred,
            "kl": kl,
        }

    # ------------------------------------------------------------------
    # Imagination step: (h,z) → prior, sample z, next h
    # ------------------------------------------------------------------
    def imagine_step(self, prev: WorldModelState) -> WorldModelState:
        """
        Used for imagination rollouts (no real observations).
        """
        if self.use_stochastic_latent:
            assert self.prior is not None, "Prior must be defined for stochastic latent"
            prior = self.prior(prev.h)
            z = prior["z"]
            h = self.rssm(prev.h, z)
            return WorldModelState(h=h, z=z, prior_stats=prior, post_stats=None)
        else:
            z = torch.zeros_like(prev.z)
            h = self.rssm(prev.h, z)
            return WorldModelState(h=h, z=z)

    # ------------------------------------------------------------------
    # KL divergence between posterior and prior (diagonal Gaussians)
    # ------------------------------------------------------------------
    @staticmethod
    def kl_divergence(post: Dict[str, torch.Tensor], prior: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        KL(q || p) for diagonal Gaussians.
        post / prior: dict with 'mean' and 'std'
        Returns scalar tensor.
        """
        mean_q, std_q = post["mean"], post["std"]
        mean_p, std_p = prior["mean"], prior["std"]

        var_q = std_q**2
        var_p = std_p**2

        # KL per dimension
        kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5

        return kl.sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def reconstruct(self, state: WorldModelState) -> torch.Tensor:
        return self.decoder(state.h, state.z)

    def predict_reward(self, state: WorldModelState) -> torch.Tensor:
        return self.reward_head(state.h, state.z)
