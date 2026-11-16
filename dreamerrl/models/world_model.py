from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.models.actor import act_in_imagination
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig

from .aux_heads import make_aux_heads
from .base_obs_encoder import BaseEncoder
from .categorical_fields import CategoricalLossConfig
from .categorical_kl import structured_kl
from .continue_head import ContinueHead
from .decoder import ObsDecoder
from .glyph_model import GlyphDecoder, GlyphEncoder
from .multi_reward_head import MultiRewardHead
from .obs_encoder import build_obs_encoder, get_flat_obs_dim
from .posterior import Posterior
from .prior import Prior
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

      • ObsEncoder: symlog‑MLP encoder
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

        self.encoder: BaseEncoder
        self.decoder: ObsDecoder | GlyphDecoder
        cells = 0
        if self.cfg.glyph_shape is not None:
            if len(self.cfg.glyph_shape) != 2 or min(self.cfg.glyph_shape) < 1:
                raise ValueError("Glyph grid dimensions must be positive (height, width)")
            cells = self.cfg.glyph_shape[0] * self.cfg.glyph_shape[1]
            if self.cfg.glyph_vocab_size < 2 or self.flat_obs_dim not in (cells, cells + 1):
                raise ValueError("Glyph model requires a valid vocabulary and matching grid dimensions")
            if self.cfg.glyph_changed_weight < 1 or not torch.isfinite(torch.tensor(self.cfg.glyph_changed_weight)):
                raise ValueError("glyph_changed_weight must be finite and at least 1")
            if any(
                not isinstance(glyph, int) or not 0 <= glyph < self.cfg.glyph_vocab_size
                for glyph in self.cfg.glyph_foreground_ids
            ):
                raise ValueError("Foreground glyph IDs must belong to the configured vocabulary")
            CategoricalLossConfig(
                balance=self.cfg.glyph_loss_balance,
                salient_weight=self.cfg.glyph_changed_weight,
                salient_fraction=self.cfg.glyph_salient_fraction,
            )
            self.encoder = GlyphEncoder(
                self.cfg.glyph_shape, self.cfg.glyph_vocab_size, self.embed_size, clock=self.flat_obs_dim == cells + 1
            ).to(self.device)
        else:
            self.encoder = build_obs_encoder(obs_space, embed_dim=self.embed_size).to(self.device)
        self.rssm: RSSMCore = RSSMCore(latent=latent, net=net).to(self.device)

        deterministic_latent_for_tests = dreamer_cfg.train.deterministic_env

        self.prior: Prior = Prior(
            latent=latent, net=net, deterministic_latent_for_tests=deterministic_latent_for_tests
        ).to(self.device)

        self.posterior: Posterior = Posterior(
            latent=latent, net=net, deterministic_latent_for_tests=deterministic_latent_for_tests
        ).to(self.device)

        if self.cfg.glyph_shape is not None:
            self.decoder = GlyphDecoder(
                latent=latent,
                net=net,
                shape=self.cfg.glyph_shape,
                vocab_size=self.cfg.glyph_vocab_size,
                clock=self.flat_obs_dim == cells + 1,
            ).to(self.device)
        else:
            self.decoder = ObsDecoder(latent=latent, net=net, output_dim=self.flat_obs_dim).to(self.device)

        self.reward_heads = MultiRewardHead(
            latent=latent,
            net=net,
            num_aux=0,
        ).to(self.device)

        self.continue_head: ContinueHead = ContinueHead(latent=latent, net=net).to(self.device)

        # -------------------------------------------------------------
        # Auxiliary heads (novelty, reachability, affordance, skill, resource)
        # -------------------------------------------------------------
        if self.cfg.disable_aux_losses:
            # Disable auxiliary losses entirely
            self.aux_heads = nn.ModuleDict({}).to(self.device)
            self.aux_objectives = []
        else:
            all_aux_heads = make_aux_heads(
                deter_size=latent.deter_size,
                z_dim=latent.z_dim,
                action_dim=net.action_dim or 0,
                num_skills=net.action_dim or 0,
                hidden=net.hidden_size,
            )

            self.aux_objectives = dreamer_cfg.world.aux_objectives or []

            self.aux_heads = nn.ModuleDict({cfg.name: all_aux_heads[cfg.name] for cfg in self.aux_objectives}).to(
                self.device
            )

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
        reward_main_logits, reward_aux_logits = self.reward_heads(h, z)
        cont_logits = self.continue_head(h, z)

        # Disable aux heads in reproducibility mode
        if self.cfg.disable_aux_losses:
            aux_logits = {}
        else:
            aux_logits = {name: head(h, z) for name, head in self.aux_heads.items()}

        kl_dict = structured_kl(q_probs=post_stats["probs"], p_probs=prior_stats["probs"], kl_cfg=self.cfg.kl_cfg)

        if self.probe:
            self.probe.wm_observe(embed, post_stats, prior_stats)
            self.probe.wm_kl(kl_dict)

        for key in ["raw_kl_dyn", "kl_dyn", "raw_kl_rep", "kl_rep", "kl_total"]:
            if not torch.isfinite(kl_dict[key]).all():
                raise ValueError(f"KL divergence {key} is not finite: {kl_dict[key]}")
            post_stats[key] = kl_dict[key]

        # NOTE:
        #   We return `reward_logits` for backward‑compatibility with the Dreamer‑V3 API and test suite. The training
        #   pipeline (imagination, actor‑critic update, value learning) all expect a single key named `reward_logits`
        #   that contains the *main* reward‑head logits.
        #
        #   Even though this model now supports multiple reward heads (`reward_main_logits`, `reward_aux_logits`), the
        #   legacy key must remain so downstream code continues to work without modification.
        #
        #   In short: `reward_logits` is an alias for the main reward head.
        return {
            "post": post,
            "prior": prior,
            "post_stats": post_stats,
            "prior_stats": prior_stats,
            "recon": recon,
            "reward_logits": reward_main_logits,
            "reward_main_logits": reward_main_logits,
            "reward_aux_logits": reward_aux_logits,
            "cont_logits": cont_logits,
            "aux_logits": aux_logits,
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
