from __future__ import annotations

import torch
import torch.nn.functional as F

from dreamerrl.models.glyph_model import GlyphDecoder
from dreamerrl.utils.transforms import symlog
from dreamerrl.utils.types import WorldModelMetrics


def observe_batch(world_model, batch):
    """Replay is batch-major; action[t] led to obs[t]."""
    device = next(world_model.parameters()).device
    obs = batch.get("obs", batch.get("state"))
    if obs is None:
        raise KeyError("Batch must contain 'obs' or 'state'")
    obs = obs.to(device)
    B, L, _ = obs.shape
    state = world_model.init_state(B)
    if "burn_in_obs" in batch:
        # Reconstruct with current weights, then truncate gradients at the window.
        with torch.no_grad():
            for t in range(batch["burn_in_obs"].shape[1]):
                action = batch["burn_in_action"][:, t].to(device)
                if action.ndim == 1:
                    action = F.one_hot(action.long(), world_model.net_cfg.action_dim).float()
                post = world_model.observe_step(
                    state,
                    batch["burn_in_obs"][:, t].to(device),
                    action,
                    is_first=batch["burn_in_is_first"][:, t].to(device),
                )["post"]
                valid = batch["burn_in_valid"][:, t].to(device).unsqueeze(-1)
                state.h = torch.where(valid, post.h, state.h)
                state.z = torch.where(valid.unsqueeze(-1), post.z, state.z)
        state = state.detach()
    posts = []
    for t in range(L):
        action = batch["action"][:, t].to(device)
        if action.ndim == 1:
            action = F.one_hot(action.long(), world_model.net_cfg.action_dim).float()
        first = batch.get("is_first")
        out = world_model.observe_step(
            state,
            obs[:, t],
            action,
            is_first=None if first is None else first[:, t].to(device),
        )
        state = out["post"]
        posts.append(state.post_stats)
    return posts


def world_model_training_step(world_model, batch, kl_scale=None) -> WorldModelMetrics:
    device = next(world_model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    obs = batch.get("obs", batch.get("state"))
    B, L, _ = obs.shape
    batch["action_dim"] = world_model.net_cfg.action_dim
    posts = observe_batch(world_model, batch)
    h = torch.stack([s["h"] for s in posts], dim=1).reshape(B * L, -1)
    z = torch.stack([s["z"] for s in posts], dim=1).reshape(
        B * L,
        world_model.latent.num_classes,
        world_model.latent.stoch_size,
    )
    decoded = world_model.decoder(h, z)
    reward_logits, _ = world_model.reward_heads(h, z)
    cont_logits = world_model.continue_head(h, z).reshape(B, L, 1)
    cont_target = 1.0 - batch.get("is_terminal", torch.zeros_like(batch["reward"])).float()
    reconstruction_metrics = {}
    if isinstance(world_model.decoder, GlyphDecoder):
        decoder = world_model.decoder
        cell_loss = decoder.cell_loss(decoded, obs)
        changed = torch.zeros_like(cell_loss, dtype=torch.bool)
        changed[:, 1:] = obs[:, 1:, : decoder.cells] != obs[:, :-1, : decoder.cells]
        if "is_first" in batch:
            changed[:, 1:] &= ~batch["is_first"][:, 1:].bool().unsqueeze(-1)
        if "is_last" in batch or "done" in batch:
            changed[:, 1:] &= ~batch.get("is_last", batch.get("done"))[:, :-1].bool().unsqueeze(-1)
        weights = 1 + changed.float() * (world_model.cfg.glyph_changed_weight - 1)
        recon_loss = (cell_loss * weights).sum() / weights.sum()
        clock = decoded["clock"]
        if clock is not None:
            recon_loss = recon_loss + F.mse_loss(clock.reshape(B, L), obs[..., -1])
        with torch.no_grad():
            correct = decoder.predict(decoded).reshape(B, L, -1) == obs[..., : decoder.cells].long()
            reconstruction_metrics = {"glyph_accuracy": correct.float().mean(), "glyph_changed_cells": changed.sum()}
            if changed.any():
                reconstruction_metrics["glyph_changed_loss"] = cell_loss.detach()[changed].mean()
                reconstruction_metrics["glyph_changed_accuracy"] = correct[changed].float().mean()
    else:
        recon_loss = F.mse_loss(decoded.reshape(B, L, -1), symlog(obs))
    reward_loss = world_model.reward_heads.main.loss_from_logits(
        reward_logits.reshape(B, L, -1),
        batch["reward"],
    )
    cont_loss = world_model.continue_head.loss_from_logits(cont_logits, cont_target)
    aux_losses = []
    if not world_model.cfg.disable_aux_losses:
        for cfg in world_model.aux_objectives:
            head = world_model.aux_heads[cfg.name]
            logits = head(h, z).reshape(B, L, -1)
            target = cfg.fn(batch, world_model.net_cfg.discount).detach().float()
            if target.shape != logits.shape:
                raise ValueError(f"{cfg.name}: target {target.shape} != prediction {logits.shape}")
            if cfg.name == "novelty":
                # No future observation exists at the sequence boundary.
                valid = torch.ones(B, max(L - 1, 0), dtype=torch.bool, device=device)
                if "is_first" in batch:
                    valid &= ~batch["is_first"][:, 1:].bool()
                if "is_last" in batch or "done" in batch:
                    valid &= ~batch.get("is_last", batch.get("done"))[:, :-1].bool()
                loss = (
                    head.loss_from_logits(logits[:, :-1][valid], symlog(target[:, :-1][valid]))
                    if valid.any()
                    else logits.sum() * 0
                )
            else:
                if cfg.name == "resource":
                    target = symlog(target)
                loss = head.loss_from_logits(logits, target)
            aux_losses.append(loss)
    kl_dyn = torch.stack([p["kl_dyn"] for p in posts], dim=1).mean()
    kl_rep = torch.stack([p["kl_rep"] for p in posts], dim=1).mean()
    cfg = world_model.cfg.kl_cfg
    total = recon_loss + reward_loss + cont_loss
    total = total + (cfg.scale if kl_scale is None else kl_scale) * (
        cfg.dynamics_scale * kl_dyn + cfg.representation_scale * kl_rep
    )
    if aux_losses:
        total = total + world_model.cfg.aux_reward_scale * torch.stack(aux_losses).mean()
    return WorldModelMetrics(
        total_loss=total,
        recon_loss=recon_loss,
        reward_loss=reward_loss,
        cont_loss=cont_loss,
        raw_kl_dyn=torch.stack([p["raw_kl_dyn"].sum(-1) for p in posts]).mean(),
        raw_kl_rep=torch.stack([p["raw_kl_rep"].sum(-1) for p in posts]).mean(),
        kl_dyn=kl_dyn,
        kl_rep=kl_rep,
        aux_losses=aux_losses,
        reconstruction_metrics=reconstruction_metrics,
    )
