from __future__ import annotations

import torch
from torch.distributions import Categorical

from dreamerrl.models.world_model import WorldModelState
from dreamerrl.training.core.imagination import imagine_trajectory_for_training
from dreamerrl.training.core.lambda_return import lambda_return
from dreamerrl.training.core.world_model_update import observe_batch
from dreamerrl.utils.transforms import symlog
from dreamerrl.utils.twohot import twohot_encode


def actor_critic_update(
    world_model,
    actor,
    critic,
    batch,
    imagination_horizon,
    discount,
    lam,
    deterministic_imagination=False,
    slow_critic=None,
    slow_critic_scale=1.0,
    entropy_scale=3e-4,
):
    # Use every replay posterior as an imagination start, without model gradients.
    with torch.no_grad():
        posts = observe_batch(world_model, batch)
        h = torch.stack([p["h"] for p in posts], dim=1)
        z = torch.stack([p["z"] for p in posts], dim=1)
        B, L = h.shape[:2]
        start = WorldModelState(h.reshape(B * L, -1), z.reshape(B * L, *z.shape[2:]))
        terminal = batch.get("is_terminal", torch.zeros(B, L, device=h.device))
        start_weight = 1.0 - terminal.to(h.device).float().reshape(-1)
    traj = imagine_trajectory_for_training(
        world_model,
        actor,
        critic,
        start,
        imagination_horizon,
        deterministic_imagination,
    )
    h, z = traj["h"], traj["z"]
    T, N = h.shape[:2]
    h_flat, z_flat = h.reshape(T * N, -1), z.reshape(T * N, *z.shape[2:])
    with torch.no_grad():
        values = critic.readout(critic(h_flat, z_flat)).reshape(T, N)
        value_seq = torch.cat([values, traj["bootstrap_value"].unsqueeze(0)])
        discounts = discount * traj["continue"]
        returns = lambda_return(traj["reward"], value_seq, discounts, lam)
        weights = torch.cumprod(torch.cat([start_weight.unsqueeze(0), discounts[:-1]]), dim=0)
        live_returns = returns[weights > 0].float()
        if live_returns.numel() == 0:
            live_returns = torch.zeros(1, device=h.device)
        lo, hi = torch.quantile(live_returns, torch.tensor([0.05, 0.95], device=h.device))
        if hasattr(actor, "return_low"):
            if not actor.return_norm_initialized:
                actor.return_low.copy_(lo)
                actor.return_high.copy_(hi)
                actor.return_norm_initialized.fill_(True)
            else:
                actor.return_low.lerp_(lo, 0.01)
                actor.return_high.lerp_(hi, 0.01)
            lo, hi = actor.return_low, actor.return_high
        advantage = (returns - values) / (hi - lo).clamp_min(1.0)
    critic_logits = critic(h_flat, z_flat).reshape(T, N, -1)
    # Keep per-example weights while reusing the head's two-hot target convention.
    targets = twohot_encode(symlog(returns), critic.bin_values)
    critic_loss = -(weights * (targets * critic_logits.log_softmax(-1)).sum(-1)).mean()
    if slow_critic is not None:
        with torch.no_grad():
            slow_values = slow_critic.readout(slow_critic(h_flat, z_flat)).reshape(T, N)
            slow_targets = twohot_encode(symlog(slow_values), critic.bin_values)
        critic_loss = (
            critic_loss - slow_critic_scale * (weights * (slow_targets * critic_logits.log_softmax(-1)).sum(-1)).mean()
        )
    dist = Categorical(logits=actor(h_flat, z_flat).reshape(T, N, -1))
    actor_loss = -(weights * (dist.log_prob(traj["action"]) * advantage + entropy_scale * dist.entropy())).mean()
    return actor_loss, critic_loss
