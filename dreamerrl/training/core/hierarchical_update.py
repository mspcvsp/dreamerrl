"""Extrinsic-reward hierarchical REINFORCE with primitive and semi-Markov value targets."""

import torch
from torch.distributions import Categorical

from dreamerrl.models.world_model import WorldModelState
from dreamerrl.utils.transforms import symlog
from dreamerrl.utils.twohot import twohot_encode

from .lambda_return import lambda_return
from .world_model_update import observe_batch


@torch.no_grad()
def imagine_hierarchy(world, actor, start, horizon, deterministic=False):
    device_type = start.h.device.type
    # Autocast caches made under no_grad would otherwise be reused by policy
    # recomputation in the enclosing training context, silently dropping gradients.
    with torch.autocast(
        device_type,
        enabled=torch.is_autocast_enabled(device_type),
        dtype=torch.get_autocast_dtype(device_type),
        cache_enabled=False,
    ):
        if horizon < 1:
            raise ValueError("Hierarchical imagination horizon must be positive")
        state = start
        policy = actor.initial_state(start.h.shape[0], start.h.device)
        rows = {name: [] for name in ("h", "z", "goal_id", "remaining", "decision", "action", "reward", "continue")}
        action_dim = world.net_cfg.action_dim
        for _ in range(horizon):
            chosen = actor.step(state.h, state.z, policy, deterministic=deterministic)
            policy = chosen.state
            for name, value in (
                ("h", state.h),
                ("z", state.z),
                ("goal_id", policy.goal_id),
                ("remaining", chosen.remaining_before_action),
                ("decision", chosen.manager_decision),
                ("action", chosen.action),
            ):
                rows[name].append(value)
            action = torch.nn.functional.one_hot(chosen.action, action_dim).float()
            state = world.imagine_step(state, actor.worker, action=action, deterministic_imagination=deterministic)
            reward = world.reward_heads.main(state.h, state.z)
            rows["reward"].append(world.reward_heads.main.readout(reward))
            rows["continue"].append(world.continue_head.readout(world.continue_head(state.h, state.z)))
        result = {name: torch.stack(values) for name, values in rows.items()}
        result.update(tail_h=state.h, tail_z=state.z, tail_goal_id=policy.goal_id, tail_remaining=policy.remaining)
        return result


@torch.no_grad()
def manager_returns(rewards, discounts, values, tail_value, interval, lam):
    """Macro rewards sum discounted primitive rewards, including a shortened final interval."""
    starts = list(range(0, len(rewards), interval))
    returns = []
    following = tail_value
    for start in reversed(starts):
        stop = min(start + interval, len(rewards))
        reward, product = torch.zeros_like(tail_value), torch.ones_like(tail_value)
        for step in range(start, stop):
            reward = reward + product * rewards[step]
            product = product * discounts[step]
        bootstrap = values[stop] if stop < len(rewards) else tail_value
        following = reward + product * ((1 - lam) * bootstrap + lam * following)
        returns.append(following)
    return torch.stack(list(reversed(returns)))


@torch.no_grad()
def normalized_advantage(policy, target, baseline, weights):
    live = target[weights > 0].float()
    if live.numel():
        lo, hi = torch.quantile(live, torch.tensor([0.05, 0.95], device=target.device))
        if not policy.return_norm_initialized:
            policy.return_low.copy_(lo)
            policy.return_high.copy_(hi)
            policy.return_norm_initialized.fill_(True)
        else:
            policy.return_low.lerp_(lo, 0.01)
            policy.return_high.lerp_(hi, 0.01)
    return (target - baseline) / (policy.return_high - policy.return_low).clamp_min(1)


def value_loss(head, logits, targets, weights, slow_values=None, slow_scale=1.0):
    target = twohot_encode(symlog(targets.detach()), head.bin_values)
    loss = -(weights * (target * logits.log_softmax(-1)).sum(-1)).mean()
    if slow_values is not None:
        slow_target = twohot_encode(symlog(slow_values.detach()), head.bin_values)
        loss = loss - slow_scale * (weights * (slow_target * logits.log_softmax(-1)).sum(-1)).mean()
    return loss


@torch.no_grad()
def goal_diagnostics(ids, weights, interval, embeddings):
    """Summarize only live imagined decisions and transitions."""
    live = weights > 0
    decisions = live[::interval]
    counts = torch.zeros(len(embeddings), device=ids.device, dtype=torch.long)
    counts.scatter_add_(0, ids[::interval].flatten(), decisions.flatten().long())
    probability = counts.float() / counts.sum().clamp_min(1)
    adjacent = live[1:] & live[:-1]
    boundary = adjacent[interval - 1::interval]
    repeated = ids[1:] == ids[:-1]
    # Live steps per observed interval include horizon/task-censored segments.
    lengths = torch.stack([part.sum(0) for part in live.split(interval)])
    norms = embeddings.detach().float().norm(dim=-1)
    result = dict(
        goal_entropy=-(probability * probability.clamp_min(1e-12).log()).sum(),
        goal_active_count=(counts > 0).sum(),
        goal_persistence=(repeated & adjacent).sum() / adjacent.sum().clamp_min(1),
        goal_boundary_repeat_rate=(repeated[interval - 1::interval] & boundary).sum()
        / boundary.sum().clamp_min(1),
        goal_boundary_pairs=boundary.sum(),
        goal_adjacent_pairs=adjacent.sum(),
        goal_configured_interval=ids.new_tensor(interval),
        goal_observed_interval_mean=live.sum() / decisions.sum().clamp_min(1),
        goal_full_intervals=(lengths == interval).sum(),
        goal_embedding_norm_mean=norms.mean(),
        goal_embedding_norm_max=norms.max(),
    )
    result.update({f"goal_{i}_decisions": count for i, count in enumerate(counts.unbind())})
    return result


def hierarchical_actor_critic_update(
    world,
    actor,
    critic,
    batch,
    horizon,
    discount,
    lam,
    deterministic=False,
    slow_critic=None,
    slow_scale=1.0,
    entropy_scale=3e-4,
    remaining_steps=None,
    diagnostics=None,
):
    with torch.no_grad():
        posts = observe_batch(world, batch)
        h, z = torch.stack([p["h"] for p in posts], 1), torch.stack([p["z"] for p in posts], 1)
        B, L = h.shape[:2]
        start = WorldModelState(h.reshape(B * L, -1), z.reshape(B * L, *z.shape[2:]))
        terminal = batch.get("is_terminal", torch.zeros(B, L, device=h.device))
        start_weight = 1 - terminal.to(h.device).float().reshape(-1)
        if remaining_steps is not None:
            if remaining_steps.shape != (B, L):
                raise ValueError("remaining_steps must have replay shape (B, L)")
            remaining_steps = remaining_steps.to(h.device).reshape(-1)
            start_weight = start_weight * (remaining_steps > 0)
    trajectory = imagine_hierarchy(world, actor, start, horizon, deterministic)
    h, z = trajectory["h"], trajectory["z"]
    T, N = h.shape[:2]
    flat_h, flat_z = h.reshape(T * N, -1), z.reshape(T * N, *z.shape[2:])
    ids, remaining = trajectory["goal_id"], trajectory["remaining"]
    starts = torch.arange(0, T, actor.interval, device=h.device)
    device_type = h.device.type
    with (
        torch.no_grad(),
        torch.autocast(
            device_type,
            enabled=torch.is_autocast_enabled(device_type),
            dtype=torch.get_autocast_dtype(device_type),
            cache_enabled=False,
        ),
    ):
        manager_values = critic.manager.readout(critic.manager(flat_h, flat_z)).reshape(T, N)
        worker_values = critic.worker.readout(
            critic.worker_logits(flat_h, flat_z, ids.flatten(), remaining.flatten())
        ).reshape(T, N)
        tail_manager = critic.manager.readout(critic.manager(trajectory["tail_h"], trajectory["tail_z"]))
        tail_worker = critic.worker.readout(
            critic.worker_logits(
                trajectory["tail_h"], trajectory["tail_z"], trajectory["tail_goal_id"], trajectory["tail_remaining"]
            )
        )
        tail_value = torch.where(trajectory["tail_remaining"] == 0, tail_manager, tail_worker)
        discounts = discount * trajectory["continue"]
        if remaining_steps is not None:
            offsets = torch.arange(1, T + 1, device=h.device)[:, None]
            discounts = discounts * (offsets < remaining_steps[None, :])
        weights = torch.cumprod(torch.cat((start_weight[None, :], discounts[:-1])), 0)
        # At a new goal boundary, bootstrap the expectation before selecting a goal.
        bootstrap_values = torch.where(trajectory["decision"], manager_values, worker_values)
        worker_target = lambda_return(
            trajectory["reward"], torch.cat((bootstrap_values, tail_value[None, :])), discounts, lam
        )
        manager_target = manager_returns(
            trajectory["reward"], discounts, manager_values, tail_value, actor.interval, lam
        )
        worker_advantage = normalized_advantage(actor.worker, worker_target, worker_values, weights)
        manager_advantage = normalized_advantage(actor.manager, manager_target, manager_values[starts], weights[starts])
    # Recompute actor-owned embeddings outside no_grad; critic never shares this table.
    goal = actor.goal_embedding(ids.flatten())
    worker_dist = Categorical(logits=actor.worker_logits(flat_h, flat_z, goal, remaining.flatten()).reshape(T, N, -1))
    manager_dist = Categorical(
        logits=actor.manager(h[starts].reshape(-1, h.shape[-1]), z[starts].reshape(-1, *z.shape[2:])).reshape(
            len(starts), N, -1
        )
    )
    worker_loss = -(
        weights
        * (worker_dist.log_prob(trajectory["action"]) * worker_advantage + entropy_scale * worker_dist.entropy())
    ).mean()
    manager_loss = -(
        weights[starts]
        * (manager_dist.log_prob(ids[starts]) * manager_advantage + entropy_scale * manager_dist.entropy())
    ).mean()
    worker_logits = critic.worker_logits(flat_h, flat_z, ids.flatten(), remaining.flatten()).reshape(T, N, -1)
    manager_logits = critic.manager(h[starts].reshape(-1, h.shape[-1]), z[starts].reshape(-1, *z.shape[2:])).reshape(
        len(starts), N, -1
    )
    slow_worker = slow_manager = None
    if slow_critic is not None:
        with torch.no_grad():
            slow_worker = slow_critic.worker.readout(
                slow_critic.worker_logits(flat_h, flat_z, ids.flatten(), remaining.flatten())
            ).reshape(T, N)
            slow_manager = slow_critic.manager.readout(
                slow_critic.manager(h[starts].reshape(-1, h.shape[-1]), z[starts].reshape(-1, *z.shape[2:]))
            ).reshape(len(starts), N)
    worker_critic_loss = value_loss(critic.worker, worker_logits, worker_target, weights, slow_worker, slow_scale)
    manager_critic_loss = value_loss(
        critic.manager, manager_logits, manager_target, weights[starts], slow_manager, slow_scale
    )
    if diagnostics is not None:
        diagnostics.update(goal_diagnostics(ids, weights, actor.interval, actor.goal_embedding.weight))
        diagnostics.update(
            manager_critic_loss=manager_critic_loss.detach(),
            worker_critic_loss=worker_critic_loss.detach(),
            manager_actor_loss=manager_loss.detach(),
            worker_actor_loss=worker_loss.detach(),
            manager_decisions=(weights[starts] > 0).sum(),
            worker_decisions=(weights > 0).sum(),
            imagined_return_mean=(weights * worker_target).sum() / weights.sum().clamp_min(1),
            manager_return_mean=(weights[starts] * manager_target).sum() / weights[starts].sum().clamp_min(1),
        )
    return worker_loss + manager_loss, worker_critic_loss + manager_critic_loss
