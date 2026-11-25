"""Explicit manager/worker composition. Goal state belongs to a rollout, never the module."""

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.distributions import Categorical

from .actor import Actor
from .goal_autoencoder import GoalAutoencoder
from .value_head import ValueHead


def validate_hierarchy(goals, goal_dim, interval):
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (goals, goal_dim, interval))
        or goals < 2
    ):
        raise ValueError("Hierarchy requires goals>=2, goal_dim>=1, interval>=1 integers")


@dataclass
class HierarchyState:
    goal_id: torch.Tensor
    goal: torch.Tensor
    remaining: torch.Tensor

    def reset(self, mask):
        mask = mask.to(self.remaining.device).bool()
        if mask.shape != self.remaining.shape:
            raise ValueError("Hierarchy reset mask must have shape (batch,)")
        return HierarchyState(
            torch.where(mask, 0, self.goal_id),
            torch.where(mask[:, None], 0, self.goal),
            torch.where(mask, 0, self.remaining),
        )


@dataclass
class HierarchyStep:
    action: torch.Tensor
    logits: torch.Tensor
    state: HierarchyState
    manager_decision: torch.Tensor
    remaining_before_action: torch.Tensor


class HierarchicalActor(nn.Module):
    def __init__(
        self, *, latent, net, goals=8, goal_dim=16, interval=4, probe=None, director_goals=False, goal_progress=False,
    ):
        super().__init__()
        validate_hierarchy(goals, goal_dim, interval)
        self.manager = Actor(latent=latent, net=replace(net, action_dim=goals))
        self.goal_autoencoder = None
        self.goal_progress = goal_progress
        if director_goals:
            features = latent.deter_size + latent.stoch_size * latent.num_classes
            self.goal_autoencoder = GoalAutoencoder(features, goals, goal_dim)
            goal_dim = features
        else:
            self.goal_embedding = nn.Embedding(goals, goal_dim)
        self.num_goals, self.goal_dim, self.interval = goals, goal_dim, interval
        self.worker = Actor(latent=replace(latent, deter_size=latent.deter_size + goal_dim + 1), net=net, probe=probe)

    def forward(self, h, z):
        raise ValueError("Hierarchical policy requires explicit goal state: use step or policy_action")

    def act(self, state, deterministic=False) -> tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("Hierarchical policy requires explicit goal state: use step or policy_action")

    def initial_state(self, batch_size, device=None):
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        return HierarchyState(
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.zeros(batch_size, self.goal_dim, device=device, dtype=parameter.dtype),
            torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def decode_goals(self, ids):
        if self.goal_autoencoder is None:
            return self.goal_embedding(ids)
        with torch.no_grad(), torch.autocast(ids.device.type, enabled=False):
            return self.goal_autoencoder.decode_ids(ids)

    def worker_logits(self, h, z, goal, remaining):
        phase = remaining.to(h.dtype).unsqueeze(-1) / self.interval
        return self.worker(torch.cat((h, goal, phase), -1), z)

    @torch.no_grad()
    def step(self, h, z, state, *, reset=None, deterministic=False):
        if h.ndim != 2 or z.ndim != 3 or h.shape[0] != z.shape[0]:
            raise ValueError("Hierarchy expects h=(B,H), z=(B,K,C)")
        if state.remaining.shape != h.shape[:1] or state.goal.shape != (h.shape[0], self.goal_dim):
            raise ValueError("Hierarchy state batch/goal shape mismatch")
        if (
            state.goal_id.shape != state.remaining.shape
            or state.goal_id.dtype != torch.long
            or state.remaining.dtype != torch.long
        ):
            raise ValueError("Hierarchy goal IDs and remaining counters must be integer batch vectors")
        if (state.remaining < 0).any() or (state.remaining > self.interval).any():
            raise ValueError("Hierarchy remaining counter is outside its interval")
        if (state.goal_id < 0).any() or (state.goal_id >= self.num_goals).any():
            raise ValueError("Hierarchy goal ID is outside the codebook")
        if any(value.device != h.device for value in (z, state.goal_id, state.goal, state.remaining)):
            raise ValueError("Hierarchy features and policy state must use the same device")
        if reset is not None:
            state = state.reset(reset)
        goal_id, goal, remaining = state.goal_id.clone(), state.goal.clone(), state.remaining.clone()
        decision = remaining == 0
        if decision.any():
            logits = self.manager(h[decision], z[decision])
            selected = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
            goal_id[decision] = selected
            # Latch the vector, including across later actor/codebook optimizer updates.
            goal[decision] = self.decode_goals(selected).to(goal.dtype)
            remaining[decision] = self.interval
        logits = self.worker_logits(h, z, goal, remaining)
        action = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
        return HierarchyStep(action, logits, HierarchyState(goal_id, goal, remaining - 1), decision, remaining)


class HierarchicalCritic(nn.Module):
    def __init__(self, *, latent, net, goals=8, interval=4):
        super().__init__()
        validate_hierarchy(goals, 1, interval)
        self.num_goals, self.interval = goals, interval
        self.manager = ValueHead(latent=latent, net=net)
        self.worker = ValueHead(latent=replace(latent, deter_size=latent.deter_size + goals + 1), net=net)
        self.probe = None

    def worker_logits(self, h, z, goal_id, remaining):
        # Fixed IDs avoid sharing actor-owned trainable embeddings with value learning.
        goals = torch.nn.functional.one_hot(goal_id, self.num_goals).to(h.dtype)
        phase = remaining.to(h.dtype).unsqueeze(-1) / self.interval
        return self.worker(torch.cat((h, goals, phase), -1), z)


def policy_action(actor, state, policy_state=None, *, reset=None, deterministic=False):
    """Evaluation helper: flat policies preserve their existing act path."""
    if isinstance(actor, HierarchicalActor):
        if policy_state is None:
            policy_state = actor.initial_state(state.h.shape[0], state.h.device)
        result = actor.step(state.h, state.z, policy_state, reset=reset, deterministic=deterministic)
        return result.action, result.logits, result.state
    actions, logits = actor.act(state, deterministic=deterministic)
    return actions, logits, None
