from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch

from dreamerrl.models.world_model import WorldModel


def imagination_rollout(
    world_model: WorldModel,
    actor: Optional[torch.nn.Module],
    critic: Optional[torch.nn.Module],
    state: Any,
    horizon: int,
    with_values: bool = True,
    with_actions: bool = True,
    no_grad: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Tiny rollout inspector.

    Rolls out the world model for `horizon` steps starting from `state`
    and optionally annotates with values and actions.

    Returns a dict with:
        "h":      (T, B, deter_size)
        "z":      (T, B, stoch_size)
        "value":  (T, B) or None
        "action": (T, B) or None
    """
    device = next(world_model.parameters()).device
    start_state = world_model._ensure_state(state).to(device)

    ctx = torch.no_grad() if no_grad else nullcontext()
    with ctx:
        states = world_model.imagination_rollout(start_state, horizon=horizon)

        h = torch.stack([s.h for s in states], dim=0)
        z = torch.stack([s.z for s in states], dim=0)

        value: Optional[torch.Tensor] = None
        action: Optional[torch.Tensor] = None

        if critic is not None and with_values:
            # critic expects (T, B, ...) tensors
            value = critic(h, z).squeeze(-1)

        if actor is not None and with_actions:
            logits = actor(h, z)  # (T, B, A)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()  # (T, B)

    return {
        "h": h,
        "z": z,
        "value": value,
        "action": action,
    }


def imagine_trajectory(
    world_model: WorldModel,
    actor: Optional[torch.nn.Module],
    critic: Optional[torch.nn.Module],
    state: Any,
    horizon: int,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Backwards‑compatible wrapper used by tests.

    Delegates to `imagination_rollout` with no values/actions by default.
    """
    return imagination_rollout(
        world_model=world_model,
        actor=actor,
        critic=critic,
        state=state,
        horizon=horizon,
        with_values=False,
        with_actions=False,
        no_grad=True,
    )
