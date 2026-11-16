"""Action-conditioned prior rollouts anchored in real posterior states."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.models.world_model import WorldModelState


def rollout_anchors(is_first, is_last, horizon, max_anchors=64, seed=42):
    """Choose common anchors eligible through the entire requested horizon."""
    if is_first.shape != is_last.shape or is_first.ndim != 2 or horizon < 1 or max_anchors < 1:
        raise ValueError("Need matching (batch,time) flags and positive limits")
    B, T = is_first.shape
    if horizon >= T:
        return torch.empty(0, 2, dtype=torch.long)
    valid = torch.ones(B, T - horizon, dtype=torch.bool, device=is_first.device)
    for step in range(1, horizon + 1):
        valid &= ~is_first[:, step : step + T - horizon].bool()
        valid &= ~is_last[:, step - 1 : step - 1 + T - horizon].bool()
    candidates = valid.nonzero().cpu()
    selected = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed))[:max_anchors]
    return candidates[selected]


def validate_anchors(anchors, is_first, is_last, horizon):
    if (
        anchors.ndim != 2
        or anchors.shape[1] != 2
        or len(anchors) == 0
        or anchors.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Need nonempty (anchor,batch/time) index pairs")
    if horizon < 1 or is_first.shape != is_last.shape or is_first.ndim != 2:
        raise ValueError("Invalid horizon or episode flags")
    B, T = is_first.shape
    if (anchors < 0).any() or (anchors[:, 0] >= B).any() or (anchors[:, 1] + horizon >= T).any():
        raise ValueError("Anchor endpoint outside trajectory")
    b, t = anchors[:, 0].to(is_first.device), anchors[:, 1].to(is_first.device)
    for step in range(1, horizon + 1):
        if is_first[b, t + step].any() or is_last[b, t + step - 1].any():
            raise ValueError("Rollout crosses an episode boundary")


def endpoint_pair(values, anchors, horizon):
    """Pack anchor and endpoint as a two-observation sequence for scoring."""
    b, t = anchors[:, 0].to(values.device), anchors[:, 1].to(values.device)
    return torch.stack([values[b, t], values[b, t + horizon]], dim=1)


@torch.no_grad()
def action_conditioned_rollout(
    world, posterior_h, posterior_z, actions, anchors, horizons, *, is_first=None, is_last=None, sampled=False, seed=42
):
    """Use only anchor posteriors and recorded incoming actions t+1...t+H.

    No future encoder/posterior calls or policy action selection occur. Each
    subsequent latent is sampled/mode-selected from the model's prior.
    """
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("Horizons must be positive")
    if posterior_h.ndim != 3 or posterior_z.ndim != 4 or posterior_h.shape[:2] != posterior_z.shape[:2]:
        raise ValueError("Posterior arrays must have matching batch/time dimensions")
    B, T = posterior_h.shape[:2]
    first = torch.zeros(B, T, dtype=torch.bool) if is_first is None else is_first.bool()
    last = torch.zeros(B, T, dtype=torch.bool) if is_last is None else is_last.bool()
    if first.shape != (B, T) or last.shape != (B, T):
        raise ValueError("Episode flags must match posterior batch/time dimensions")
    validate_anchors(anchors, first, last, max(horizons))
    if actions.shape[:2] != (B, T):
        raise ValueError("Actions must align with posterior batch/time dimensions")
    device = next(world.parameters()).device
    b, t = anchors[:, 0].to(device), anchors[:, 1].to(device)
    actions = actions.to(device)
    if actions.ndim == 2:
        action_dim = world.net_cfg.action_dim
        if action_dim is None:
            raise ValueError("Discrete actions require action_dim")
        actions = F.one_hot(actions.long(), action_dim).float()
    if actions.ndim != 3 or actions.shape[-1] != world.net_cfg.action_dim:
        raise ValueError("Actions must be indices or correctly sized one-hot vectors")
    state = WorldModelState(posterior_h.to(device)[b, t].detach(), posterior_z.to(device)[b, t].detach())
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    old_mode = world.prior.deterministic_latent_for_tests
    try:
        world.prior.deterministic_latent_for_tests = not sampled
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            outputs = {}
            for step in range(1, max(horizons) + 1):
                state = world.imagine_step(
                    state, nn.Identity(), action=actions[b, t + step], deterministic_imagination=not sampled
                )
                if step in horizons:
                    outputs[step] = state.detach()
            return outputs
    finally:
        world.prior.deterministic_latent_for_tests = old_mode
