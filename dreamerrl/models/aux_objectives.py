"""Optional project objectives, separate from the DreamerV3 baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dreamerrl.utils.types import AuxObjectiveConfig


def novelty_target(batch, gamma):
    """Next-observation change proxy; training masks the unknown final target."""
    obs = batch["obs"]
    change = (obs[:, 1:] - obs[:, :-1]).abs().mean(dim=-1)
    return torch.cat([change, torch.zeros_like(obs[:, :1, 0])], dim=1).unsqueeze(-1)


def skill_target(batch, gamma):
    """Action prediction diagnostic; this does not discover temporally extended skills."""
    action = batch["action"]
    if action.ndim == 2:
        action = F.one_hot(action.long(), batch["action_dim"])
    return action.float()


def latent_cluster_skill_target(batch, gamma, z_t, num_skills, temperature=1.0, cluster_proj=None):
    """Experimental helper requiring a persistent projection owned by the caller.

    Not wired into training: option discovery needs its own learning objective
    and evaluation before it can replace the action prediction diagnostic.
    """
    if cluster_proj is None:
        raise ValueError("Latent clustering requires a persistent trained projection")
    B, L = z_t.shape[:2]
    logits = cluster_proj(z_t.reshape(B * L, -1)) / temperature
    return logits.softmax(-1).reshape(B, L, num_skills)


def reachability_target(batch, gamma):
    """Continuation diagnostic, duplicating the main continuation objective."""
    return (1.0 - batch["is_terminal"].float()).unsqueeze(-1)


def affordance_target(batch, gamma):
    """Supervise action availability only when the environment provides labels."""
    if "action_mask" not in batch:
        raise ValueError("Affordance supervision requires an explicit action_mask")
    return batch["action_mask"].float()


def resource_target(batch, gamma):
    """Reward diagnostic; genuine resource labels would provide a distinct signal."""
    return batch["reward"].unsqueeze(-1)


AUX_OBJECTIVES = {
    "novelty": AuxObjectiveConfig(name="novelty", fn=novelty_target),
    "skill": AuxObjectiveConfig(name="skill", fn=skill_target),
    "reachability": AuxObjectiveConfig(name="reachability", fn=reachability_target),
    "affordance": AuxObjectiveConfig(name="affordance", fn=affordance_target),
    "resource": AuxObjectiveConfig(name="resource", fn=resource_target),
}
