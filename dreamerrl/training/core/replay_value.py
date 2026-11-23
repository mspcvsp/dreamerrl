"""Recorded-reward value targets with destination-aligned replay transitions."""

import torch

from dreamerrl.utils.transforms import symlog
from dreamerrl.utils.twohot import twohot_encode


@torch.no_grad()
def replay_value_targets(batch, values, discount, lam):
    """V(obs[t]) targets use reward[t+1], never reward[t].

    Actual terminals cut bootstrapping. Truncations cut the lambda trace while
    retaining the final-observation bootstrap. Reset crossings have no target.
    The final replay observation is bootstrap-only.
    """
    reward = batch["reward"].to(values.device)
    terminal = batch.get("is_terminal", torch.zeros_like(reward)).to(values.device).bool()
    last = batch.get("is_last", batch.get("done", terminal)).to(values.device).bool()
    first = batch.get("is_first", torch.zeros_like(terminal)).to(values.device).bool()
    valid = ~(last[:, :-1] | terminal[:, :-1] | first[:, 1:])
    targets = torch.zeros_like(values[:, :-1])
    next_return = values[:, -1]
    for t in reversed(range(values.shape[1] - 1)):
        bootstrap = values[:, t + 1]
        # Do not let rewards from the next episode leak through a trace.
        boundary = last[:, t + 1] | terminal[:, t + 1]
        if t + 2 < values.shape[1]:
            boundary = boundary | first[:, t + 2]
        traced = torch.where(boundary, bootstrap, next_return)
        next_return = reward[:, t + 1] + discount * (~terminal[:, t + 1]).float() * (
            (1 - lam) * bootstrap + lam * traced
        )
        targets[:, t] = next_return
    return targets, valid


def replay_value_loss(critic, logits, batch, discount, lam):
    targets, valid = replay_value_targets(batch, critic.readout(logits.detach()), discount, lam)
    if not valid.any():
        return logits.sum() * 0, {}
    target_probs = twohot_encode(symlog(targets), critic.bin_values)
    loss = -(target_probs * logits[:, :-1].log_softmax(-1)).sum(-1)[valid].mean()
    with torch.no_grad():
        values = critic.readout(logits.detach())[:, :-1]
        metrics = {
            "replay_value_loss": loss.detach(),
            "replay_value_target_mean": targets[valid].mean(),
            "replay_value_target_mae": (values - targets).abs()[valid].mean(),
            "replay_value_targets": valid.sum(),
        }
    return loss, metrics
