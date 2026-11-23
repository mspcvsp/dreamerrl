"""BlueTable observation fields; these describe observed beliefs, not hidden truth."""

import math
from collections.abc import Mapping, Sequence

import torch

from dreamerrl.models.categorical_fields import CategoricalField, CategoricalFieldDecoder, CategoricalLossConfig

ACTIVITY = CategoricalField("Activity", 3, ("None", "Scan", "Exploit"))
COMPROMISED = CategoricalField("Compromised", 4, ("No", "Unknown", "User", "Privileged"))
CAGE2_FIELDS = (ACTIVITY, COMPROMISED)


def make_cage2_decoder(feature_dim: int) -> CategoricalFieldDecoder:
    """Caller supplies one identity-aware feature token per host."""
    return CategoricalFieldDecoder(feature_dim, CAGE2_FIELDS)


def encode_host_fields(observations: Mapping[str, Mapping[str, str | None]], host_order: Sequence[str]):
    """Explicit host ordering, with missing fields masked and Unknown kept valid."""
    if len(set(host_order)) != len(host_order):
        raise ValueError("host_order must contain unique identities")
    if set(observations) - set(host_order):
        raise ValueError("Observed host missing from host_order")
    targets, observed = {}, {}
    for field in CAGE2_FIELDS:
        values, masks = [], []
        for host in host_order:
            value = observations.get(host, {}).get(field.name)
            if value is None:
                values.append(-1)
                masks.append(False)
            elif value not in field.labels:
                raise ValueError(f"Invalid {field.name} category for {host}: {value}")
            else:
                values.append(field.labels.index(value))
                masks.append(True)
        targets[field.name] = torch.tensor(values, dtype=torch.long)
        observed[field.name] = torch.tensor(masks, dtype=torch.bool)
    return targets, observed


def salient_host_fields(targets, observed):
    """Observed non-default fields are salient; Unknown remains meaningful."""
    return {field.name: observed[field.name] & (targets[field.name] != 0) for field in CAGE2_FIELDS}


def informative_host_loss(
    targets, observed, *, is_first=None, is_last=None, activity_weight=2.0, compromise_change_weight=2.0
):
    """Modest weighting with nonzero supervision on persistent valid fields.

    Temporal targets use (..., batch, time, hosts). Single observations use
    (hosts,) or (batch, hosts); those shapes receive no temporal change bonus.
    Activity non-default classes receive activity_weight. Compromise receives
    extra weight only on observed within-episode transitions, including changes
    to/from Unknown. Missing observations and reset boundaries never imply changes.
    """
    if any(not math.isfinite(weight) or weight < 1 for weight in (activity_weight, compromise_change_weight)):
        raise ValueError("Informative-field multipliers must be finite and at least 1")
    for field in CAGE2_FIELDS:
        if targets[field.name].shape != observed[field.name].shape:
            raise ValueError("Observed masks must match host field targets")
    values, valid = targets[COMPROMISED.name], observed[COMPROMISED.name].bool()
    if values.shape != targets[ACTIVITY.name].shape or values.ndim < 1:
        raise ValueError("Host fields must share batch/time/host dimensions")
    if values.ndim < 3 and (is_first is not None or is_last is not None):
        raise ValueError("Temporal flags require explicit batch, time, host dimensions")
    changed = torch.zeros_like(valid)
    if values.ndim >= 3:
        pair = valid[..., 1:, :] & valid[..., :-1, :]
        if is_first is not None:
            if is_first.shape != values.shape[:-1]:
                raise ValueError("is_first must match the host field batch/time dimensions")
            pair &= ~is_first[..., 1:].bool().unsqueeze(-1)
        if is_last is not None:
            if is_last.shape != values.shape[:-1]:
                raise ValueError("is_last must match the host field batch/time dimensions")
            pair &= ~is_last[..., :-1].bool().unsqueeze(-1)
        changed[..., 1:, :] = pair & (values[..., 1:, :] != values[..., :-1, :])
    configs = {
        ACTIVITY.name: CategoricalLossConfig(class_weights=(1.0, activity_weight, activity_weight)),
        COMPROMISED.name: CategoricalLossConfig(salient_weight=compromise_change_weight),
    }
    salient = {ACTIVITY.name: torch.zeros_like(observed[ACTIVITY.name], dtype=torch.bool), COMPROMISED.name: changed}
    return configs, salient
