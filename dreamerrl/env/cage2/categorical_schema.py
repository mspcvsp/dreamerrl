"""BlueTable observation fields; these describe observed beliefs, not hidden truth."""

from collections.abc import Mapping, Sequence

import torch

from dreamerrl.models.categorical_fields import CategoricalField, CategoricalFieldDecoder

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
