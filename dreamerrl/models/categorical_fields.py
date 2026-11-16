"""Categorical fields over entities, with explicit masking and loss budgets."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class CategoricalField:
    name: str
    num_classes: int
    labels: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name or "." in self.name or self.num_classes < 2:
            raise ValueError("Categorical fields need a module-safe name and at least two classes")
        if self.labels and (len(self.labels) != self.num_classes or len(set(self.labels)) != len(self.labels)):
            raise ValueError("Field labels must be unique and match num_classes")


@dataclass(frozen=True)
class CategoricalLossConfig:
    field_weight: float = 1.0
    balance: str = "weighted"
    salient_weight: float = 1.0
    salient_fraction: float = 0.5
    class_weights: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.balance not in ("weighted", "groups"):
            raise ValueError("balance must be weighted or groups")
        if not math.isfinite(self.field_weight) or self.field_weight < 0:
            raise ValueError("field_weight must be finite and nonnegative")
        if not math.isfinite(self.salient_weight) or self.salient_weight <= 0:
            raise ValueError("salient_weight must be finite and positive")
        if not math.isfinite(self.salient_fraction) or not 0 < self.salient_fraction < 1:
            raise ValueError("salient_fraction must be between 0 and 1")
        if self.balance == "groups" and self.salient_weight != 1:
            raise ValueError("groups uses salient_fraction; salient_weight must stay 1")
        if self.class_weights is not None and (
            not self.class_weights
            or not all(math.isfinite(w) and w >= 0 for w in self.class_weights)
            or not any(w > 0 for w in self.class_weights)
        ):
            raise ValueError("class_weights must be finite, nonnegative, and not all zero")


@dataclass
class CategoricalFieldLoss:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class CategoricalFieldHead(nn.Linear):
    """Linear categorical head; weight/bias checkpoint keys match nn.Linear."""

    def __init__(self, feature_dim: int, field: CategoricalField, chunk_size: int = 1024):
        super().__init__(feature_dim, field.num_classes)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.field = field
        self.chunk_size = chunk_size

    def per_entity_loss(self, features, targets, observed=None):
        if features.shape[:-1] != targets.shape:
            raise ValueError("Categorical targets must match feature entity dimensions")
        mask = observed.bool() if observed is not None else None
        if mask is not None and mask.shape != targets.shape:
            raise ValueError("Observation mask must match target shape")
        selected = targets.flatten() if mask is None else targets[mask]
        if (
            not torch.isfinite(selected).all()
            or (selected != selected.round()).any()
            or (selected < 0).any()
            or (selected >= self.out_features).any()
        ):
            raise ValueError(f"Invalid observed target for {self.field.name}")
        flat = features.reshape(-1, self.in_features)
        indices = mask.flatten().nonzero(as_tuple=True)[0] if mask is not None else None
        if selected.numel() == 0:
            zero = flat[:0].sum() + self.weight.sum() * 0
            return features.new_zeros(targets.shape, dtype=torch.float32) + zero
        active = flat if indices is None else flat[indices]
        ids = selected.long()

        def cross_entropy(x, y):
            return F.cross_entropy(self(x).float(), y, reduction="none")

        chunks = []
        for x, y in zip(active.split(self.chunk_size), ids.split(self.chunk_size)):
            chunks.append(
                checkpoint(cross_entropy, x, y, use_reentrant=False) if torch.is_grad_enabled() else cross_entropy(x, y)
            )
        losses = torch.cat(chunks)
        if indices is None:
            return losses.reshape(targets.shape)
        return losses.new_zeros(targets.numel()).index_copy(0, indices, losses).reshape(targets.shape)

    @torch.no_grad()
    def predict(self, features):
        ids = [self(chunk).argmax(-1) for chunk in features.reshape(-1, self.in_features).split(self.chunk_size)]
        return torch.cat(ids).reshape(features.shape[:-1])


def categorical_field_loss(losses, targets, *, observed=None, salient=None, config=None) -> CategoricalFieldLoss:
    """Reduce CE with normalized class weights and optional fixed group budgets."""
    config = config or CategoricalLossConfig()
    valid = torch.ones_like(targets, dtype=torch.bool) if observed is None else observed.bool()
    important = torch.zeros_like(valid) if salient is None else salient.bool()
    if losses.shape != targets.shape or valid.shape != targets.shape or important.shape != targets.shape:
        raise ValueError("Field losses, targets, and masks must have matching shapes")
    weights = torch.ones_like(losses)
    if config.class_weights is not None:
        values = losses.new_tensor(config.class_weights)
        selected = targets[valid]
        if (selected < 0).any() or (selected >= len(values)).any():
            raise ValueError("Target outside class_weights")
        safe = torch.where(valid, targets, torch.zeros_like(targets)).long()
        weights = values[safe]
    active = valid & (weights > 0)
    high, low = active & important, active & ~important
    zero = losses[valid].sum() * 0

    def mean(mask):
        return (losses[mask] * weights[mask]).sum() / weights[mask].sum()

    high_mass = losses.new_zeros(())
    if not active.any():
        result = zero
    elif config.balance == "groups":
        if high.any() and low.any():
            result = config.salient_fraction * mean(high) + (1 - config.salient_fraction) * mean(low)
            high_mass = losses.new_tensor(config.salient_fraction)
        elif high.any():
            result, high_mass = mean(high), losses.new_ones(())
        else:
            result = mean(low)
    else:
        weights = weights * torch.where(important, config.salient_weight, 1.0)
        result = mean(active)
        high_mass = weights[high].sum() / weights[active].sum()
    metrics = {
        "observed_entities": valid.sum(),
        "active_entities": active.sum(),
        "balanced_ce": result.detach(),
        "weighted_loss": (config.field_weight * result).detach(),
        "salient_entities": (valid & important).sum(),
        "salient_loss_mass": high_mass,
        "field_weight": losses.new_tensor(config.field_weight),
        "unweighted_ce": losses[valid].detach().mean() if valid.any() else zero.detach(),
    }
    if high.any():
        metrics["salient_ce"] = losses[high].detach().mean()
    if low.any():
        metrics["other_ce"] = losses[low].detach().mean()
    return CategoricalFieldLoss(config.field_weight * result, metrics)


class CategoricalFieldDecoder(nn.Module):
    """Shared heads on (..., entity, feature) tokens carrying entity identity."""

    def __init__(self, feature_dim: int, fields: Sequence[CategoricalField], chunk_size=1024):
        super().__init__()
        if not fields or len({field.name for field in fields}) != len(fields):
            raise ValueError("Fields must be nonempty and uniquely named")
        self.heads = nn.ModuleDict(
            {field.name: CategoricalFieldHead(feature_dim, field, chunk_size) for field in fields}
        )

    def forward(self, features):
        return {name: head(features) for name, head in self.heads.items()}

    def loss(self, features, targets: Mapping, *, observed=None, salient=None, configs=None):
        losses, metrics = [], {}
        for name, head in self.heads.items():
            assert isinstance(head, CategoricalFieldHead)
            mask = observed.get(name) if observed is not None else None
            important = salient.get(name) if salient is not None else None
            config = configs.get(name, CategoricalLossConfig()) if configs is not None else CategoricalLossConfig()
            if config.class_weights is not None and len(config.class_weights) != head.out_features:
                raise ValueError(f"Class weights must match {name} vocabulary")
            ce = head.per_entity_loss(features, targets[name], mask)
            reduced = categorical_field_loss(ce, targets[name], observed=mask, salient=important, config=config)
            losses.append(reduced.loss)
            metrics[name] = reduced.metrics
        return torch.stack(losses).sum(), metrics
