"""Event-conditioned categorical scores with matched persistence baselines."""

from typing import Any

import torch


def previous_label_pairs(targets, observed=None, *, is_first=None, is_last=None):
    """Temporal inputs use (..., batch, time, entities); lower ranks are static."""
    valid = torch.ones_like(targets, dtype=torch.bool) if observed is None else observed.bool()
    if valid.shape != targets.shape:
        raise ValueError("Observation masks must match targets")
    previous = torch.zeros_like(targets)
    paired = torch.zeros_like(valid)
    if targets.ndim < 3:
        if is_first is not None or is_last is not None:
            raise ValueError("Temporal flags require explicit batch/time/entity dimensions")
        return previous, paired, paired.clone()
    previous[..., 1:, :] = targets[..., :-1, :]
    pair = valid[..., 1:, :] & valid[..., :-1, :]
    for flag, current in ((is_first, True), (is_last, False)):
        if flag is not None:
            if flag.shape != targets.shape[:-1]:
                raise ValueError("Episode flags must match batch/time dimensions")
            pair &= ~(flag[..., 1:] if current else flag[..., :-1]).bool().unsqueeze(-1)
    paired[..., 1:, :] = pair
    changed = paired & (targets != previous)
    return previous, paired, changed


def accuracy_counts(correct, mask) -> dict[str, torch.Tensor]:
    count = mask.sum()
    result = {"count": count, "correct": (correct & mask).sum()}
    if count > 0:
        result["accuracy"] = result["correct"].float() / count
    return result


def classification_report(predictions, targets, mask, classes) -> dict[str, Any]:
    result: dict[str, Any] = dict(accuracy_counts(predictions == targets, mask))
    p, y = predictions[mask], targets[mask]
    if (
        (p < 0).any()
        or (p >= classes).any()
        or (y < 0).any()
        or (y >= classes).any()
        or not torch.isfinite(p).all()
        or not torch.isfinite(y).all()
        or (p != p.round()).any()
        or (y != y.round()).any()
    ):
        raise ValueError("Observed prediction/target outside categorical vocabulary")
    matrix = torch.bincount(y.long() * classes + p.long(), minlength=classes * classes).reshape(classes, classes)
    result["confusion"] = matrix
    result["per_class"] = {}
    for category in range(classes):
        tp = matrix[category, category]
        support = matrix[category].sum()
        predicted = matrix[:, category].sum()
        fp, fn = predicted - tp, support - tp
        negatives = matrix.sum() - support
        values = {
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": negatives - fp,
        }
        if support > 0:
            values["recall"] = tp.float() / support
        if negatives > 0:
            values["false_positive_rate"] = fp.float() / negatives
        result["per_class"][category] = values
    return result


@torch.no_grad()
def categorical_event_scores(predictions, targets, classes, *, observed=None, is_first=None, is_last=None):
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must align")
    valid = torch.ones_like(targets, dtype=torch.bool) if observed is None else observed.bool()
    previous, paired, changed = previous_label_pairs(targets, valid, is_first=is_first, is_last=is_last)
    return {
        "all_observed": classification_report(predictions, targets, valid, classes),
        "paired_model": classification_report(predictions, targets, paired, classes),
        "persistence": classification_report(previous, targets, paired, classes),
        "changed_model": classification_report(predictions, targets, changed, classes),
        "changed_persistence": classification_report(previous, targets, changed, classes),
        "persistent_model": classification_report(predictions, targets, paired & ~changed, classes),
        "persistent_baseline": classification_report(previous, targets, paired & ~changed, classes),
    }


def json_metrics(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {key: json_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metrics(item) for item in value]
    return value
