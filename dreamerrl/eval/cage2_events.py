"""Per-class and changed-host-field scores for CAGE2 observation labels."""

import torch

from dreamerrl.env.cage2.categorical_schema import CAGE2_FIELDS
from dreamerrl.eval.field_events import categorical_event_scores


@torch.no_grad()
def score_host_fields(predictions, targets, observed, *, is_first=None, is_last=None):
    results = {}
    for field in CAGE2_FIELDS:
        report = categorical_event_scores(
            predictions[field.name],
            targets[field.name],
            field.num_classes,
            observed=observed[field.name],
            is_first=is_first,
            is_last=is_last,
        )
        for section in report.values():
            section["per_class"] = {field.labels[index]: scores for index, scores in section["per_class"].items()}
            section["confusion_labels"] = field.labels
        results[field.name] = report
    return results
