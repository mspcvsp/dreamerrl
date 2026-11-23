"""Per-class and changed-host-field scores for CAGE2 observation labels."""

import torch

from dreamerrl.env.cage2.categorical_schema import CAGE2_FIELDS
from dreamerrl.eval.field_events import categorical_event_scores, evidence_retention, previous_label_pairs


@torch.no_grad()
def score_host_fields(predictions, targets, observed, *, is_first=None, is_last=None, score_frames=None):
    results = {}
    for field in CAGE2_FIELDS:
        report = categorical_event_scores(
            predictions[field.name],
            targets[field.name],
            field.num_classes,
            observed=observed[field.name],
            is_first=is_first,
            is_last=is_last,
            score_frames=score_frames,
        )
        for section in report.values():
            section["per_class"] = {field.labels[index]: scores for index, scores in section["per_class"].items()}
            section["confusion_labels"] = field.labels
        results[field.name] = report
    return results


@torch.no_grad()
def compare_host_rollouts(posterior_predictions, rollout_predictions, targets, observed, anchors, *, is_first, is_last):
    """Compare endpoint posterior/prior predictions against anchor-label persistence.

    Targets/posterior predictions are (batch,time,hosts). Each horizon maps to
    prior predictions of shape (anchors,hosts). No host simulator is assumed.
    """
    from dreamerrl.eval.imagination_probe import endpoint_pair, validate_anchors

    if not rollout_predictions:
        raise ValueError("Need at least one rollout horizon")
    validate_anchors(anchors, is_first, is_last, max(rollout_predictions))
    frames = torch.tensor([False, True], device=targets[CAGE2_FIELDS[0].name].device).expand(len(anchors), 2)
    results = {}
    for horizon, prior_fields in rollout_predictions.items():
        truth = {name: endpoint_pair(value, anchors, horizon) for name, value in targets.items()}
        masks = {name: endpoint_pair(value, anchors, horizon) for name, value in observed.items()}
        posterior, prior = {}, {}
        for name in targets:
            endpoint = endpoint_pair(posterior_predictions[name], anchors, horizon)[:, 1]
            if prior_fields[name].shape != endpoint.shape:
                raise ValueError("Prior host predictions must align with anchors and hosts")
            posterior[name] = torch.stack([torch.zeros_like(endpoint), endpoint], 1)
            prior[name] = torch.stack([torch.zeros_like(endpoint), prior_fields[name]], 1)
        results[horizon] = {
            "posterior_reconstruction": score_host_fields(posterior, truth, masks, score_frames=frames),
            "open_loop_prior": score_host_fields(prior, truth, masks, score_frames=frames),
            "cue_retention": {},
        }
        for name in targets:
            _, _, changed = previous_label_pairs(truth[name], masks[name])
            results[horizon]["cue_retention"][name] = evidence_retention(
                posterior[name], prior[name], truth[name], changed
            )
    return results
