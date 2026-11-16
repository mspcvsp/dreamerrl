"""Player, changed-tile, and nearby terrain scores, separate from overall accuracy."""

import torch

from dreamerrl.env.minihack.minihack_utils import glyph_scoring_schema
from dreamerrl.eval.field_events import accuracy_counts, previous_label_pairs


@torch.no_grad()
def score_minihack_events(
    predictions,
    targets,
    shape,
    *,
    is_first=None,
    is_last=None,
    observed=None,
    player_positions=None,
    player_ids=None,
    passability_lookup=None,
    score_frames=None,
):
    """Inputs are (batch,time,cells); supplied positions are diagnostic labels only.

    Otherwise locate a single standard player-role glyph. Ambiguous/missing positions
    are excluded from player/neighborhood scoring, with explicit count metrics.
    """
    if predictions.shape != targets.shape or targets.ndim != 3 or targets.shape[-1] != shape[0] * shape[1]:
        raise ValueError("MiniHack scores require aligned (batch,time,cells) tensors")
    if passability_lookup is None or player_ids is None:
        lookup, default_ids = glyph_scoring_schema()
        passability_lookup = lookup if passability_lookup is None else passability_lookup
        player_ids = default_ids if player_ids is None else player_ids
    valid = torch.ones_like(targets, dtype=torch.bool) if observed is None else observed.bool()
    frames = torch.ones_like(targets[..., 0], dtype=torch.bool) if score_frames is None else score_frames.bool()
    if frames.shape != targets.shape[:-1]:
        raise ValueError("score_frames must match batch/time dimensions")
    previous, paired, changed = previous_label_pairs(targets, valid, is_first=is_first, is_last=is_last)
    mask = valid & frames.unsqueeze(-1)
    paired = paired & frames.unsqueeze(-1)
    changed = changed & frames.unsqueeze(-1)
    correct, repeats = predictions == targets, previous == targets
    result = {}

    def add(name, good, eligible):
        result.update({f"glyph_{name}_{metric}": value for metric, value in accuracy_counts(good, eligible).items()})

    add("all", correct, mask)
    add("paired", correct, paired)
    add("persistence", repeats, paired)
    add("changed", correct, changed)
    add("changed_persistence", repeats, changed)
    if is_last is not None:
        terminal = is_last.bool().unsqueeze(-1)
        add("changed_nonterminal", correct, changed & ~terminal)
        add("changed_nonterminal_persistence", repeats, changed & ~terminal)
        add("changed_terminal", correct, changed & terminal)
        add("changed_terminal_persistence", repeats, changed & terminal)
    add("persistent", correct, paired & ~changed)
    add("persistent_baseline", repeats, paired & ~changed)
    B, T, E = targets.shape
    if player_positions is not None:
        if player_positions.shape != (B, T, 2):
            raise ValueError("Player positions must have shape (batch,time,2)")
        x, y = player_positions[..., 0], player_positions[..., 1]
        located = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
        center = (y * shape[1] + x).long().clamp(0, E - 1)
    else:
        marker = torch.zeros_like(valid)
        for glyph in player_ids:
            marker |= (targets == glyph) & valid
        located = marker.sum(-1) == 1
        center = marker.long().argmax(-1)
        x, y = center % shape[1], center // shape[1]
    # Even supplied coordinates can point to a cleared terminal screen. Score
    # player glyphs only when that glyph is actually present at the position.
    at_position = targets.gather(-1, center.unsqueeze(-1)).squeeze(-1)
    visible = torch.zeros_like(located)
    for glyph in player_ids:
        visible |= at_position == glyph
    located &= frames & visible & valid.gather(-1, center.unsqueeze(-1)).squeeze(-1)
    player = torch.zeros_like(valid).scatter(-1, center.unsqueeze(-1), located.unsqueeze(-1)) & valid
    add("player", correct, player)
    add("player_paired", correct, player & paired)
    add("player_persistence", repeats, player & paired)
    result["glyph_player_unresolved_frames"] = (frames & ~located).sum()
    offsets = targets.new_tensor([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy])
    xx, yy = x.unsqueeze(-1) + offsets[:, 0], y.unsqueeze(-1) + offsets[:, 1]
    neighbors = (yy * shape[1] + xx).long().clamp(0, E - 1)
    in_bounds = located.unsqueeze(-1) & (xx >= 0) & (xx < shape[1]) & (yy >= 0) & (yy < shape[0])
    lookup = torch.as_tensor(passability_lookup, dtype=torch.long, device=targets.device)
    target_ids, pred_ids, prev_ids = (
        tensor.gather(-1, neighbors).long() for tensor in (targets, predictions, previous)
    )

    # A genuinely missing target may use -1; it is never an observed terrain label.
    def terrain(ids):
        known_ids = (ids >= 0) & (ids < len(lookup))
        result = torch.full_like(ids, -1)
        result[known_ids] = lookup[ids[known_ids]]
        return result

    target_pass, pred_pass, prev_pass = (terrain(ids) for ids in (target_ids, pred_ids, prev_ids))
    known = in_bounds & valid.gather(-1, neighbors) & (target_pass >= 0)
    neighborhood_pair = known & paired.gather(-1, neighbors)
    add("nearby_passability", pred_pass == target_pass, known)
    add("nearby_passability_paired", pred_pass == target_pass, neighborhood_pair)
    add("nearby_passability_persistence", prev_pass == target_pass, neighborhood_pair)
    result["glyph_nearby_passability_predicted_unknown"] = (known & (pred_pass < 0)).sum()
    if known.any():
        result["glyph_nearby_passability_predicted_unknown_rate"] = (
            result["glyph_nearby_passability_predicted_unknown"].float() / known.sum()
        )
    for name, category in (("passable", 1), ("blocked", 0)):
        add(f"nearby_{name}_recall", pred_pass == category, known & (target_pass == category))
        key = f"glyph_nearby_{name}_recall_accuracy"
        if key in result:
            result[f"glyph_nearby_{name}_recall"] = result.pop(key)
    return result
