"""Event metrics expose static-label shortcuts without reset/missing-label leakage."""

import torch

from dreamerrl.eval.cage2_events import score_host_fields
from dreamerrl.eval.field_events import categorical_event_scores, json_metrics
from dreamerrl.eval.minihack_events import score_minihack_events


def test_per_class_recall_false_positives_and_matched_persistence():
    target = torch.tensor([[[0, 1], [1, 1], [1, 2], [2, 2]]])
    pred = torch.tensor([[[1, 1], [1, 0], [1, 0], [2, 2]]])
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[:, 2, 1] = False
    first = torch.tensor([[True, False, False, True]])
    report = categorical_event_scores(pred, target, 3, observed=observed, is_first=first)
    scan = report["all_observed"]["per_class"][1]
    assert scan["support"] == 4
    assert scan["false_positives"] == 1
    assert scan["recall"] == 0.75
    torch.testing.assert_close(scan["false_positive_rate"], torch.tensor(1 / 3))
    assert report["paired_model"]["count"] == report["persistence"]["count"] == 3
    assert report["changed_model"]["count"] == 1
    assert report["changed_model"]["accuracy"] == 1
    assert report["changed_persistence"]["accuracy"] == 0
    assert report["persistent_baseline"]["accuracy"] == 1
    assert json_metrics(report)["changed_model"]["count"] == 1


def test_missing_gaps_and_episode_ends_do_not_create_events_or_baselines():
    target = torch.tensor([[[0], [-1], [1], [2], [0]]])
    observed = target >= 0
    first = torch.tensor([[True, False, False, True, False]])
    last = torch.tensor([[False, False, False, True, False]])
    pred = target.clamp_min(0)
    report = categorical_event_scores(pred, target, 3, observed=observed, is_first=first, is_last=last)
    assert report["all_observed"]["count"] == 4
    assert report["paired_model"]["count"] == 0
    assert report["changed_model"]["count"] == 0
    assert "accuracy" not in report["changed_model"]


def test_host_unknown_is_scored_and_absent_class_rates_are_undefined():
    targets = {"Activity": torch.tensor([[[0], [1], [2]]]), "Compromised": torch.tensor([[[0], [1], [1]]])}
    observed = {name: torch.ones_like(value, dtype=torch.bool) for name, value in targets.items()}
    predictions = {name: value.clone() for name, value in targets.items()}
    report = score_host_fields(predictions, targets, observed)
    comp = report["Compromised"]["all_observed"]["per_class"]
    assert comp["Unknown"]["support"] == 2 and comp["Unknown"]["recall"] == 1
    assert comp["Privileged"]["support"] == 0 and "recall" not in comp["Privileged"]
    assert comp["Privileged"]["false_positive_rate"] == 0
    assert report["Compromised"]["changed_model"]["accuracy"] == 1
    assert report["Compromised"]["changed_persistence"]["accuracy"] == 0


def mini_trace():
    first = [0, 0, 0, 0, 2, 1, 0, 0, 0]
    moved = [0, 0, 0, 0, 1, 2, 0, 0, 0]
    targets = torch.tensor([[first, moved, moved]])
    pred = torch.tensor([[first, first, moved]])
    return pred, targets


def test_player_changed_tiles_and_nearby_passability_have_separate_supports():
    pred, targets = mini_trace()
    scores = score_minihack_events(pred, targets, (3, 3), player_ids=(2,), passability_lookup=(0, 1, -1, -1))
    assert scores["glyph_player_count"] == 3
    torch.testing.assert_close(scores["glyph_player_accuracy"], torch.tensor(2 / 3))
    assert scores["glyph_player_paired_count"] == scores["glyph_player_persistence_count"] == 2
    assert scores["glyph_player_paired_accuracy"] == scores["glyph_player_persistence_accuracy"] == 0.5
    assert scores["glyph_changed_count"] == 2
    assert scores["glyph_changed_accuracy"] == scores["glyph_changed_persistence_accuracy"] == 0
    assert scores["glyph_persistent_accuracy"] == 1
    assert scores["glyph_nearby_passability_count"] == 18
    torch.testing.assert_close(scores["glyph_nearby_passability_accuracy"], torch.tensor(17 / 18))
    assert scores["glyph_nearby_passability_predicted_unknown"] == 1
    torch.testing.assert_close(scores["glyph_nearby_passability_persistence_accuracy"], torch.tensor(0.9))


def test_ambiguous_player_is_excluded_and_selected_frames_keep_previous_labels():
    pred, targets = mini_trace()
    targets[:, 0, 0] = 2  # two markers: position cannot be inferred safely
    frames = torch.tensor([[False, True, False]])
    scores = score_minihack_events(
        pred, targets, (3, 3), player_ids=(2,), passability_lookup=(0, 1, -1, -1), score_frames=frames
    )
    assert scores["glyph_player_count"] == 1
    assert scores["glyph_player_persistence_count"] == 1  # previous frame need not itself be scored
    assert scores["glyph_player_persistence_accuracy"] == 0
    all_frames = score_minihack_events(pred, targets, (3, 3), player_ids=(2,), passability_lookup=(0, 1, -1, -1))
    assert all_frames["glyph_player_unresolved_frames"] == 1
    assert all_frames["glyph_player_count"] == 2


def test_installed_terrain_schema_does_not_treat_dark_or_hazardous_cells_as_known():
    from nle import nethack

    from dreamerrl.env.minihack.minihack_utils import glyph_scoring_schema

    lookup, player = glyph_scoring_schema()
    offset = int(getattr(nethack, "GLYPH_CMAP_OFF"))
    symbols = getattr(nethack, "symdef")
    expected = {
        "floor of a room": 1,
        "corridor": 1,
        "wall": 0,
        "closed door": 0,
        "dark part of a room": -1,
        "water": -1,
        "molten lava": -1,
    }
    found = set()
    for index in range(int(getattr(nethack, "MAXPCHARS"))):
        name = symbols.from_idx(index).explanation
        if name in expected:
            assert lookup[offset + index] == expected[name]
            found.add(name)
    assert found == set(expected)
    rogue = next(i for i in range(int(getattr(nethack, "NUMMONS"))) if getattr(nethack, "permonst")(i).mname == "rogue")
    assert int(getattr(nethack, "GLYPH_MON_OFF")) + rogue in player
    assert all(lookup[glyph] == -1 for glyph in player)


def test_supplied_position_does_not_score_a_missing_player_on_cleared_terminal_screen():
    targets = torch.tensor([[[0, 0, 0, 0, 2, 1, 0, 0, 0], [0] * 9]])
    positions = torch.tensor([[[1, 1], [1, 1]]])
    scores = score_minihack_events(
        targets,
        targets,
        (3, 3),
        player_ids=(2,),
        passability_lookup=(0, 1, -1),
        player_positions=positions,
        is_last=torch.tensor([[False, True]]),
    )
    assert scores["glyph_player_count"] == 1
    assert scores["glyph_player_unresolved_frames"] == 1
    assert scores["glyph_player_accuracy"] == 1
    assert scores["glyph_nearby_passability_count"] == 8
    assert scores["glyph_changed_terminal_count"] == 2
    assert scores["glyph_changed_terminal_accuracy"] == 1
    assert scores["glyph_changed_nonterminal_count"] == 0
