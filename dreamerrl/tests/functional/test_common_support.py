from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamerrl.eval.memory.interfaces import Episode
from dreamerrl.eval.memory.matching import ContrastFrame, exact_contrasts
from dreamerrl.eval.memory.statistics import binary_probe, exact_binomial_ci95
from environments.cage2.history_common_support import (
    PROBE_NAMES, check_controls, compare_fit_predictors, coverage, eligible_frames, paired_metrics,
    readout_metrics,
)


def test_canonical_matching_maximizes_pairs_and_uses_earliest_qualifying_frame():
    # The first lexicographic edge cannot be chosen without losing a pair.
    rows = [ContrastFrame(i, t, (t, b"observed", 1), y, 1 if y else None)
            for a, b, t in ((1, 2, 6), (1, 3, 5), (1, 3, 4), (2, 4, 7))
            for i, y in ((a, False), (b, True))]
    rows += [ContrastFrame(8, 4, (4, b"observed", 2), True, 1)]
    pairs, matchable = exact_contrasts(reversed(rows))
    assert [(a.episode, b.episode, a.time) for a, b in pairs] == [(1, 3, 4), (2, 4, 7)]
    assert len(matchable) == 8
    counts = coverage(rows, matchable, pairs, 3, 5)
    assert counts["eligible_without_opposite_episode"] == 1
    assert counts["selected"]["by_elapsed_time"] == {"4": {"negative": 1, "positive": 1},
                                                     "7": {"negative": 1, "positive": 1}}
    assert counts["selected"]["by_time_since_last_active"]["never"]["negative"] == 2


def test_recent_target_looks_backward_and_disallows_currently_active_frames():
    obs = torch.zeros(30, 52)
    actions = torch.arange(30) % 3
    ep = Episode(4, 4, obs, actions, torch.zeros(30), False, True)
    labels = np.zeros(30, dtype=bool)
    labels[[1, 5, 18]] = True
    spec = dict(first_timestep=4, last_timestep=29, lookback_frames=[1, 2, 3, 4])
    rows, excluded = eligible_frames([ep], {4: labels}, spec)
    assert excluded == 2
    positive = [(row.time, row.age) for row in rows if row.positive]
    assert positive == [(4, 3), (6, 1), (7, 2), (8, 3), (9, 4), (19, 1), (20, 2), (21, 3), (22, 4)]
    assert rows[0].key[-1] == int(actions[3])


def test_controls_and_locked_probe_on_unseen_pairs():
    def features(pairs):
        shared = np.repeat(np.arange(pairs, dtype=float), 2).reshape(-1, 1)
        labels = np.tile([-1., 1.], pairs)
        return {"current": shared, "fixed_carry": np.column_stack((shared, shared)),
                "recurrent": np.column_stack((shared, labels)),
                "raw_history": np.column_stack((shared, -labels))}

    models, tuning = compare_fit_predictors(features(10), features(4), (.001, .03, 1.))
    assert set(models) == set(PROBE_NAMES)
    assert all(v["penalty"] == 1. for v in tuning.values())
    x = features(12)
    controls = check_controls(x)
    pairs = [(SimpleNamespace(episode=i*2, age=None), SimpleNamespace(episode=i*2+1, age=1, time=7))
             for i in range(12)]
    scores = readout_metrics(pairs, x, models, controls)
    assert scores["recurrent"]["correct"] == scores["raw_history"]["correct"] == 12
    assert controls["current"]["score_ties"] and controls["fixed_carry"]["score_ties"]
    assert paired_metrics(pairs, np.zeros(12), np.zeros(12))["correct_fraction"] == 0


def test_exact_binomial_threshold_at_pinned_minimum():
    assert exact_binomial_ci95(10, 12)[0] == pytest.approx(.5158622513140328)
    assert exact_binomial_ci95(9, 12)[0] < .5
    assert exact_binomial_ci95(0, 12)[0] == 0
    assert exact_binomial_ci95(12, 12)[1] == 1


def test_freeze_preserves_class_weighted_predictor_on_unbalanced_rows():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(16, 4))
    y = np.array([False, False, True, False] * 4)
    split = {name: np.isin(np.arange(16), positions) for name, positions in
             (("fit", range(8)), ("validation", range(8, 12)), ("test", range(12, 16)))}
    result = binary_probe(x[:, None], y, split, balanced=True, freeze=True)
    np.testing.assert_allclose(result["scores"][:, 0], result["predictor"].predict(x), atol=1e-12)
