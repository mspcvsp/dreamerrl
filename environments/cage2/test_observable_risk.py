"""Observed labels and future-target alignment; no simulator needed."""

import numpy as np
from observable_risk import calibration, future_targets, observable_risk


def test_native_visible_codes_and_unknown_are_not_privileged():
    obs = np.zeros((4, 52))
    obs[0, :4] = [1, 1, 0, 0]  # exploit evidence
    obs[1, :4] = [0, 0, 0, 1]  # User belief
    obs[2, :4] = [0, 0, 1, 1]  # Privileged belief
    obs[3, :4] = [0, 0, 1, 0]  # Unknown is tracked separately
    risk, unknown = observable_risk(obs)
    np.testing.assert_allclose(risk, [1 / 13, 1 / 13, 1 / 13, 0])
    np.testing.assert_allclose(unknown, [0, 0, 0, 1 / 13])


def test_future_target_excludes_current_and_other_episode():
    episode = np.repeat([0, 1], 30)
    timestep = np.tile(np.arange(30), 2)
    risk = np.concatenate([np.arange(30) / 30, np.ones(30)])
    target, valid = future_targets(risk, episode, timestep, 5)
    np.testing.assert_allclose(target[0], np.arange(1, 6).mean() / 30)
    assert valid.sum() == 50
    assert not valid[25:30].any()
    np.testing.assert_allclose(target[30:55], 1)


def test_regression_calibration_identity():
    y = np.linspace(0, 1, 100)
    result = calibration(y, y)
    assert result["mean_bias"] == 0
    np.testing.assert_allclose(result["slope"], 1)
