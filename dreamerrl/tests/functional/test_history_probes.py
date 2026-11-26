import numpy as np
import pytest
import torch

from dreamerrl.eval.memory.history import (AttentionEvidence, budgeted_probe, fit_history_probe, parameter_count,
                                           past_windows, project_fit_scaled)
from dreamerrl.eval.memory.statistics import grouped_split


def test_history_windows_are_causal_and_stop_at_episode_boundary():
    frames = torch.arange(6).float()[:, None]
    ids, time = np.repeat([5, 7], 3), np.tile(np.arange(3), 2)
    windows, valid = past_windows(frames, ids, time, 2)
    assert windows[:, :, 0].tolist() == [[0, 0], [0, 1], [1, 2], [3, 0], [3, 4], [4, 5]]
    assert valid.sum(1).tolist() == [1, 2, 2, 1, 2, 2]
    changed = frames.clone()
    changed[2] += 100
    history, _ = past_windows(changed, ids, time, 2)
    torch.testing.assert_close(windows[[0, 1, 3, 4, 5]], history[[0, 1, 3, 4, 5]])
    with pytest.raises(ValueError, match="timestep"):
        past_windows(frames, ids, np.zeros_like(time), 2)


def test_model_budget_and_padded_prefix_invariance():
    x = torch.randn(3, 5, 33)
    valid = torch.arange(5)[None] < torch.tensor([1, 4, 5])[:, None]
    altered = x.masked_fill(~valid[..., None], 50)
    for kind in ("linear", "current_mlp", "fixed_mlp", "gru", "transformer"):
        model = budgeted_probe(kind, 33, 5, 4096).eval()
        count = parameter_count(model)
        assert count <= 4096
        assert count >= 3200 if kind != "linear" else count == 34
        with torch.no_grad():
            torch.testing.assert_close(model(x, valid), model(altered, valid))
    with pytest.raises(ValueError, match="budget"):
        budgeted_probe("gru", 33, 5, 2)


def test_fit_ignores_heldout_labels_and_projection_uses_fit_only():
    ids = np.arange(50)
    raw = np.random.default_rng(15).normal(size=(50, 5)).astype(np.float32)
    split, _ = grouped_split(ids, 4)
    selected = split["test"]
    x = project_fit_scaled(raw, split["fit"], 4, 18)
    different = raw.copy()
    different[selected] = 1000
    y = project_fit_scaled(different, split["fit"], 4, 18)
    torch.testing.assert_close(x[~selected], y[~selected])
    windows, valid = past_windows(x, ids, np.zeros(50), 1)
    targets = raw[:, 0] * 5
    kwargs = dict(seed=17, budget=128, learning_rates=(.01,), epochs=3)
    first = fit_history_probe("linear", windows, valid, targets, split, **kwargs)
    targets[selected] = 1000
    second = fit_history_probe("linear", windows, valid, targets, split, **kwargs)
    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    assert first["epoch"] == second["epoch"]


def test_attention_requires_independent_coverage_and_positive_increment():
    assert not AttentionEvidence(60, 20, 5, True, (.01, .05)).permits_fit
    assert not AttentionEvidence(60, 20, 20, True, (0, .05)).permits_fit
    assert not AttentionEvidence(60, 20, 20, False, (.01, .05)).permits_fit
    assert AttentionEvidence(60, 20, 20, True, (.01, .05)).permits_fit
    with pytest.raises(ValueError, match="independent temporal"):
        fit_history_probe("transformer", None, None, None, None, seed=0, budget=4096, learning_rates=(.01,), epochs=1)
