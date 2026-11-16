"""Open-loop predictions use recorded actions without future observation leakage."""

from unittest.mock import patch

import pytest
import torch

from dreamerrl.eval.cage2_events import compare_host_rollouts
from dreamerrl.eval.field_events import evidence_retention
from dreamerrl.eval.imagination_probe import action_conditioned_rollout, rollout_anchors
from dreamerrl.models.world_model import WorldModelState
from dreamerrl.tests.functional.test_replay_burn_in import model


def test_anchors_reject_cross_episode_rollouts_but_allow_terminal_endpoint():
    first = torch.tensor([[True, False, False, True, False, False]])
    last = torch.tensor([[False, False, True, False, False, True]])
    anchors = rollout_anchors(first, last, 2)
    assert set(map(tuple, anchors.tolist())) == {(0, 0), (0, 3)}
    assert rollout_anchors(first, last, 3).shape == (0, 2)


def test_rollout_action_alignment_no_future_filtering_and_rng_isolation():
    world = model()
    h = torch.randn(1, 6, 8)
    z = torch.nn.functional.one_hot(torch.zeros(1, 6, 2, dtype=torch.long), 3).float()
    actions = torch.tensor([[2, 0, 1, 2, 1, 0]])
    anchors = torch.tensor([[0, 1]])
    rng = torch.get_rng_state().clone()
    with (
        patch.object(world.encoder, "forward", side_effect=AssertionError("future encoding")),
        patch.object(world.posterior, "forward", side_effect=AssertionError("future posterior")),
    ):
        result = action_conditioned_rollout(world, h, z, actions, anchors, [1, 3])
    assert torch.equal(rng, torch.get_rng_state())
    state = WorldModelState(h[:, 1], z[:, 1])
    for step in range(1, 4):
        action = torch.nn.functional.one_hot(actions[:, 1 + step], 3).float()
        state = world.imagine_step(state, torch.nn.Identity(), action=action, deterministic_imagination=True)
        if step in result:
            torch.testing.assert_close(result[step].h, state.h)
            torch.testing.assert_close(result[step].z, state.z)
    poisoned_h, poisoned_z = h.clone(), z.clone()
    poisoned_h[:, 2:] = 999
    poisoned_z[:, 2:] = 999
    again = action_conditioned_rollout(world, poisoned_h, poisoned_z, actions, anchors, [3])
    torch.testing.assert_close(again[3].h, result[3].h)
    changed = actions.clone()
    changed[:, 2] = 2
    altered = action_conditioned_rollout(world, h, z, changed, anchors, [3])
    assert not torch.allclose(altered[3].h, result[3].h)
    first = torch.zeros(1, 6, dtype=torch.bool)
    first[:, 3] = True
    with pytest.raises(ValueError, match="boundary"):
        action_conditioned_rollout(world, h, z, actions, anchors, [3], is_first=first)


def test_host_rollouts_score_anchor_persistence_and_lost_posterior_evidence():
    targets = {"Activity": torch.tensor([[[0], [1], [2], [2]]]), "Compromised": torch.tensor([[[0], [1], [2], [3]]])}
    observed = {key: torch.ones_like(value, dtype=torch.bool) for key, value in targets.items()}
    predictions = {3: {key: torch.zeros(1, 1, dtype=torch.long) for key in targets}}
    flags = torch.zeros(1, 4, dtype=torch.bool)
    report = compare_host_rollouts(
        targets, predictions, targets, observed, torch.tensor([[0, 0]]), is_first=flags, is_last=flags
    )[3]
    for name in targets:
        assert report["posterior_reconstruction"][name]["changed_model"]["count"] == 1
        assert (
            report["open_loop_prior"][name]["changed_persistence"]["per_class"][
                "Exploit" if name == "Activity" else "Privileged"
            ]["false_negatives"]
            == 1
        )
        retention = report["cue_retention"][name]
        assert retention["reference_correct_count"] == 1
        assert retention["lost_count"] == 1
        assert retention["retention_given_reference"] == 0


def test_retention_undefined_when_posterior_already_misses_evidence():
    report = evidence_retention(torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), torch.tensor([True]))
    assert report["reference_correct_count"] == 0
    assert "retention_given_reference" not in report


def test_navigation_retention_counts_only_scored_endpoint_cues():
    from dreamerrl.eval.minihack_events import score_minihack_events

    # A player moves right; 0 denotes blocked terrain and 1 passable terrain.
    truth = torch.tensor([[[0, 0, 0, 0, 2, 1, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, 0, 0]]])
    posterior = truth.clone()
    prior = truth[:, :1].expand_as(truth)
    metrics = score_minihack_events(
        prior,
        truth,
        (3, 3),
        player_ids=(2,),
        passability_lookup=(0, 1, -1),
        score_frames=torch.tensor([[False, True]]),
        reference_predictions=posterior,
        is_last=torch.zeros(1, 2, dtype=torch.bool),
    )
    assert metrics["glyph_player_retention_reference_correct_count"] == 1
    assert metrics["glyph_player_retention_lost_count"] == 1
    assert metrics["glyph_changed_nonterminal_retention_reference_correct_count"] == 2
    assert metrics["glyph_changed_nonterminal_retention_retained_count"] == 0


def test_sampled_rollout_restores_prior_mode_and_random_state():
    world = model()
    h = torch.zeros(1, 3, 8)
    z = torch.nn.functional.one_hot(torch.zeros(1, 3, 2, dtype=torch.long), 3).float()
    rng = torch.get_rng_state().clone()
    result = action_conditioned_rollout(
        world, h, z, torch.zeros(1, 3, dtype=torch.long), torch.tensor([[0, 0]]), [2], sampled=True
    )
    assert world.prior.deterministic_latent_for_tests
    assert torch.equal(torch.get_rng_state(), rng)
    assert result[2].z.shape == (1, 2, 3)
