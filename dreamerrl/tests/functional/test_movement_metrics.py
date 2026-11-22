"""Movement counters distinguish standing still, oscillation, and game-time stalls."""

from dreamerrl.eval.movement import MovementTracker


def test_stationary_streaks_and_failed_moves():
    tracker = MovementTracker((1, 1), ["N", "OPEN"])
    tracker.step(0, (1, 1), movement_action=True, turn_advanced=False)
    tracker.step(1, (1, 1), turn_advanced=True)
    tracker.step(0, (1, 2), movement_action=True, turn_advanced=True)
    tracker.step(0, (1, 1), movement_action=True, turn_advanced=True)
    r = tracker.report()
    assert r["longest_stationary_streak"] == 2
    assert r["no_movement_rate"] == 0.5
    assert r.get("failed_movement_rate") == 1 / 3
    assert r["frozen_game_turns"] == 1
    assert r["two_step_returns"] == 1
    assert r["unique_tiles"] == 2
    assert r["action_counts"] == {"0": 3, "1": 1}
    assert sum(r["action_frequencies"].values()) == 1


def test_episode_reset_starts_fresh_tracker():
    tracker = MovementTracker((10, 10), ["SEARCH"])
    tracker.step(0, (10, 10), turn_advanced=True)
    assert tracker.report()["longest_stationary_streak"] == 1
    assert tracker.report()["unique_tiles"] == 1
    assert tracker.report()["two_step_returns"] == 0
    assert "failed_movement_rate" not in tracker.report()
