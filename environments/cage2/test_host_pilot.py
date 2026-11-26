"""Pilot scope, scoring/reporting and detached process supervision without training."""

import json
import sys

import host_pilot
import host_pilot_job
import pytest


def test_pilot_is_one_fixed_budget_host_run_without_heldout():
    config = json.loads(host_pilot.PILOT.read_text())
    assert config["training_seed"] == 0 and config["arm"] == "cage2_host"
    assert config["final_env_steps"] == 480 + 30 * config["final_update"] == 507450
    assert config["max_wall_seconds"] == 14400
    assert config["finalization_reserve_seconds"] >= 120
    assert not config["heldout_evaluation"] and len(config["development_seeds"]) == 20
    assert config["tmux_session"] == "cage2-host-aware-v1"


def test_score_requires_matched_seed_order():
    current = dict(mean_return=2.0, episodes=[dict(seed=i, total_return=2.0) for i in range(20)])
    historical_episodes = [dict(seed=i, total_return=0.0) for i in range(20)]
    historical = dict(mean_return=0.0, episodes=historical_episodes)
    assert host_pilot.score(current, historical, 16899, 507450)["historical_paired_change"] == 2
    historical_episodes.reverse()
    with pytest.raises(AssertionError):
        host_pilot.score(current, historical, 16899, 507450)


def test_automatic_reports_create_scorecard_curve_and_summary(tmp_path):
    row = dict(
        update=16899,
        env_steps=507450,
        mean_return=-30.0,
        historical_mean=-35.39,
        historical_paired_change=5.39,
        conditional_episode_ci95=[1.0, 10.0],
    )
    final = dict(
        stop_reason="fixed_budget_complete",
        update=16899,
        env_steps=507450,
        mean_return=-30.0,
        elapsed_seconds=100.0,
        matched_final_budget=True,
    )
    host_pilot.save_reports(tmp_path, [row], final)
    for name in ("scorecard.csv", "learning_curve.png", "final_report.txt"):
        assert (tmp_path / name).stat().st_size > 0
    assert "one-training-seed" in (tmp_path / "final_report.txt").read_text()


def test_supervisor_records_exit_status_and_uses_hard_limit(tmp_path, monkeypatch):
    (tmp_path / "launch.json").write_text("{}")

    class Process:
        pid = 12345

        def wait(self, timeout=None):
            assert timeout == 14400
            return 0

    monkeypatch.setattr(host_pilot_job.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(
        sys, "argv", ["host_pilot_job", "--manifest", str(tmp_path / "manifest.json"), "--run-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as stopped:
        host_pilot_job.main()
    assert stopped.value.code == 0
    assert (tmp_path / "exit_code").read_text() == "0\n"
    launch = json.loads((tmp_path / "launch.json").read_text())
    assert launch["exit_code"] == 0 and launch["pid"] == 12345 and launch["end_time_utc"]
