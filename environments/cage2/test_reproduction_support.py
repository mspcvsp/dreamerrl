"""CPU-only checks for schedule replay, checkpoint identity and tracked launch paths."""

import json
import shutil
import subprocess
from pathlib import Path

import experiment_protocol
import pytest
from launch_learning import launch_command
from launch_learning import main as launch_main


def test_original_schedule_matches_recorded_lr_horizons_and_evaluations():
    protocol = experiment_protocol.original_protocol()
    final, evaluations = experiment_protocol.original_schedule()
    assert (final, evaluations) == (16899, [4244, 8462, 12681, 16899])
    assert protocol["execution"]["calibration"]["scheduler_total_updates"] == 100000
    assert protocol["execution"]["fixed_original_schedule"]["scheduler_total_updates"] == final


def test_wrong_checkpoint_rejected_before_unpickling(tmp_path):
    source = tmp_path / "untrusted.pt"
    source.write_bytes(b"not the original checkpoint")
    with pytest.raises(ValueError, match="fingerprint"):
        experiment_protocol.verify_original_inputs(source, None)


def test_scientific_config_checked_but_log_paths_relocatable(tmp_path, monkeypatch):
    import hashlib

    source = tmp_path / "trusted.pt"
    source.write_bytes(b"fixture")
    config = {"world": {"hidden": 200}, "log": {"tb_logdir": "old"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_checkpoint": {"sha256": hashlib.sha256(b"fixture").hexdigest()},
                "configuration": {"file": "config.json"},
            }
        )
    )
    monkeypatch.setattr(experiment_protocol, "MANIFEST", manifest)
    config["log"]["tb_logdir"] = "new"
    experiment_protocol.verify_original_inputs(source, config)
    config["world"]["hidden"] = 201
    with pytest.raises(ValueError, match="scientific configuration"):
        experiment_protocol.verify_original_inputs(source, config)


def test_tmux_dry_run_has_no_side_effects_or_runs_script_dependency(tmp_path, capsys):
    run = tmp_path / "new run"
    launch_main(["--run-dir", str(run), "--session", "diagnostic", "--dry-run"])
    output = capsys.readouterr().out
    assert "run_learning_clean.sh" in output and "--schedule original" in output
    assert not run.exists() and not Path(str(run) + "_job").exists()


def test_launcher_shell_quotes_paths_and_calls_only_tracked_entrypoint(tmp_path):
    marker = tmp_path / "must-not-exist"
    run = tmp_path / ("run; touch " + str(marker).replace("/", "_") + " ' quote")
    command = launch_command(run, "safe", tmp_path / "source ' quoted.pt", "original")
    assert command[:4] == ["tmux", "new-session", "-d", "-s"]
    # Parse shell syntax without executing training or tmux.
    subprocess.run(["bash", "-n", "-c", command[-1]], check=True)
    assert not marker.exists()


def test_report_regenerated_only_in_scratch_matches_historical_outputs(tmp_path):
    from report_learning import main as report_main

    original = Path(__file__).resolve().parents[2] / "runs/cage2_blue_learning_seed0_20261123"
    if not original.exists():
        pytest.skip("Historical report artifacts are not bundled with source")
    for name in ("scorecard.json", "complete.json"):
        shutil.copyfile(original / name, tmp_path / name)
    report_main(["--run-dir", str(tmp_path)])
    for name in ("scorecard.csv", "final_report.txt"):
        assert (tmp_path / name).read_bytes() == (original / name).read_bytes()
