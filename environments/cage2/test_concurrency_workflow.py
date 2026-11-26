"""Offline workflow and recorded timing regressions; never launches a learner."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from concurrency_benchmark import summarize
from host_aware_experiment import execution_order

ROOT = Path(__file__).resolve().parents[2]


def test_existing_order_preserved_and_single_seed_isolated():
    spec = json.loads((ROOT / "environments/cage2/host_aware_v1_config.json").read_text())
    before = json.dumps(spec, sort_keys=True)
    assert execution_order(spec) == spec["sequential_run_order"]
    assert execution_order(spec, "cage2_host", 1) == [{"arm": "cage2_host", "seed": 1}]
    assert execution_order(spec, "cage2_host", 2) == [{"arm": "cage2_host", "seed": 2}]
    assert json.dumps(spec, sort_keys=True) == before


@pytest.mark.parametrize("arm,seed", [("flat", None), (None, 1), ("cage2_host", 999)])
def test_invalid_selection_is_rejected(arm, seed):
    spec = json.loads((ROOT / "environments/cage2/host_aware_v1_config.json").read_text())
    with pytest.raises(ValueError):
        execution_order(spec, arm, seed)


@pytest.mark.parametrize("number", [1, 2])
def test_recorded_aggregate_timing_recomputes_without_touching_evidence(number, tmp_path):
    run = ROOT / "runs/cage2_concurrency_20261124"
    if not run.exists():
        pytest.skip("Recorded Spark artifacts are not distributed with source")
    phase = run / f"processes_{number}"
    paths = list(phase.rglob("*"))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    shutil.copytree(phase, tmp_path / "phase")
    expected = json.loads((phase / "summary.json").read_text())
    end = json.loads((phase / "go.json").read_text())["end"]
    actual = summarize(tmp_path / "phase", end)
    for key in ("measured_seconds", "measured_updates", "aggregate_updates_per_second", "available_gib_min"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    assert actual["updates_per_worker"] == expected["updates_per_worker"]
    for path, digest in before.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_single_arm_cli_keeps_training_runner_and_defers_heldout(tmp_path, monkeypatch):
    import sys

    import host_aware_experiment as experiment

    calls = []
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "verify_gate", lambda *args, **kwargs: {})
    monkeypatch.setattr(experiment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(experiment.torch.cuda, "get_device_name", lambda index: "NVIDIA GB10")
    monkeypatch.setattr(experiment.torch, "set_num_threads", lambda threads: None)
    monkeypatch.setattr(experiment, "run_arm", lambda manifest, root, arm, seed: calls.append((arm, seed)))

    def forbidden(*args):
        raise AssertionError("Single-arm execution must not inspect held-out results")

    monkeypatch.setattr(experiment, "evaluate_heldout", forbidden)
    monkeypatch.setattr(experiment, "report_comparison", forbidden)
    run = tmp_path / "runs/isolated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "host_aware_experiment",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-root",
            str(run),
            "--arm",
            "cage2_host",
            "--seed",
            "1",
        ],
    )
    experiment.main()
    assert calls == [("cage2_host", 1)]
    assert json.loads((run / "execution_selection.json").read_text())["single_arm"]


def test_recorded_artifact_inventory_and_benchmark_source_match():
    inventory = json.loads((ROOT / "environments/cage2/concurrency_benchmark_20261124.json").read_text())
    source = ROOT / "environments/cage2/concurrency_benchmark.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == inventory["source_sha256"]
    run = ROOT / inventory["artifact_root"]
    if not run.exists():
        pytest.skip("Recorded Spark artifacts are not distributed with source")
    assert len(inventory["artifacts"]) == 30
    for relative, expected in inventory["artifacts"].items():
        path = run / relative
        assert path.stat().st_size == expected["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
