"""Hierarchy-v2 gate and reporting tests without formal training."""

import json
from typing import Any

import host_experiment_support as support
import launch_hierarchy_v2 as launcher
import pytest
import torch
from thermal_stop import publish

from dreamerrl.tests.functional.test_hierarchical_actor import small_config
from dreamerrl.training.trainer import DreamerTrainer


def protocol_manifest() -> dict[str, Any]:
    return dict(
        source_commit="fixture", hierarchy_config_sha256=support.sha256(support.HIERARCHY_CONFIG),
        execution=json.loads(support.HIERARCHY_CONFIG.read_text()),
    )


def test_resolved_configs_match_across_seeds_and_sealed_schedule(tmp_path):
    manifest = protocol_manifest()
    spec = support.experiment_spec(manifest)
    assert spec["evaluation_env_steps"] == [510, 127800, 254340, 380910, 507450]
    normalized = []
    for seed in range(3):
        cfg = support.load_config("cage2_host", seed, tmp_path / str(seed), spec)
        assert cfg.ac.hierarchical and cfg.train.seed == cfg.env.seed == seed
        data = cfg.to_dict()
        assert data["log"].pop("tb_logdir") == str(tmp_path / str(seed) / "tensorboard")
        data["train"].pop("seed")
        data["env"].pop("seed")
        normalized.append(data)
    assert normalized[0] == normalized[1] == normalized[2]
    manifest["execution"]["training_seeds"] = [2]
    with pytest.raises(ValueError, match="fingerprint"):
        support.experiment_spec(manifest)


def test_fresh_popgym_hierarchy_initialization_preserves_policy(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    trainer = DreamerTrainer(small_config(tmp_path))

    def forbidden(*args, **kwargs):
        pytest.fail("Fresh model initialization read a checkpoint")

    monkeypatch.setattr(torch, "load", forbidden)
    try:
        before = {key: value.clone() for key, value in trainer.actor.state_dict().items()}
        assert set(support.initialize_paired(trainer)) == {"world", "actor", "critic"}
        assert all(torch.equal(value, trainer.actor.state_dict()[key]) for key, value in before.items())
        assert trainer.replay.size == 0 and trainer.completed_updates == 0
        assert all(not optimizer.state for optimizer in (trainer.model_opt, trainer.actor_opt, trainer.critic_opt))
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def test_launcher_gates_and_reports_only_matching_budget(tmp_path, monkeypatch):
    manifest = protocol_manifest()
    protocol = manifest["execution"]
    spec = support.experiment_spec(manifest)
    gate_calls = []

    def gate(path, *, require_approval):
        gate_calls.append(require_approval)
        return manifest

    def supervisor(args):
        assert args == launcher.supervisor_args(tmp_path / "manifest.json", tmp_path / "group", protocol)
        root = tmp_path / "group"
        jobs = []
        for seed in range(3):
            run = root / f"job_{seed}" / "results" / f"cage2_host_seed{seed}"
            run.mkdir(parents=True)
            (run / "latest.pt").write_text(str(seed))
            publish(run / "development_16899.json", {"update": 16899, "mean_return": -27.0 + seed})
            publish(run / "development_curve.json", {})
            (run / "metrics.jsonl").write_text('{"credit_assignment":{"goal_0_decisions":3,"goal_1_decisions":1}}\n')
            jobs.append(dict(
                run=str(run), state="completed" if seed < 2 else "checkpointed",
                env_steps=spec["final_env_steps"], updates=spec["final_update"],
                checkpoint={"checkpoint_status": "saved"}, runtime_seconds=100,
            ))
        publish(root / "execution_summary.json", {"jobs": jobs})
        return 75

    monkeypatch.setattr(launcher, "verify_gate", gate)
    monkeypatch.setattr(launcher.experiment_supervisor, "main", supervisor)
    args = ["--manifest", str(tmp_path / "manifest.json"), "--output-root", str(tmp_path / "group")]
    assert launcher.main([*args, "--check-only"]) == 0
    assert not (tmp_path / "group").exists()
    assert launcher.main(args) == 75
    assert gate_calls == [False, True]
    report = json.loads((tmp_path / "group/final_report.json").read_text())
    assert report["matched_seeds"] == 2 and report["matched_mean"] == -26.5
    assert report["runs"][2]["delta_vs_historical"] is None
    assert report["runs"][0]["diagnostics"]["goal_decision_counts"] == {"goal_0_decisions": 3, "goal_1_decisions": 1}
    assert len({row["checkpoint_sha256"] for row in report["runs"]}) == 3
