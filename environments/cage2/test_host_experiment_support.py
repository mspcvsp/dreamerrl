"""No formal training: test paired initialization and fail-closed source gates."""

import json
from dataclasses import asdict

import host_experiment_support as support
import pytest
import torch

from dreamerrl.training.trainer import DreamerTrainer


def test_config_changes_only_encoder_schema_and_explicit_output_seed(tmp_path):
    flat = support.load_config("flat", 1, tmp_path)
    host = support.load_config("cage2_host", 1, tmp_path)
    a, b = asdict(flat), asdict(host)
    for key in ("obs_encoder", "cage2_host_order", "cage2_subnet_ids"):
        a["world"].pop(key)
        b["world"].pop(key)
    assert a == b
    spec = json.loads(support.CONFIG.read_text())
    assert spec["final_env_steps"] == 480 + 30 * spec["final_update"]
    assert not (set(spec["development_seeds"]) & set(spec["heldout_seeds"]))
    assert all((value - 480) % 30 == 0 for value in spec["evaluation_env_steps"])


def test_common_modules_start_identically_without_checkpoint_loading(tmp_path, monkeypatch):
    trainers = []

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Fresh initialization must not read checkpoint weights")

    monkeypatch.setattr(torch, "load", forbidden_load)
    try:
        hashes = []
        for arm in ("flat", "cage2_host"):
            cfg = support.load_config(arm, 0, tmp_path / arm)
            cfg.train.cuda = False
            trainer = DreamerTrainer(cfg)
            trainers.append(trainer)
            hashes.append(support.initialize_paired(trainer))
            assert trainer.completed_updates == 0 and trainer.replay.size == 0
            assert all(not opt.state for opt in (trainer.model_opt, trainer.actor_opt, trainer.critic_opt))
        assert hashes[0] == hashes[1]
        for name in ("actor", "critic", "slow_critic"):
            for key, value in getattr(trainers[0], name).state_dict().items():
                torch.testing.assert_close(value, getattr(trainers[1], name).state_dict()[key], atol=0, rtol=0)
    finally:
        for trainer in trainers:
            trainer.env.venv.close()
            trainer.tb.close()


def test_unsealed_experiment_cannot_train():
    manifest = support.ROOT / "environments/cage2/host_aware_v1_manifest.json"
    with pytest.raises(ValueError, match="Unsealed"):
        support.verify_gate(manifest)


def test_no_launch_approval_is_fail_closed(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"source_commit": "a" * 40, "source_fingerprints": {"config.json": "x"}}))
    with pytest.raises(ValueError, match="approval"):
        support.verify_gate(path)


def test_changed_fingerprint_is_rejected_before_dependencies(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("changed configuration")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                source_commit="a" * 40,
                source_tag="fixture-v1",
                tag_object="b" * 40,
                launch_approval="fixture only",
                source_fingerprints={"config.json": "wrong", "manifest.json": "wrong"},
            )
        )
    )

    def fake_git(*args, root=None):
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args[0] == "status":
            return ""
        if args[0] == "ls-files":
            return "config.json\nmanifest.json"
        if args[0] == "cat-file":
            return "tag"
        return "b" * 40 if args == ("rev-parse", "fixture-v1") else "a" * 40

    monkeypatch.setattr(support, "git", fake_git)
    monkeypatch.setattr(support.subprocess, "check_output", lambda *a, **k: manifest.read_bytes())
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        support.verify_gate(manifest, root=tmp_path)
