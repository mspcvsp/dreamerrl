"""Offline probe and observability regressions using synthetic data and tiny POPGym runs."""

import json
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hierarchy_diagnostics import analyze, collect, future_pairs

from dreamerrl.models.hierarchical_actor import HierarchicalActor
from dreamerrl.tests.functional.test_hierarchical_actor import small_config
from dreamerrl.training.core.hierarchical_update import goal_diagnostics
from dreamerrl.training.trainer import DreamerTrainer


def test_live_goal_metrics_exclude_terminal_starts_and_censored_steps():
    ids = torch.tensor([[0, 2], [0, 1], [1, 2], [1, 2], [1, 2]])
    weights = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0], [0, 0]])
    report = goal_diagnostics(ids, weights, 2, torch.eye(3))
    assert [report[f"goal_{index}_decisions"].item() for index in range(3)] == [1, 1, 0]
    assert report["goal_entropy"].item() == pytest.approx(np.log(2))
    assert report["goal_persistence"].item() == pytest.approx(2 / 3)
    assert report["goal_boundary_repeat_rate"].item() == 0
    assert report["goal_boundary_pairs"].item() == 1
    assert report["goal_observed_interval_mean"].item() == 2
    assert report["goal_full_intervals"].item() == 2
    assert all(torch.isfinite(value) for value in goal_diagnostics(ids, weights * 0, 2, torch.eye(3)).values())


def test_future_pairs_mask_and_episode_disjoint_prediction():
    dataset = []
    for episode in range(30):
        goal = episode % 2
        dataset.append(dict(
            episode=episode, latent=np.array([[0.0], [0.0], [float(goal * 2 - 1)]]),
            observation=np.zeros((3, 1)), goals=np.array([goal, 2]),
            decisions=np.array([True, True]), episode_return=0.0,
        ))
    episodes, steps, selected, _, _, future = future_pairs(dataset, 2)
    assert steps.tolist() == [0] * 30
    assert len(episodes) == 30 and 2 not in selected
    assert set(future.ravel()) == {-1, 1}
    report = analyze(dataset, np.array([[1.0, 0], [1.0, 0], [0.0, 1]]), lag=2, seed=20, draws=100)
    groups = [set(ids) for ids in report["split_episodes"].values()]
    assert len(set.union(*groups)) == 30 and sum(map(len, groups)) == 30
    component = report["components"][0]
    assert component["scores"]["controls_and_goal"]["rmse"] < 0.01
    assert component["added_goal_interval"]["delta_mae_ci95"][1] < -0.5
    assert report["goal_usage"]["redundant_embedding_pairs"] == [[0, 1]]
    assert report["goal_usage"]["decision_counts"] == [15, 15, 30]
    json.dumps(report, allow_nan=False)


def test_popgym_probe_leaves_rng_parameters_and_training_state_unchanged(tmp_path):
    torch.set_num_threads(2)
    trainer = DreamerTrainer(small_config(tmp_path))
    try:
        models = (trainer.world, trainer.actor, trainer.critic)
        original = [{key: value.clone() for key, value in model.state_dict().items()} for model in models]
        trainer.world.posterior.eval()  # Verify restoration of mixed submodule modes.
        modes = [module.training for model in models for module in model.modules()]
        replay = trainer.replay.size, trainer.total_env_steps, trainer.sample_step
        rng = pickle.dumps((random.getstate(), np.random.get_state())), torch.get_rng_state().clone()
        dataset = collect(trainer, 15, 90210)
        assert isinstance(trainer.actor, HierarchicalActor)
        report = analyze(dataset, trainer.actor.goal_embedding.weight.detach().numpy(), lag=3, seed=17)
        assert len(dataset) == 15 and report["pairs"] >= 15
        assert all(np.isfinite(row["latent"]).all() for row in dataset)
        json.dumps(report, allow_nan=False)
        assert rng[0] == pickle.dumps((random.getstate(), np.random.get_state()))
        assert torch.equal(rng[1], torch.get_rng_state())
        assert replay == (trainer.replay.size, trainer.total_env_steps, trainer.sample_step)
        assert modes == [module.training for model in models for module in model.modules()]
        for model, expected in zip(models, original, strict=True):
            assert all(torch.equal(value, model.state_dict()[key]) for key, value in expected.items())
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def write_worker_metrics(run, seed):
    torch.set_num_threads(2)
    run.mkdir()
    cfg = small_config(run / "tensorboard")
    cfg.train.seed = cfg.env.seed = seed
    (run / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), allow_nan=False))
    trainer = DreamerTrainer(cfg)
    try:
        with (run / "metrics.jsonl").open("x") as stream:
            def emit(record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")

            trainer.train(4, checkpoint_path=run / "latest.pt", on_update=emit)
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def test_concurrent_seed_metrics_checkpoints_and_tensorboard_are_isolated(tmp_path):
    program = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from test_hierarchy_diagnostics import write_worker_metrics; "
        "write_worker_metrics(Path(sys.argv[2]), int(sys.argv[3]))"
    )
    seeds = (17, 29)
    processes = []
    try:
        for seed in seeds:
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", program, str(Path(__file__).parent), str(tmp_path / str(seed)), str(seed)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            processes.append(process)
        for process in processes:
            output, _ = process.communicate(timeout=75)
            assert process.returncode == 0, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    for seed in seeds:
        run = tmp_path / str(seed)
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        config = json.loads((run / "resolved_config.json").read_text())
        checkpoint = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        assert config["train"]["seed"] == checkpoint["config"].train.seed == seed
        assert [row["update"] for row in rows] == [1, 2, 3, 4]
        assert checkpoint["completed_updates"] == 4
        assert list((run / "tensorboard").rglob("events.out.tfevents.*"))
        for row in rows:
            assert row["training_seed"] == row["environment_seed"] == seed
            assert row["tensorboard_dir"] == str(run / "tensorboard")
            assert row["train_call_elapsed_seconds"] > 0 and row["env_steps_per_second"] > 0
            assert row["world_model_grad_norm"] > 0
            assert row["cadence"]["actor_optimizer_steps"] == row["cadence"]["critic_optimizer_steps"] == 1
            credit = row["credit_assignment"]
            assert np.isfinite(list(credit.values())).all()
            assert credit["manager_critic_loss"] + credit["worker_critic_loss"] == pytest.approx(row["critic_loss"])
            assert sum(credit[f"goal_{index}_decisions"] for index in range(4)) == credit["manager_decisions"]
