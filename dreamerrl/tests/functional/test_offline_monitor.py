"""Durable metrics, baseline identity, safe stopping, and terminal-independent output."""

import io
import json
import math
import random
from statistics import mean

import numpy as np
import pytest
import torch
from rich.console import Console

from dreamerrl.eval.minihack_eval import compare_returns, evaluate_minihack
from dreamerrl.logging.jsonl_logger import JSONLLogger
from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.run_monitor import MiniHackRunMonitor
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.rl_dashboard import RLDashboard


def read_events(directory):
    return [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]


def close(trainer):
    trainer.env.venv.close()
    trainer.tb.close()


def test_paired_return_comparison():
    result = compare_returns([3.0, 4.0, 5.0], [1.0, 2.0, 3.0])
    assert result == dict(mean_difference=2.0, ci95_low=2.0, ci95_high=2.0)
    assert compare_returns([1.0], [0.0])["ci95_low"] is None
    with pytest.raises(ValueError):
        compare_returns([1.0], [])


def test_jsonl_appends_and_repairs_interrupted_tail(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JSONLLogger(path)
    logger.log(dict(event="first", value=1.0))
    with pytest.raises(ValueError):
        logger.log(dict(value=math.nan))
    logger.close()
    with path.open("ab") as file:
        file.write(b'{"interrupted":')
    logger = JSONLLogger(path)
    logger.log(dict(event="second", value=2.0))
    logger.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in records] == ["first", "second"]
    assert all("timestamp" in r for r in records)


def test_dashboard_renders_dreamer_fields_without_textual(tmp_path):
    stream = io.StringIO()
    console = Console(file=stream, width=100, force_terminal=True)
    dashboard = RLDashboard(tmp_path, "cpu", mode="off", console=console)
    dashboard.update(dict(update=2, target_updates=10, world_loss=1.0, env_steps=40))
    console.print(dashboard.render())
    assert "Dreamer training" in stream.getvalue()
    assert "PPO" not in stream.getvalue()
    fallback = RLDashboard(tmp_path, "cpu", mode="on", console=Console(file=io.StringIO()))
    assert not fallback.enabled


def test_monitor_persists_original_baselines_and_updates(tmp_path, monkeypatch):
    cfg = minihack_config(7, tmp_path / "tensorboard")
    trainer = DreamerTrainer(cfg)
    resumed = None
    path = tmp_path / "latest.pt"
    evaluations = []

    def evaluate(cfg, world, actor, seeds, random_policy=False, should_stop=None):
        evaluations.append(random_policy)
        if random_policy:
            return [0.0, 1.0]
        # Initial versus trained evaluation is controlled by completed updates.
        return [-1.0, 0.0] if trainer.completed_updates == 0 else [2.0, 3.0]

    monkeypatch.setattr("dreamerrl.training.run_monitor.evaluate_minihack", evaluate)
    try:
        with MiniHackRunMonitor(trainer, tmp_path, eval_every=1, eval_episodes=2, dashboard="off") as monitor:
            monitor.initialize()
            trainer.train(2, checkpoint_path=path, checkpoint_every=1, on_update=monitor.on_update)
            monitor.finish()
        events = read_events(tmp_path)
        assert len([r for r in events if r["event"] == "train"]) == 2
        assert any(r["event"] == "episode" for r in events)
        assert any(r["event"] == "checkpoint" for r in events)
        evaluated = [r for r in events if r["event"] == "evaluation"]
        assert len(evaluated) == 2 and evaluated[-1]["trained_mean"] == 2.5
        assert evaluated[-1]["untrained_mean"] == -0.5
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "run.log").exists()
        resumed = DreamerTrainer.from_checkpoint(path, cuda=False)
        original = resumed.run_state["minihack_monitor"]["baselines"]
        evaluations.clear()
        with MiniHackRunMonitor(resumed, tmp_path, eval_every=1, eval_episodes=2, dashboard="off") as monitor:
            monitor.initialize()
            assert evaluations == []  # resumed models must never be relabeled as untrained
            assert monitor.state["baselines"] == original
        assert len([r for r in read_events(tmp_path) if r["event"] == "run_start"]) == 2
        assert trainer.on_episode is None and trainer.on_checkpoint is None
    finally:
        close(trainer)
        if resumed is not None:
            close(resumed)


def test_stop_callback_saves_at_completed_update_boundary(tmp_path):
    trainer = DreamerTrainer(minihack_config(0, tmp_path / "logs"))
    try:
        updates = []

        def stop(record):
            updates.append(record["update"])
            trainer.run_state["stopped"] = True
            return False

        path = tmp_path / "latest.pt"
        trainer.train(5, checkpoint_path=path, checkpoint_every=100, on_update=stop)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert updates == [1]
        assert saved["completed_updates"] == 1 and saved["total_updates_target"] == 5
        assert saved["run_state"]["stopped"]
    finally:
        close(trainer)


def test_evaluation_preserves_all_rngs_and_handles_stop(tmp_path):
    trainer = DreamerTrainer(minihack_config(0, tmp_path / "logs"))
    try:
        before_python = random.getstate()
        before_numpy = np.random.get_state(legacy=True)
        before_torch = torch.get_rng_state()
        assert isinstance(before_numpy, tuple)
        results = evaluate_minihack(trainer.cfg, trainer.world, trainer.actor, [10000, 10001])
        assert len(results) == 2 and math.isfinite(mean(results))
        assert random.getstate() == before_python
        actual_numpy = np.random.get_state(legacy=True)
        assert isinstance(actual_numpy, tuple)
        assert actual_numpy[0] == before_numpy[0] and actual_numpy[2:] == before_numpy[2:]
        np.testing.assert_array_equal(actual_numpy[1], before_numpy[1])
        assert torch.equal(torch.get_rng_state(), before_torch)
        with pytest.raises(InterruptedError):
            evaluate_minihack(trainer.cfg, trainer.world, trainer.actor, [10000], should_stop=lambda: True)
        assert trainer.world.training and trainer.actor.training
        assert random.getstate() == before_python
        assert torch.equal(torch.get_rng_state(), before_torch)
    finally:
        close(trainer)


def test_final_evaluation_rejects_validation_seeds(tmp_path):
    from dreamerrl.scripts.evaluate_minihack import evaluate_checkpoints

    trainer = DreamerTrainer(minihack_config(0, tmp_path / "logs"))
    try:
        trainer.run_state["minihack_monitor"] = {"baselines": {"spec": {"evaluation_seeds": [10000, 10001]}}}
        path = tmp_path / "initial.pt"
        trainer.save_checkpoint(path)
        with pytest.raises(ValueError, match="overlap"):
            evaluate_checkpoints(path, path, episodes=2, seed_start=10000, cpu=True)
        result = evaluate_checkpoints(path, path, episodes=2, seed_start=20000, cpu=True)
        assert result["trained_returns"] == result["untrained_returns"]
        assert result["vs_untrained"]["mean_difference"] == 0
    finally:
        close(trainer)


def test_cli_sigterm_stops_at_checkpoint_boundary(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    run_dir = tmp_path / "run"
    command = [
        sys.executable,
        "-m",
        "dreamerrl.scripts.run_minihack",
        "--cpu",
        "--updates",
        "1000",
        "--num-envs",
        "2",
        "--episode-steps",
        "6",
        "--collect-steps",
        "3",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--imagination-horizon",
        "3",
        "--eval-episodes",
        "1",
        "--eval-every",
        "0",
        "--run-dir",
        str(run_dir),
        "--dashboard",
        "off",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            path = run_dir / "metrics.jsonl"
            if path.exists():
                records = []
                for line in path.read_text().splitlines():
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        pass  # writer may still be finishing its latest line
                if any(r["event"] == "train" for r in records):
                    break
            if process.poll() is not None:
                output, _ = process.communicate()
                pytest.fail(f"CLI exited before its first update: {output}")
            time.sleep(0.02)
        else:
            pytest.fail("CLI did not publish a training update")
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=30)
        assert process.returncode == 0, output
        checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        assert 0 < checkpoint["completed_updates"] < 1000
        events = read_events(run_dir)
        last = [r for r in events if r["event"] == "run_end"][-1]
        assert last["status"] == "stopped"
        assert checkpoint["completed_updates"] == last["update"]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
