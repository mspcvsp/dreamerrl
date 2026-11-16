"""Offline MiniHack run journal, fixed-seed evaluations, and terminal dashboard."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from dreamerrl.eval.minihack_eval import compare_returns, evaluate_minihack
from dreamerrl.logging.jsonl_logger import JSONLLogger
from dreamerrl.logging.repo_logger import create_run_logger
from dreamerrl.utils.rl_dashboard import RLDashboard


def write_json(path, value):
    """Atomic small metadata files, separate from append-only event metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def configuration(cfg):
    result = asdict(cfg)
    result["world"]["aux_objectives"] = [objective.name for objective in cfg.world.aux_objectives]
    return result


def runtime_info(device):
    packages = {}
    for name in ["torch", "gymnasium", "minihack", "nle", "numpy", "rich"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, timeout=2
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True, timeout=2
            ).stdout
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        commit, dirty = None, None
    result: dict[str, Any] = dict(
        python=sys.version,
        pid=os.getpid(),
        architecture=platform.machine(),
        packages=packages,
        device=str(device),
        torch_threads=torch.get_num_threads(),
        cuda_build=torch.version.cuda,
        git_commit=commit,
        git_dirty=dirty,
    )
    if device.type == "cuda":
        result.update(
            gpu=torch.cuda.get_device_name(device),
            capability=list(torch.cuda.get_device_capability(device)),
            compiled_architectures=torch.cuda.get_arch_list(),
        )
    return result


class MiniHackRunMonitor:
    def __init__(
        self,
        trainer,
        run_dir,
        *,
        eval_every=50,
        eval_episodes=20,
        eval_seed_start=10000,
        dashboard="auto",
        console=None,
    ):
        if eval_every < 0 or eval_episodes < 1 or eval_seed_start < 0:
            raise ValueError("Evaluation interval/seed must be nonnegative and episode count positive")
        self.trainer, self.run_dir = trainer, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every = eval_every
        self.seeds = list(range(eval_seed_start, eval_seed_start + eval_episodes))
        self.session = uuid.uuid4().hex
        self.started = time.perf_counter()
        self.stop_requested = False
        self.update_seconds_ema = None
        self.dashboard = RLDashboard(self.run_dir, trainer.device, dashboard, console)
        self.logger = create_run_logger(self.run_dir, self.dashboard.console)
        self.jsonl = JSONLLogger(self.run_dir / "metrics.jsonl")
        self.state = trainer.run_state.setdefault("minihack_monitor", {})
        self.last_record: dict[str, Any] = dict(
            update=trainer.completed_updates,
            target_updates=trainer.total_updates_target or 0,
            env_steps=trainer.total_env_steps,
            episodes=trainer.completed_episodes,
            replay_size=trainer.replay.size,
        )
        self.old_episode_callback = trainer.on_episode
        self.old_checkpoint_callback = trainer.on_checkpoint

    def event(self, record):
        self.jsonl.log({"session": self.session, **record})

    def __enter__(self):
        self.dashboard.__enter__()
        self.trainer.on_episode = self.event
        self.trainer.on_checkpoint = self.event
        cfg = configuration(self.trainer.cfg)
        if not (self.run_dir / "config.json").exists():
            write_json(self.run_dir / "config.json", cfg)
        self.event(
            dict(
                event="run_start",
                update=self.trainer.completed_updates,
                env_steps=self.trainer.total_env_steps,
                config=cfg,
                runtime=runtime_info(self.trainer.device),
            )
        )
        self.logger.info("Run directory: %s", self.run_dir)
        self.logger.info("PID %d | device %s", os.getpid(), self.trainer.device)
        return self

    def initialize(self):
        cfg = configuration(self.trainer.cfg)
        spec = dict(environment=cfg["env"], world=cfg["world"], actor_critic=cfg["ac"], evaluation_seeds=self.seeds)
        baseline = self.state.get("baselines")
        path = self.run_dir / "baselines.json"
        if baseline is None and path.exists():
            baseline = json.loads(path.read_text())
        if baseline is not None:
            # Older runs predate the optional replay critic loss (disabled then).
            baseline["spec"]["actor_critic"].setdefault("replay_value_scale", 0.0)
            # Missing fields describe the previous uniform/weighted objective.
            for name, default in (
                ("glyph_foreground_ids", ()),
                ("glyph_loss_balance", "weighted"),
                ("glyph_salient_fraction", 0.5),
                ("reconstruction_scale", 1.0),
            ):
                baseline["spec"]["world"].setdefault(name, default)
            if json.loads(json.dumps(baseline["spec"])) != json.loads(json.dumps(spec)):
                raise ValueError(
                    "Saved baselines use a different evaluation configuration; use the original seeds/episodes"
                )
            self.logger.info("Restored original random/untrained baselines")
        else:
            self.dashboard.update(self.last_record, phase="evaluating random baseline")
            self.logger.info("Evaluating random baseline on %d fixed levels", len(self.seeds))
            random_returns = self.evaluate(random=True)
            untrained = None
            if self.trainer.completed_updates == 0:
                self.dashboard.update(phase="evaluating untrained baseline")
                untrained = self.evaluate()
            else:
                self.logger.warning("Legacy checkpoint has no untrained baseline; leaving it unavailable")
            baseline = dict(spec=spec, random_returns=random_returns, untrained_returns=untrained)
        self.state["baselines"] = baseline
        self.state["eval_every"] = self.eval_every
        write_json(path, baseline)
        self.event(dict(event="baselines", **baseline))
        self.dashboard.update(self.last_record, self.state.get("last_evaluation"), phase="training")

    def evaluate(self, random=False):
        return evaluate_minihack(
            self.trainer.cfg,
            self.trainer.world,
            self.trainer.actor,
            self.seeds,
            random_policy=random,
            should_stop=lambda: self.stop_requested,
        )

    def evaluation(self):
        update = self.trainer.completed_updates
        self.dashboard.update(phase="evaluating trained policy")
        self.logger.info("Evaluating update %d on %d fixed levels", update, len(self.seeds))
        started = time.perf_counter()
        returns = self.evaluate()
        baseline = self.state["baselines"]
        random_returns, untrained = baseline["random_returns"], baseline["untrained_returns"]
        record: dict[str, Any] = dict(
            event="evaluation",
            update=update,
            env_steps=self.trainer.total_env_steps,
            evaluation_seeds=self.seeds,
            trained_returns=returns,
            trained_mean=mean(returns),
            random_mean=mean(random_returns),
            untrained_mean=mean(untrained) if untrained is not None else None,
            vs_random=compare_returns(returns, random_returns),
            vs_untrained=compare_returns(returns, untrained) if untrained is not None else None,
            evaluation_seconds=time.perf_counter() - started,
        )
        self.state["last_evaluation"] = record
        self.event(record)
        write_json(self.run_dir / "evaluation.json", record)
        self.dashboard.update(self.last_record, record, phase="training")
        self.logger.info(
            "Validation mean %.4f; random %.4f; untrained %s",
            record["trained_mean"],
            record["random_mean"],
            record["untrained_mean"],
        )
        self.trainer.tb.add_scalar("eval/trained_return", record["trained_mean"], update)
        if record["trained_mean"] > self.state.get("best_mean", -math.inf):
            self.state["best_mean"] = record["trained_mean"]
            self.state["best_update"] = update
            self.trainer.save_checkpoint(self.run_dir / "best.pt")

    def on_update(self, record):
        scalar_metrics = [record[key] for key in ["world_loss", "actor_loss", "critic_loss", "kl_dyn", "kl_rep"]]
        if not all(math.isfinite(value) for value in scalar_metrics):
            raise FloatingPointError("Non-finite training metrics")
        self.last_record = {**record, "session_elapsed_seconds": time.perf_counter() - self.started}
        remaining = record["target_updates"] - record["update"]
        duration = record["update_seconds"]
        self.update_seconds_ema = (
            duration if self.update_seconds_ema is None else 0.9 * self.update_seconds_ema + 0.1 * duration
        )
        self.last_record["eta_training_seconds"] = remaining * self.update_seconds_ema
        self.event(self.last_record)
        self.dashboard.update(self.last_record, phase="stopping after update" if self.stop_requested else "training")
        if not self.dashboard.enabled and (record["update"] == 1 or record["update"] % 10 == 0):
            self.logger.info(
                "update=%d/%d env_steps=%d model=%.4f recent_return=%s",
                record["update"],
                record["target_updates"],
                record["env_steps"],
                record["world_loss"],
                record["recent_return_mean"],
            )
        if self.stop_requested:
            return False
        if record["update"] == record["target_updates"] or (
            self.eval_every and record["update"] % self.eval_every == 0
        ):
            try:
                self.evaluation()
            except InterruptedError:
                self.event(dict(event="evaluation_interrupted", update=record["update"]))
                return False
        return True

    def finish(self):
        if (
            not self.stop_requested
            and self.state.get("last_evaluation", {}).get("update") != self.trainer.completed_updates
        ):
            self.evaluation()
        status = "stopped" if self.stop_requested else "completed"
        self.event(
            dict(
                event="run_end",
                status=status,
                update=self.trainer.completed_updates,
                env_steps=self.trainer.total_env_steps,
            )
        )
        self.dashboard.update(phase=status)
        self.logger.info("Run %s at update %d", status, self.trainer.completed_updates)

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None:
                self.event(
                    dict(
                        event="run_error",
                        update=self.trainer.completed_updates,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback="".join(traceback.format_exception(exc_type, exc, tb)),
                    )
                )
                self.logger.error("Run failed", exc_info=(exc_type, exc, tb))
        finally:
            self.trainer.on_episode = self.old_episode_callback
            self.trainer.on_checkpoint = self.old_checkpoint_callback
            self.dashboard.__exit__(exc_type, exc, tb)
            self.jsonl.close()
            for handler in self.logger.handlers:
                handler.close()
            self.logger.handlers.clear()
