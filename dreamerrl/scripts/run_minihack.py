"""Offline MiniHack runs with Rich, JSONL, checkpoints and fixed-seed evaluation."""

import argparse
import math
import signal
import traceback
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import torch

from dreamerrl.env.minihack.minihack_utils import glyph_entity_ids, glyph_scoring_schema, glyph_vocab_size
from dreamerrl.eval.corridor_diagnostics import CORRIDOR_R2
from dreamerrl.training.run_monitor import MiniHackRunMonitor, write_json
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


def main(
    *, default_env="MiniHack-Room-5x5-v0", default_episode_steps=200,
    on_update=None, argv=None, frozen_world=None, goal_normalizer=None,
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=default_env)
    parser.add_argument(
        "--action-cost",
        type=float,
        default=None,
        help="Nonnegative cost per training action; default 0, resume uses saved cost",
    )
    parser.add_argument(
        "--glyph-model",
        choices=["spatial", "mlp"],
        default="spatial",
        help="Categorical spatial glyph model for new runs; mlp selects the legacy baseline",
    )
    parser.add_argument(
        "--glyph-changed-weight",
        "--glyph-salient-weight",
        type=float,
        default=None,
        help="Weighted-mode multiplier for salient cells (default 2 for new runs)",
    )
    parser.add_argument(
        "--glyph-foreground",
        choices=["none", "entities"],
        help="Foreground glyphs: new runs default to entities (player/monsters/objects)",
    )
    parser.add_argument(
        "--glyph-loss-balance",
        choices=["weighted", "groups"],
        help="New runs default to weighted: modest extra weight for salient cells",
    )
    parser.add_argument(
        "--glyph-salient-fraction", type=float, help="Changed/foreground-cell budget in groups mode; default 0.5"
    )
    parser.add_argument(
        "--reconstruction-scale", type=float, help="Observation loss scale relative to other heads; default 1"
    )
    parser.add_argument(
        "--local-glyph-branch", action="store_true", help="Opt into a player-centered 3x3 glyph encoder branch"
    )
    parser.add_argument("--actor-mode", choices=["flat", "hierarchical", "director"])
    parser.add_argument("--director-goal-progress", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=None, help="Total desired updates, including resumed updates")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--cpu", action="store_true")
    device.add_argument("--require-cuda", action="store_true", help="Fail instead of falling back to CPU")
    parser.add_argument("--amp", action="store_true", help="Opt into CUDA FP16; default is FP32")
    parser.add_argument("--finite-horizon", action="store_true", help="Finite task ends with observable remaining time")
    parser.add_argument(
        "--run-dir", type=Path, help="Persistent run directory; resume defaults to checkpoint's directory"
    )
    parser.add_argument("--dashboard", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint; uses saved training configuration")
    parser.add_argument(
        "--eval-every", type=int, default=None, help="Completed updates between evaluations; 0 = final only"
    )
    parser.add_argument("--eval-episodes", type=int, default=None, help="Fixed validation levels, default 20")
    parser.add_argument("--eval-seed-start", type=int, default=None)
    parser.add_argument(
        "--stair-diagnostics",
        action="store_true",
        help="Log first stair opportunities in greedy and sampled R2 evaluation",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument(
        "--burn-in", type=int, default=16, help="Replay context steps before each training window; 0 disables"
    )
    parser.add_argument(
        "--acting-context",
        type=int,
        help="Refresh acting state: -1 full episode (default), 0 disabled, positive history cap",
    )
    parser.add_argument("--collect-steps", type=int, default=16, help="Steps per environment between updates")
    parser.add_argument("--episode-steps", type=int, default=default_episode_steps)
    parser.add_argument("--imagination-horizon", type=int, default=15)
    parser.add_argument("--discount", type=float, help="Return discount in (0, 1]; default 0.99")
    parser.add_argument("--lambda-return", type=float, help="Lambda-return mixture in [0, 1]; default 0.95")
    parser.add_argument(
        "--replay-value-scale", type=float, help="Recorded-reward critic loss scale; default 0 (disabled)"
    )
    parser.add_argument(
        "--random-exploration-steps", type=int, help="Initial random environment transitions; default 1000"
    )
    parser.add_argument("--warmup-updates", type=int, default=500)
    args = parser.parse_args(argv)
    for name in [
        "checkpoint_every",
        "num_envs",
        "batch_size",
        "seq_len",
        "collect_steps",
        "episode_steps",
        "imagination_horizon",
    ]:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.glyph_changed_weight is not None and (
        not math.isfinite(args.glyph_changed_weight) or args.glyph_changed_weight < 1
    ):
        parser.error("--glyph-changed-weight must be finite and at least 1")
    if args.glyph_salient_fraction is not None and (
        not math.isfinite(args.glyph_salient_fraction) or not 0 < args.glyph_salient_fraction < 1
    ):
        parser.error("--glyph-salient-fraction must be finite and in (0, 1)")
    if args.reconstruction_scale is not None and (
        not math.isfinite(args.reconstruction_scale) or args.reconstruction_scale < 0
    ):
        parser.error("--reconstruction-scale must be finite and nonnegative")
    if args.glyph_loss_balance == "groups" and args.glyph_changed_weight not in (None, 1):
        parser.error("groups uses --glyph-salient-fraction; --glyph-changed-weight must be 1")
    if args.discount is not None and (not math.isfinite(args.discount) or not 0 < args.discount <= 1):
        parser.error("--discount must be finite and in (0, 1]")
    if args.lambda_return is not None and (not math.isfinite(args.lambda_return) or not 0 <= args.lambda_return <= 1):
        parser.error("--lambda-return must be finite and in [0, 1]")
    if args.replay_value_scale is not None and (
        not math.isfinite(args.replay_value_scale) or args.replay_value_scale < 0
    ):
        parser.error("--replay-value-scale must be finite and nonnegative")
    if args.random_exploration_steps is not None and args.random_exploration_steps < 0:
        parser.error("--random-exploration-steps must be nonnegative")
    if args.acting_context is not None and args.acting_context < -1:
        parser.error("--acting-context must be -1, 0, or positive")
    if args.burn_in < 0:
        parser.error("--burn-in must be nonnegative")
    if args.action_cost is not None and (not math.isfinite(args.action_cost) or args.action_cost < 0):
        parser.error("--action-cost must be finite and nonnegative")
    if args.updates is not None and args.updates < 1:
        parser.error("--updates must be positive")
    if args.warmup_updates < 0 or (args.eval_every is not None and args.eval_every < 0):
        parser.error("Warmup and evaluation intervals must be nonnegative")
    if args.eval_episodes is not None and args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("CUDA is required but unavailable in this Python environment")
    if args.resume:
        run_dir = args.run_dir or args.resume.parent
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = args.run_dir or (
            args.checkpoint.parent
            if args.checkpoint
            else Path("runs") / f"{args.env}_seed{args.seed}_{stamp}"
        )
        if (run_dir / "config.json").exists() or (args.checkpoint and args.checkpoint.exists()):
            parser.error("Run already exists: use --resume or choose a new --run-dir/--checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = None
    old_handlers = {}
    try:
        if args.resume:
            trainer = DreamerTrainer.from_checkpoint(
                args.resume, cuda=False if args.cpu else True if args.require_cuda else None, log_dir=run_dir
            )
        else:
            cfg = make_default_config()
            cfg.env.env_id, cfg.env.seed = args.env, args.seed
            cfg.env.num_envs, cfg.env.max_episode_steps = args.num_envs, args.episode_steps
            cfg.env.parallel, cfg.env.finite_horizon = args.parallel, args.finite_horizon
            cfg.world.disable_aux_losses = True
            cfg.world.aux_objectives = []
            cfg.world.deter_size, cfg.world.stoch_size, cfg.world.num_classes = 128, 32, 32
            cfg.world.hidden_size, cfg.world.imagination_horizon = 256, args.imagination_horizon
            cfg.train.action_cost = args.action_cost if args.action_cost is not None else 0.0
            cfg.train.seed, cfg.train.cuda = args.seed, torch.cuda.is_available() and not args.cpu
            cfg.train.use_amp = args.amp and cfg.train.cuda
            cfg.train.enable_fused_adamw_kernels = False
            cfg.train.batch_size, cfg.train.seq_len = args.batch_size, args.seq_len
            cfg.train.burn_in = args.burn_in
            if args.acting_context is not None:
                cfg.train.acting_context = args.acting_context
            cfg.train.collect_steps, cfg.train.warmup_steps = args.collect_steps, args.warmup_updates
            cfg.train.model_lr = cfg.train.actor_lr = cfg.train.critic_lr = 3e-4
            cfg.train.random_exploration_steps = (
                args.random_exploration_steps if args.random_exploration_steps is not None else 1000
            )
            cfg.train.freeze_actor_critic_steps = 0
            cfg.ac.hierarchical = args.actor_mode in ("hierarchical", "director")
            cfg.ac.director_goals = args.actor_mode == "director"
            cfg.ac.director_goal_progress = bool(args.director_goal_progress)
            if args.discount is not None:
                cfg.ac.discount = args.discount
            if args.lambda_return is not None:
                cfg.ac.lambda_ = args.lambda_return
            if args.replay_value_scale is not None:
                cfg.ac.replay_value_scale = args.replay_value_scale
            cfg.log.enable_wandb = False
            cfg.log.tb_logdir, cfg.log.run_name = str(run_dir / "tensorboard"), "train"
            if args.glyph_model == "spatial":
                with gym.make(cfg.env.env_id) as probe:
                    observation, _ = probe.reset(seed=cfg.env.seed)
                    rows, columns = observation["glyphs"].shape
                    cfg.world.glyph_shape = (int(rows), int(columns))
                cfg.world.glyph_vocab_size = glyph_vocab_size()
                cfg.world.glyph_loss_balance = args.glyph_loss_balance or "weighted"
                foreground_policy = args.glyph_foreground or "entities"
                cfg.world.glyph_foreground_ids = glyph_entity_ids() if foreground_policy == "entities" else ()
                cfg.world.glyph_changed_weight = (
                    args.glyph_changed_weight
                    if args.glyph_changed_weight is not None
                    else (1.0 if cfg.world.glyph_loss_balance == "groups" else 2.0)
                )
                if args.glyph_salient_fraction is not None:
                    cfg.world.glyph_salient_fraction = args.glyph_salient_fraction
            if args.reconstruction_scale is not None:
                cfg.world.reconstruction_scale = args.reconstruction_scale
            if args.local_glyph_branch:
                if cfg.world.glyph_shape is None:
                    parser.error("--local-glyph-branch requires --glyph-model spatial")
                cfg.world.glyph_local_branch = True
                cfg.world.glyph_player_ids = tuple(glyph_scoring_schema()[1])
            if frozen_world is not None:
                if cfg.world != frozen_world["config"].world or cfg.env.env_id != frozen_world["config"].env.env_id:
                    parser.error("Source world configuration differs from the resolved MiniHack configuration")
                cfg.train.freeze_world_model = True
                cfg.ac.director_goal_standardized = goal_normalizer is not None
            trainer = DreamerTrainer(cfg, world_weights=frozen_world["world"] if frozen_world is not None else None)
            if goal_normalizer is not None:
                trainer.actor.goal_autoencoder.set_normalization(*goal_normalizer)
        if args.resume and args.director_goal_progress and not trainer.cfg.ac.director_goal_progress:
            parser.error("--director-goal-progress must match the checkpoint")
        if args.resume and args.actor_mode is not None:
            saved_mode = (
                "director" if trainer.cfg.ac.director_goals else "hierarchical" if trainer.cfg.ac.hierarchical else "flat"
            )
            if args.actor_mode != saved_mode:
                parser.error("--actor-mode must match the checkpoint")
        if args.resume and args.local_glyph_branch and not getattr(trainer.cfg.world, "glyph_local_branch", False):
            parser.error("--local-glyph-branch differs from the checkpoint; use a fresh run")
        if args.resume and args.action_cost is not None and args.action_cost != trainer.cfg.train.action_cost:
            parser.error("--action-cost must match the checkpoint; use a fresh run to change the reward objective")
        if args.resume and args.glyph_foreground is not None:
            expected_ids = glyph_entity_ids() if args.glyph_foreground == "entities" else ()
            if expected_ids != trainer.cfg.world.glyph_foreground_ids:
                parser.error("--glyph-foreground must match the checkpoint; use a fresh run to change it")
        for name, saved_value in (
            ("glyph_loss_balance", trainer.cfg.world.glyph_loss_balance),
            ("glyph_salient_fraction", trainer.cfg.world.glyph_salient_fraction),
            ("glyph_changed_weight", trainer.cfg.world.glyph_changed_weight),
            ("reconstruction_scale", trainer.cfg.world.reconstruction_scale),
            ("acting_context", trainer.cfg.train.acting_context),
            ("discount", trainer.cfg.ac.discount),
            ("lambda_return", trainer.cfg.ac.lambda_),
            ("replay_value_scale", trainer.cfg.ac.replay_value_scale),
            ("random_exploration_steps", trainer.cfg.train.random_exploration_steps),
        ):
            if args.resume and getattr(args, name) is not None and getattr(args, name) != saved_value:
                parser.error(f"--{name.replace('_', '-')} must match the checkpoint; use a fresh run to change it")
        target = args.updates if args.updates is not None else (trainer.total_updates_target or 10000)
        if target < trainer.completed_updates:
            parser.error("--updates is below the checkpoint's completed update count")
        trainer.total_updates_target = target
        checkpoint = args.checkpoint or (args.resume if args.resume and args.run_dir is None else run_dir / "latest.pt")
        if args.stair_diagnostics and trainer.cfg.env.env_id != CORRIDOR_R2:
            parser.error("--stair-diagnostics requires MiniHack Corridor R2")
        saved = trainer.run_state.get("minihack_monitor", {})
        saved_seeds = saved.get("baselines", {}).get("spec", {}).get("evaluation_seeds", [])
        episodes = args.eval_episodes if args.eval_episodes is not None else (len(saved_seeds) or 20)
        eval_seed = (
            args.eval_seed_start if args.eval_seed_start is not None else (saved_seeds[0] if saved_seeds else 10000)
        )
        interval = args.eval_every if args.eval_every is not None else saved.get("eval_every", 50)
        with MiniHackRunMonitor(
            trainer,
            run_dir,
            eval_every=interval,
            eval_episodes=episodes,
            eval_seed_start=eval_seed,
            dashboard=args.dashboard,
            stair_diagnostics=args.stair_diagnostics,
        ) as monitor:

            def request_stop(signum, frame):
                monitor.stop_requested = True

            for sig in [signal.SIGINT, signal.SIGTERM]:
                old_handlers[sig] = signal.signal(sig, request_stop)
            try:
                monitor.initialize()
                if trainer.completed_updates == 0 and not (run_dir / "initial.pt").exists():
                    trainer.save_checkpoint(run_dir / "initial.pt")
                def update(record):
                    if on_update is not None:
                        on_update(run_dir, trainer)
                    return monitor.on_update(record)

                trainer.train(
                    target,
                    checkpoint_path=checkpoint,
                    checkpoint_every=args.checkpoint_every,
                    on_update=update,
                )
                monitor.finish()
            except InterruptedError:
                monitor.stop_requested = True
                monitor.finish()
            trainer.save_checkpoint(checkpoint)
    except Exception as exc:
        write_json(
            run_dir / "failure.json",
            dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()),
        )
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if trainer is not None:
            trainer.env.venv.close()
            trainer.tb.close()


if __name__ == "__main__":
    main()
