"""One prebudgeted, single-seed CAGE2 learning curve with isolated paired evaluation."""

import argparse
import hashlib
import json
import math
import pickle
import random
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from experiment_protocol import original_schedule, verify_original_inputs
from learner_cycle import optimizer_steps, state_hash

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
from dreamerrl.models.hierarchical_actor import policy_action
from dreamerrl.training.trainer import DreamerTrainer

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
SEEDS = list(range(1_000_000_000, 1_000_000_020))


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def rng_equal(a, b):
    return (
        pickle.dumps(a[:2]) == pickle.dumps(b[:2])
        and torch.equal(a[2], b[2])
        and all(torch.equal(x, y) for x, y in zip(a[3], b[3]))
    )


@torch.no_grad()
def evaluate(trainer, policy, seeds=SEEDS) -> dict[str, Any]:
    saved = rng_state()
    replay_before = (len(trainer.replay.episodes), trainer.replay.size, trainer.sample_step, trainer.total_env_steps)
    hashes = {k: state_hash(getattr(trainer, k)) for k in ("world", "actor", "critic")}
    training = trainer.world.training, trainer.actor.training
    old_mode = trainer.world.posterior.deterministic_latent_for_tests
    rows = []
    started = time.perf_counter()
    try:
        trainer.world.eval()
        trainer.actor.eval()
        trainer.world.posterior.deterministic_latent_for_tests = True
        for seed in seeds:
            policy_seed = seed + 100_000
            random.seed(policy_seed)
            np.random.seed(policy_seed)
            torch.manual_seed(policy_seed)
            env = Cage2BlueEnv(replace(trainer.cfg.env, seed=seed), trainer.device)
            output = env.reset(seed=seed)
            # Environment construction seeds native RNGs; policy randomness uses its own generator.
            generator = np.random.default_rng(policy_seed)
            state = trainer.world.observe_step(
                trainer.world.init_state(1),
                output["state"],
                torch.zeros(1, trainer.action_dim, device=trainer.device),
                is_first=torch.ones(1, dtype=torch.bool, device=trainer.device),
            )["post"]
            actions, rewards, components = [], [], []
            policy_state = None
            for step in range(30):
                if policy == "actor":
                    selected, _, policy_state = policy_action(trainer.actor, state, policy_state, deterministic=True)
                    action = int(selected.item())
                elif policy == "sleep":
                    action = next(x["index"] for x in env.action_mapping if x["name"] == "Sleep")
                else:
                    action = int(generator.integers(trainer.action_dim))
                output = env.step(torch.tensor([action], device=trainer.device))
                actions.append(env.action_mapping[action])
                rewards.append(env.journal[-1]["reward"])
                try:
                    breakdown = env.wrapper.get_reward_breakdown("Blue")
                    components.append(json.loads(json.dumps(breakdown, default=str)))
                except (AttributeError, NotImplementedError, TypeError):
                    components.append(None)
                if step < 29:
                    state = trainer.world.observe_step(
                        state,
                        output["state"],
                        torch.nn.functional.one_hot(
                            torch.tensor([action], device=trainer.device), trainer.action_dim
                        ).float(),
                        is_first=torch.zeros(1, dtype=torch.bool, device=trainer.device),
                    )["post"]
            assert output["is_last"].item() and not output["is_terminal"].item()
            rows.append(
                dict(
                    seed=seed,
                    policy_seed=policy_seed,
                    policy=policy,
                    steps=30,
                    total_return=sum(rewards),
                    rewards=rewards,
                    actions=actions,
                    native_reward_components=components,
                )
            )
            env.close()
    finally:
        trainer.world.posterior.deterministic_latent_for_tests = old_mode
        trainer.world.train(training[0])
        trainer.actor.train(training[1])
        restore_rng(saved)
    assert rng_equal(saved, rng_state())
    assert replay_before == (
        len(trainer.replay.episodes),
        trainer.replay.size,
        trainer.sample_step,
        trainer.total_env_steps,
    )
    assert hashes == {k: state_hash(getattr(trainer, k)) for k in hashes}
    return dict(
        episodes=rows,
        mean_return=float(np.mean([r["total_return"] for r in rows])),
        seconds=time.perf_counter() - started,
        training_rng_unchanged=True,
        training_replay_unchanged=True,
        evaluation_mode="greedy actions, posterior modes" if policy == "actor" else policy,
    )


def summary(report, baseline) -> dict[str, Any]:
    returns = np.array([x["total_return"] for x in report["episodes"]])
    reference = np.array([x["total_return"] for x in baseline["episodes"]])
    assert [x["seed"] for x in report["episodes"]] == SEEDS
    differences = returns - reference
    generator = np.random.default_rng(20261123)
    means = differences[generator.integers(0, 20, size=(20000, 20))].mean(1)
    operations, hosts = Counter(), Counter()
    for row in report["episodes"]:
        for action in row["actions"]:
            operations[action["name"]] += 1
            fields = action["description"].split()
            if len(fields) > 1:
                hosts[fields[-1]] += 1
    return dict(
        mean_return=float(returns.mean()),
        seed_return_sd=float(returns.std(ddof=1)),
        paired_mean_change=float(differences.mean()),
        paired_ci95=np.quantile(means, [0.025, 0.975]).tolist(),
        per_seed_differences=dict(zip(map(str, SEEDS), differences.tolist())),
        operation_counts=dict(operations),
        operation_frequencies={k: v / 600 for k, v in operations.items()},
        target_host_counts=dict(hosts.most_common()),
        restore_count=operations["Restore"],
        restore_frequency=operations["Restore"] / 600,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/cage2_blue_cycle_seed0_20261123/latest.pt")
    parser.add_argument("--schedule", choices=["original", "measured"], default="measured")
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and torch.cuda.is_available(), "Use uv run --locked"
    torch.set_num_threads(2)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    if args.schedule == "original":
        verify_original_inputs(source, None)
    trainer = DreamerTrainer.from_checkpoint(source, cuda=True, log_dir=run / "tensorboard")
    if args.schedule == "original":
        verify_original_inputs(source, trainer.cfg.to_dict())
    env = trainer.env
    assert isinstance(env, Cage2BlueEnv)
    assert trainer.completed_updates == 1 and trainer.total_env_steps == 510
    assert trainer.env.obs_dim == 52 and trainer.action_dim == 145 and trainer.cfg.env.max_episode_steps == 30
    assert optimizer_steps(trainer) == {"model_opt": [1], "actor_opt": [1], "critic_opt": [1]}
    write(run / "resolved_config.json", trainer.cfg.to_dict())
    write(
        run / "protocol.json",
        dict(
            source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_update=1,
            schedule_mode=args.schedule,
            dev_seeds=SEEDS,
            training_seed=trainer.cfg.train.seed,
            policy_seed_rule="simulator seed + 100000; random policy uses independent numpy Generator",
            training_seeds=(
                "production reset resume: seed=total_env_steps (510), increment per episode; asserted disjoint"
            ),
            baseline_original_training_seeds=list(range(17)),
            evaluation_mode="greedy actions, posterior modes",
            budget_seconds=7200,
            calibration_updates=25,
            calibration_scheduler_target=100000,
            budget_rule=(
                "floor((7200 - elapsed setup/calibration/baselines - 4 * actor evaluation seconds "
                "- 120 reserve) / calibration seconds per update), plus 26; cap at 100000"
            ),
            checkpoint_rule=(
                "three quarter-spaced evaluations and final; latest every 100 updates; update 26 calibration checkpoint"
            ),
            bootstrap="20000 paired route-seed resamples, percentile 95%, RNG seed 20261123",
            conclusion="predeclared final; intermediate best exploratory; one training seed only",
        ),
    )
    write(
        run / "revisions.json",
        dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            cyborg=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT / "vendor/cage-challenge-2", text=True
            ).strip(),
            uv_lock_sha256=hashlib.sha256((PROJECT / "uv.lock").read_bytes()).hexdigest(),
            python=sys.executable,
            torch=torch.__version__,
        ),
    )
    (run / "code.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
    (run / "experiment_script.py").write_bytes(Path(__file__).read_bytes())
    (run / "rerun_command.txt").write_text(
        "cd "
        + shlex.quote(str(PROJECT))
        + " && "
        + shlex.join(
            [
                "bash",
                "--noprofile",
                "--norc",
                "run_learning_clean.sh",
                "--schedule",
                args.schedule,
                "--source",
                str(source),
                "--run-dir",
                str(run) + "_rerun",
            ]
        )
        + "\n"
    )
    write(run / "action_mapping.json", env.action_mapping)
    started = time.perf_counter()
    baselines = {}
    for policy in ("actor", "sleep", "random"):
        baselines[policy] = evaluate(trainer, policy)
        write(run / f"baseline_{policy}.json", baselines[policy])
        print("BASELINE", policy, baselines[policy]["mean_return"], flush=True)
    baseline = baselines["actor"]
    scorecard = [dict(update=1, env_steps=510, **summary(baseline, baseline))]
    write(run / "scorecard.json", scorecard)
    metrics = open(run / "metrics.jsonl", "a", buffering=1)
    seedlog = open(run / "training_seeds.jsonl", "a", buffering=1)

    def record(record):
        assert all(math.isfinite(record[k]) for k in ("world_loss", "actor_loss", "critic_loss"))
        assert not (set(row["seed"] for row in env.journal) & set(SEEDS))
        seedlog.write(
            json.dumps(dict(update=trainer.completed_updates, seeds=sorted(set(row["seed"] for row in env.journal))))
            + "\n"
        )
        env.journal.clear()
        metrics.write(json.dumps(record, default=str) + "\n")
        if trainer.completed_updates % 25 == 0:
            print("PROGRESS", trainer.completed_updates, trainer.total_env_steps, record["world_loss"], flush=True)

    calibration_start = time.perf_counter()

    def calibrate(record_):
        record(record_)
        return trainer.completed_updates < 26

    trainer.train(100000, checkpoint_path=run / "latest.pt", checkpoint_every=25, on_update=calibrate)
    seconds_per_update = (time.perf_counter() - calibration_start) / 25
    trainer.save_checkpoint(run / "update_26.pt")
    if args.schedule == "original":
        final, checkpoints = original_schedule()
    else:
        final = 26 + max(
            4, math.floor((7200 - (time.perf_counter() - started) - 4 * baseline["seconds"] - 120) / seconds_per_update)
        )
        final = min(final, 100000)
        checkpoints = sorted(set([26 + round((final - 26) * i / 4) for i in (1, 2, 3, 4)]))
    write(
        run / "budget.json",
        dict(
            calibration_seconds_per_update=seconds_per_update,
            elapsed_seconds=time.perf_counter() - started,
            actor_evaluation_seconds=baseline["seconds"],
            final_update=final,
            final_env_steps=510 + 30 * (final - 1),
            evaluation_updates=checkpoints,
            recorded_before_later_evaluations=True,
            schedule_mode=args.schedule,
        ),
    )
    print("BUDGET", final, checkpoints, seconds_per_update, flush=True)

    def callback(record_):
        record(record_)
        if trainer.completed_updates in checkpoints:
            trainer.save_checkpoint(run / f"update_{trainer.completed_updates}.pt")
            report = evaluate(trainer, "actor")
            write(run / f"evaluation_{trainer.completed_updates}.json", report)
            scorecard.append(
                dict(update=trainer.completed_updates, env_steps=trainer.total_env_steps, **summary(report, baseline))
            )
            write(run / "scorecard.json", scorecard)
            print(
                "EVALUATION",
                trainer.completed_updates,
                scorecard[-1]["mean_return"],
                scorecard[-1]["paired_ci95"],
                flush=True,
            )

    trainer.train(final, checkpoint_path=run / "latest.pt", checkpoint_every=100, on_update=callback)
    write(
        run / "complete.json",
        dict(
            final_update=final,
            final_env_steps=trainer.total_env_steps,
            optimizer_steps=optimizer_steps(trainer),
            elapsed_seconds=time.perf_counter() - started,
            baseline_means={k: v["mean_return"] for k, v in baselines.items()},
            main_result=scorecard[-1],
            exploratory_best=max(scorecard, key=lambda x: x["mean_return"]),
        ),
    )
    from report_learning import main as report_main

    report_main(["--run-dir", str(run)])
    print("COMPLETE", final, flush=True)
    metrics.close()
    seedlog.close()
    env.close()
    trainer.tb.close()


if __name__ == "__main__":
    main()
