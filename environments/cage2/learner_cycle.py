"""One real CAGE2 collect/train/checkpoint/fresh-process reload/evaluate cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.transforms import symlog
from dreamerrl.utils.twohot import twohot_encode
from dreamerrl.utils.types import make_default_config

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_copy(module):
    return {name: value.detach().cpu().clone() for name, value in module.named_parameters()}


def parameter_delta(before, module):
    return sum(float((value.detach().cpu() - before[name]).abs().sum()) for name, value in module.named_parameters())


def optimizer_steps(trainer):
    result = {}
    for name in ("model_opt", "actor_opt", "critic_opt"):
        states = getattr(trainer, name).state.values()
        values = sorted({int(state["step"].item()) for state in states if "step" in state})
        result[name] = values
    return result


def assert_replay_alignment(trainer):
    journal = trainer.env.journal
    episodes = trainer.replay.episodes
    assert len(journal) == len(episodes) * 30 == trainer.total_env_steps
    for index, episode in enumerate(episodes):
        assert len(episode["obs"]) == 31 and episode["is_first"][0] and not episode["is_first"][1:].any()
        assert episode["reward"][0] == 0 and episode["done"][-1] == 1 and not episode["done"][:-1].any()
        assert not episode["is_terminal"].any()
        for step in range(1, 31):
            row = journal[index * 30 + step - 1]
            assert row["step"] == step and row["seed"] == trainer.cfg.env.seed + index
            assert int(episode["action"][step]) == row["action"]
            assert abs(float(episode["reward"][step]) - row["reward"]) < 1e-5
            torch.testing.assert_close(
                episode["obs"][step - 1].cpu(), torch.tensor(row["observation_before"]), atol=0, rtol=0
            )
            torch.testing.assert_close(
                episode["obs"][step].cpu(), torch.tensor(row["observation_after"]), atol=0, rtol=0
            )
            assert row["wrapper_done"] == (step == 30) and not row["native_done"]
    return dict(
        episodes=len(episodes),
        transitions=len(journal),
        observation_rows_per_episode=31,
        is_first_only_at_reset=True,
        final_observation_preserved=True,
        cap_is_truncation=True,
        native_terminal_flags_all_false=True,
    )


def train(run, seed):
    run.mkdir(parents=True, exist_ok=False)
    cfg = make_default_config()
    cfg.env.env_id = "CAGE2-Blue-BLine-v0"
    cfg.env.num_envs, cfg.env.max_episode_steps, cfg.env.seed = 1, 30, seed
    cfg.env.parallel, cfg.env.finite_horizon = False, False
    cfg.train.seed, cfg.train.cuda = seed, True
    cfg.train.batch_size, cfg.train.seq_len, cfg.train.burn_in = 16, 16, 8
    cfg.train.collect_steps = 30
    cfg.train.random_exploration_steps = 100000
    cfg.train.freeze_actor_critic_steps, cfg.train.warmup_steps = 0, 0
    cfg.train.use_amp, cfg.train.enable_fused_adamw_kernels = False, False
    cfg.train.enable_wandb = False
    cfg.world.disable_aux_losses, cfg.world.aux_objectives = True, []
    cfg.log.tb_logdir, cfg.log.run_name = str(run / "tensorboard"), "train"
    cfg.log.enable_wandb = False
    write(run / "resolved_config.json", cfg.to_dict())
    write(
        run / "provenance.json",
        dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            cyborg_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT / "vendor/cage-challenge-2", text=True
            ).strip(),
            python=sys.executable,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            uv_lock_sha256=hashlib.sha256((PROJECT / "uv.lock").read_bytes()).hexdigest(),
            collection_policy="production collector uniform discrete warmup",
            collection_seed=seed,
            evaluation_seed=10000,
            evaluation_actions="greedy",
            evaluation_latents="posterior modes",
        ),
    )
    (run / "code.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
    (run / "rerun_command.txt").write_text(
        "cd " + str(PROJECT) + " && bash --noprofile --norc run_cycle_clean.sh --run-dir " + str(run) + "_rerun\n"
    )
    trainer = DreamerTrainer(cfg)
    assert trainer.device.type == "cuda" and isinstance(trainer.env, Cage2BlueEnv)
    env = trainer.env
    write(
        run / "interface.json",
        dict(
            scenario=str(env.scenario),
            wrapper="ChallengeWrapper(Blue,max_steps=30)",
            native_observation_shape=list(env.obs_space.shape),
            native_observation_dtype=env.native_dtype,
            adapter_dtype="float32",
            model_observation_dim=trainer.world.flat_obs_dim,
            discrete_action_count=trainer.action_dim,
            action_mapping=env.action_mapping,
            horizon="Wrapper sets done at 30; info.done retains native flag. is_last=True,is_terminal=False at cap; "
            "continuing-task bootstrap retained. If info.done is absent, done at cap is assumed truncation.",
        ),
    )
    try:
        started = time.perf_counter()
        for _ in range(16):
            trainer.collect_env_steps()
        collect_seconds = time.perf_counter() - started
        assert len(trainer.replay.episodes) == 16
        sample_events = []
        captured = {}
        reward_calls = []

        class ReplayObserver:
            def replay_sample(self, **values):
                sample_events.append({k: int(v) for k, v in values.items()})

            def replay_finalize(self, **values):
                pass

        trainer.replay.probe = ReplayObserver()
        original_sample = trainer.replay.sample

        def sample(*args, **kwargs):
            batch = original_sample(*args, **kwargs)
            captured.update({k: v.detach().cpu().clone() for k, v in batch.items() if isinstance(v, torch.Tensor)})
            return batch

        original_loss = trainer.world.reward_heads.main.loss_from_logits

        def reward_loss(logits, rewards):
            loss = original_loss(logits, rewards)
            with torch.no_grad():
                head = trainer.world.reward_heads.main
                target = twohot_encode(symlog(rewards), head.bin_values)
                per_token = -(target * logits.log_softmax(-1)).sum(-1)
                torch.testing.assert_close(per_token.mean(), loss.detach())
                reward_calls.append(
                    dict(
                        targets=rewards.detach().cpu().clone(),
                        twohot=target.cpu(),
                        per_token=per_token.cpu(),
                        mean=float(loss.detach()),
                    )
                )
            return loss

        before = {k: parameter_copy(getattr(trainer, k)) for k in ("world", "actor", "critic")}
        records = []
        started = time.perf_counter()
        with (
            patch.object(trainer.replay, "sample", sample),
            patch.object(trainer.world.reward_heads.main, "loss_from_logits", reward_loss),
        ):
            trainer.train(
                1,
                checkpoint_path=run / "latest.pt",
                checkpoint_every=1,
                on_update=lambda record: records.append(record),
            )
        update_seconds = time.perf_counter() - started
        assert trainer.completed_updates == 1 and len(records) == 1 and len(reward_calls) == 1
        assert len(sample_events) == 16 and captured["obs"].shape == (16, 16, 52)
        assert all(math.isfinite(records[0][k]) for k in ("world_loss", "actor_loss", "critic_loss", "reward_loss"))
        deltas = {k: parameter_delta(before[k], getattr(trainer, k)) for k in before}
        assert all(math.isfinite(v) and v > 0 for v in deltas.values())
        steps = optimizer_steps(trainer)
        assert all(values == [1] for values in steps.values())
        assert trainer._optimizer_steps == dict(world_model=1, actor=1, critic=1)
        alignment = assert_replay_alignment(trainer)
        row, column = map(int, (captured["reward"] != 0).nonzero()[0])
        event = sample_events[row]
        episode_index = event["ep_idx"]
        episode_step = event["start"] + column
        native = trainer.env.journal[episode_index * 30 + episode_step - 1]
        replay_reward = float(trainer.replay.episodes[episode_index]["reward"][episode_step])
        target = float(reward_calls[0]["targets"][row, column])
        assert episode_step > 0 and abs(native["reward"] - replay_reward) < 1e-5 and replay_reward == target != 0
        assert float(captured["reward"][row, column]) == target
        torch.testing.assert_close(
            captured["obs"][row, column], trainer.replay.episodes[episode_index]["obs"][episode_step].cpu()
        )
        trace = dict(
            episode_index=episode_index,
            episode_seed=native["seed"],
            transition_step=episode_step,
            action_id=native["action"],
            action_description=native["action_description"],
            environment_reward=native["reward"],
            replay_reward=replay_reward,
            batch_row=row,
            batch_column=column,
            sampler_start=event["start"],
            production_reward_target=target,
            symlog_target=float(symlog(torch.tensor(target))),
            twohot_target=reward_calls[0]["twohot"][row, column].tolist(),
            per_token_reward_ce=float(reward_calls[0]["per_token"][row, column]),
            production_mean_reward_loss=reward_calls[0]["mean"],
            production_loss_path="world_model_training_step -> reward_heads.main.loss_from_logits",
        )
        assert records[0]["reward_loss"] == reward_calls[0]["mean"]
        expectation = dict(
            model_hashes={k: state_hash(getattr(trainer, k)) for k in ("world", "actor", "critic")},
            optimizer_steps=steps,
            replay_episodes=len(trainer.replay.episodes),
            replay_size=trainer.replay.size,
            completed_updates=1,
            cadence_totals=trainer.cadence_totals,
        )
        trainer.save_checkpoint(run / "latest.pt")
        write(run / "checkpoint_expectation.json", expectation)
        write(run / "reward_trace.json", trace)
        write(run / "collection_journal.json", trainer.env.journal)
        write(
            run / "train_result.json",
            dict(
                losses={k: records[0][k] for k in ("world_loss", "reward_loss", "actor_loss", "critic_loss")},
                optimizer_steps=steps,
                parameter_change_l1=deltas,
                replay_alignment=alignment,
                cadence=records[0]["cadence"],
                collection_seconds=collect_seconds,
                one_update_seconds=update_seconds,
                real_blue_transitions=trainer.total_env_steps,
                policy="uniform_discrete_warmup",
                seed=seed,
            ),
        )
        print(
            "TRAIN_COMPLETE",
            records[0]["world_loss"],
            records[0]["actor_loss"],
            records[0]["critic_loss"],
            steps,
            flush=True,
        )
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def reload_and_evaluate(run):
    expected = json.loads((run / "checkpoint_expectation.json").read_text())
    trainer = DreamerTrainer.from_checkpoint(run / "latest.pt", cuda=True, log_dir=run / "reload")
    try:
        assert trainer.device.type == "cuda" and trainer.completed_updates == 1
        for k, v in expected["model_hashes"].items():
            assert state_hash(getattr(trainer, k)) == v
        assert optimizer_steps(trainer) == expected["optimizer_steps"]
        assert trainer.cadence_totals == expected["cadence_totals"]
        assert (
            len(trainer.replay.episodes) == expected["replay_episodes"]
            and trainer.replay.size == expected["replay_size"]
        )
        batch = trainer.replay.sample(16, seed=99)
        assert batch["obs"].shape == (16, 16, 52)
        assert isinstance(trainer.env, Cage2BlueEnv)
        trainer.cfg.train.deterministic_env = True
        trainer.world.eval()
        trainer.actor.eval()
        trainer.world.posterior.deterministic_latent_for_tests = True
        torch.manual_seed(10000)
        trainer.env_state = trainer.prepare_env_output(trainer.env.reset(seed=10000), reset=True)
        with torch.no_grad():
            trainer.world_state = trainer.world.observe_step(
                trainer.world.init_state(1),
                trainer.env_state["state"],
                torch.zeros(1, trainer.action_dim, device=trainer.device),
                is_first=torch.ones(1, dtype=torch.bool, device=trainer.device),
            )["post"].detach()
        trainer.collect_env_steps()
        journal = trainer.env.journal
        assert len(journal) == 30 and journal[-1]["wrapper_done"]
        assert optimizer_steps(trainer) == expected["optimizer_steps"]
        for k, v in expected["model_hashes"].items():
            assert state_hash(getattr(trainer, k)) == v
        report = dict(
            python_executable=sys.executable,
            virtual_environment=sys.prefix,
            restored_model_hashes_match=True,
            restored_optimizer_steps=optimizer_steps(trainer),
            restored_replay_episodes=expected["replay_episodes"],
            restored_replay_tokens=expected["replay_size"],
            replay_sample_available=True,
            seed=10000,
            action_mode="greedy",
            latent_mode="posterior_modes",
            steps=30,
            total_blue_return=sum(r["reward"] for r in journal),
            terminal=journal[-1]["native_done"],
            truncated=journal[-1]["truncated"],
            checkpoint=str(run / "latest.pt"),
            scope="Unassisted pipeline check, not training performance",
        )
        write(run / "evaluation.json", report)
        write(run / "evaluation_journal.json", journal)
        print("RELOAD_EVALUATE_COMPLETE", report, flush=True)
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "reload"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    assert sys.prefix != sys.base_prefix and torch.cuda.is_available(), "Use uv run --locked"
    torch.set_num_threads(2)
    if args.phase == "reload":
        return reload_and_evaluate(args.run_dir.resolve())
    run = args.run_dir.resolve()
    train(run, args.seed)
    torch.cuda.empty_cache()
    env = os.environ.copy()
    for key in [
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_SHLVL",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
    ]:
        env.pop(key, None)
    command = [
        "uv",
        "run",
        "--locked",
        "python",
        str(Path(__file__).resolve()),
        "--phase",
        "reload",
        "--run-dir",
        str(run),
    ]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    result = dict(
        training=json.loads((run / "train_result.json").read_text()),
        reward_trace=json.loads((run / "reward_trace.json").read_text()),
        reload_evaluation=json.loads((run / "evaluation.json").read_text()),
    )
    write(run / "results.json", result)
    print("CYCLE_COMPLETE", run, flush=True)


if __name__ == "__main__":
    main()
