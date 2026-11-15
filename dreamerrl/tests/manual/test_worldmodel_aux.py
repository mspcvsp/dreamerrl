"""Optional learning experiment: compare MiniHack baseline and novelty by seed.

Passing means the experiment completed with valid metrics. Reward improvement
and auxiliary benefit are reported outcomes, not uncalibrated test thresholds.
"""

import json
from statistics import mean
from typing import Any

import pytest
import torch

from dreamerrl.tests.manual.utils import evaluate_minihack, minihack_config
from dreamerrl.training.trainer import DreamerTrainer


def run_experiment(seed, novelty, updates, episodes, logdir, initial_policy=None):
    cfg = minihack_config(seed, logdir, novelty)
    cfg.env.max_episode_steps = cfg.train.collect_steps = 100
    cfg.train.seq_len = 20
    cfg.train.batch_size = 4
    cfg.train.random_exploration_steps = 200
    cfg.world.deter_size = 32
    cfg.world.num_classes = cfg.world.stoch_size = 8
    cfg.world.hidden_size = cfg.ac.actor_hidden = cfg.ac.critic_hidden = 64
    trainer = DreamerTrainer(cfg)
    # Same held-out levels for both arms, separate from training seeds.
    evaluation_seeds = list(range(10000 + seed * 1000, 10000 + seed * 1000 + episodes))
    try:
        if initial_policy is not None:
            trainer.actor.load_state_dict(initial_policy["actor"])
            trainer.critic.load_state_dict(initial_policy["critic"])
        initial_policy = {
            "actor": {key: value.detach().clone() for key, value in trainer.actor.state_dict().items()},
            "critic": {key: value.detach().clone() for key, value in trainer.critic.state_dict().items()},
        }
        assert [objective.name for objective in trainer.world.aux_objectives] == (["novelty"] if novelty else [])
        random_returns = evaluate_minihack(cfg, trainer.world, trainer.actor, evaluation_seeds, random_policy=True)
        untrained_returns = evaluate_minihack(cfg, trainer.world, trainer.actor, evaluation_seeds)
        losses = []
        for update in range(updates):
            trainer.collect_env_steps()
            batch = trainer.replay.sample(cfg.train.batch_size)
            metrics = trainer.update_world_model(batch, update)
            assert len(metrics.aux_losses) == int(novelty)
            actor_loss, critic_loss = trainer.update_actor_critic(batch, update)
            scalars = [metrics.total_loss.item(), actor_loss, critic_loss]
            auxiliary = [loss.item() for loss in metrics.aux_losses]
            assert torch.isfinite(torch.tensor(scalars + auxiliary)).all()
            row = dict(world=scalars[0], actor=actor_loss, critic=critic_loss, auxiliary=auxiliary)
            losses.append(row)
            if update % 25 == 0:
                print(f"[minihack] seed={seed} novelty={novelty} update={update}/{updates}", flush=True)
        trained_returns = evaluate_minihack(cfg, trainer.world, trainer.actor, evaluation_seeds)
        assert torch.isfinite(torch.tensor(random_returns + untrained_returns + trained_returns)).all()
        result: dict[str, Any] = dict(
            seed=seed,
            auxiliary="novelty" if novelty else "none",
            updates=updates,
            env_steps=trainer.total_env_steps,
            evaluation_seeds=evaluation_seeds,
            random_returns=random_returns,
            untrained_returns=untrained_returns,
            trained_returns=trained_returns,
            losses=losses,
        )
        return result, initial_policy
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


@pytest.mark.manual
@pytest.mark.worldmodel_aux
def test_minihack_baseline_vs_novelty(request, tmp_path):
    updates = request.config.getoption("--steps")
    episodes = request.config.getoption("--eval-episodes")
    if updates < 1 or episodes < 1:
        raise pytest.UsageError("--steps and --eval-episodes must be positive")
    results = []
    for seed in [0, 1, 2]:
        baseline, initial_policy = run_experiment(seed, False, updates, episodes, tmp_path / "logs")
        novelty, _ = run_experiment(seed, True, updates, episodes, tmp_path / "logs", initial_policy)
        assert baseline["env_steps"] == novelty["env_steps"]
        assert baseline["random_returns"] == novelty["random_returns"]
        assert baseline["untrained_returns"] == novelty["untrained_returns"]
        for result in [baseline, novelty]:
            print(
                f"seed={seed} aux={result['auxiliary']} env_steps={result['env_steps']} "
                f"random={mean(result['random_returns']):.3f} "
                f"untrained={mean(result['untrained_returns']):.3f} "
                f"trained={mean(result['trained_returns']):.3f}",
                flush=True,
            )
            results.append(result)
    deltas = [
        mean(results[i + 1]["trained_returns"]) - mean(results[i]["trained_returns"]) for i in range(0, len(results), 2)
    ]
    report = dict(
        environment="MiniHack-Room-5x5-v0",
        max_episode_steps=100,
        evaluation_policy="greedy actions and posterior modes on held-out episode seeds",
        paired_novelty_minus_baseline=deltas,
        mean_paired_difference=mean(deltas),
        runs=results,
    )
    path = tmp_path / "minihack_ablation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Per-seed novelty minus baseline: {deltas}; mean={mean(deltas):.3f}")
    print(f"Experiment report: {path}")
    print("Passing checks experiment integrity; inspect returns to assess learning and auxiliary benefit.")
