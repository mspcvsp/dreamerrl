"""Inspect policy entropy and repetitive behavior on seeded MiniHack episodes."""

import argparse
import math
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from minihack import MiniHack

from dreamerrl.eval.minihack_eval import evaluate_minihack
from dreamerrl.training.run_monitor import write_json
from dreamerrl.training.trainer import DreamerTrainer


def summarize_steps(steps, action_names):
    actions = [step["action"] for step in steps]
    longest = run = 0
    for i, action in enumerate(actions):
        run = run + 1 if i and action == actions[i - 1] else 1
        longest = max(longest, run)
    pairs = max(1, len(actions) - 1)
    return dict(
        steps=len(steps),
        entropy_nats=mean(s["entropy"] for s in steps),
        normalized_entropy=mean(s["entropy"] for s in steps) / math.log(len(action_names)),
        top_action_probability=mean(s["top_probability"] for s in steps),
        top_two_probability_gap=mean(s["probability_gap"] for s in steps),
        repeat_fraction=sum(a == b for a, b in zip(actions, actions[1:])) / pairs,
        longest_action_run=longest,
        unchanged_glyph_fraction=mean(s["unchanged_glyph"] for s in steps),
        stationary_fraction=mean(s["stationary"] for s in steps),
        action_counts={name: actions.count(i) for i, name in enumerate(action_names)},
    )


def diagnose(checkpoint, seeds, sampling_seed):
    with TemporaryDirectory(prefix="dreamerrl-policy-diagnostic-") as log_dir:
        trainer = DreamerTrainer.from_checkpoint(checkpoint, log_dir=log_dir)
        try:
            make_env = gym.make
            original_act = trainer.actor.act
            episodes = []
            action_names = []

            def record_act(state, deterministic=False):
                action, logits = original_act(state, deterministic=deterministic)
                log_probs = logits.log_softmax(-1)
                probs = log_probs.exp()
                top = probs.topk(2, dim=-1).values
                episodes[-1]["steps"].append(
                    dict(
                        action=int(action.item()),
                        entropy=float(-(probs * log_probs).sum(-1).item()),
                        top_probability=float(top[0, 0].item()),
                        probability_gap=float((top[0, 0] - top[0, 1]).item()),
                    )
                )
                return action, logits

            class RecordEnvironment(gym.Wrapper):
                def reset(self, **kwargs):
                    obs, info = super().reset(**kwargs)
                    self.previous_obs = {key: obs[key].copy() for key in ("glyphs", "blstats")}
                    episodes.append(dict(seed=kwargs["seed"], steps=[]))
                    return obs, info

                def step(self, action):
                    obs, reward, terminated, truncated, info = super().step(action)
                    step = episodes[-1]["steps"][-1]
                    step["unchanged_glyph"] = bool(np.array_equal(obs["glyphs"], self.previous_obs["glyphs"]))
                    step["stationary"] = bool(np.array_equal(obs["blstats"][:2], self.previous_obs["blstats"][:2]))
                    self.previous_obs = {key: obs[key].copy() for key in ("glyphs", "blstats")}
                    episodes[-1]["terminated"] = bool(terminated)
                    episodes[-1]["truncated"] = bool(truncated)
                    return obs, reward, terminated, truncated, info

            def record_env(*args, **kwargs):
                env = make_env(*args, **kwargs)
                minihack_env = cast(MiniHack, env.unwrapped)
                action_names[:] = [str(a.name) for a in minihack_env.actions]
                return RecordEnvironment(env)

            modes = []
            with (
                patch.object(trainer.actor, "act", record_act),
                patch("dreamerrl.eval.minihack_eval.gym.make", record_env),
            ):
                for sampled in (False, True):
                    episodes.clear()
                    returns = evaluate_minihack(
                        trainer.cfg,
                        trainer.world,
                        trainer.actor,
                        seeds,
                        sample_actions=sampled,
                        sample_latents=False,
                        sampling_seed=sampling_seed,
                    )
                    summaries = []
                    for episode, total in zip(episodes, returns):
                        summaries.append(
                            dict(
                                seed=episode["seed"],
                                episode_return=total,
                                terminated=episode["terminated"],
                                truncated=episode["truncated"],
                                **summarize_steps(episode["steps"], action_names),
                            )
                        )
                    aggregate = summarize_steps([s for e in episodes for s in e["steps"]], action_names)
                    # Sequence metrics must never join the end of one episode to the next.
                    aggregate["repeat_fraction"] = sum(
                        e["repeat_fraction"] * max(0, e["steps"] - 1) for e in summaries
                    ) / max(1, sum(max(0, e["steps"] - 1) for e in summaries))
                    aggregate["longest_action_run"] = max(e["longest_action_run"] for e in summaries)
                    result = dict(
                        actions="sampled" if sampled else "greedy",
                        latents="mode",
                        mean_return=mean(returns),
                        aggregate=aggregate,
                        episodes=summaries,
                    )
                    modes.append(result)
                    print(
                        f"update={trainer.completed_updates} {result['actions']}: {result['mean_return']:.4f} "
                        f"entropy={aggregate['normalized_entropy']:.3f} "
                        f"repeat={aggregate['repeat_fraction']:.3f}",
                        flush=True,
                    )
            return dict(checkpoint=str(checkpoint.resolve()), update=trainer.completed_updates, modes=modes)
        finally:
            trainer.env.venv.close()
            trainer.tb.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=3000000)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or min(args.seed_start, args.sampling_seed) < 0:
        parser.error("Episodes must be positive and seeds nonnegative")
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    report: dict[str, Any] = dict(evaluation_seeds=seeds, sampling_seed=args.sampling_seed, checkpoints=[])
    for checkpoint in args.checkpoint:
        report["checkpoints"].append(diagnose(checkpoint, seeds, args.sampling_seed))
        write_json(args.output, report)


if __name__ == "__main__":
    main()
