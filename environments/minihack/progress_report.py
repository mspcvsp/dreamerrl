"""Frozen, level-paired diagnostics for the signed-progress comparison."""

import itertools
import math
from pathlib import Path
from statistics import mean

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit

from director_goal_audit import branch, digest, prepared_start
from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.eval.minihack_eval import evaluate_minihack
from dreamerrl.training.run_monitor import write_json


def effective_goals(counts):
    probability = counts / counts.sum()
    live = probability[probability > 0]
    return math.exp(float(-(live * np.log(live)).sum()))


def metrics(levels, indexes):
    codes = np.stack([levels[i]["counts"] for i in indexes]).sum(0)
    return dict(
        task=mean(levels[i]["task"] for i in indexes),
        target_progress=mean(levels[i]["target_progress"] for i in indexes),
        specificity=mean(levels[i]["specificity"] for i in indexes),
        endpoints=mean(levels[i]["endpoints"] for i in indexes),
        tv=float(np.median(np.concatenate([levels[i]["tv"] for i in indexes]))),
        effective_goals=effective_goals(codes),
    )


def uncertainty(control, treatment, seed):
    """Resample matched levels. Preserve code pairs and intervals inside a level."""
    n = len(control)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in metrics(control, range(n))}
    for indexes in rng.integers(n, size=(1000, n)):
        a, b = metrics(control, indexes), metrics(treatment, indexes)
        for key in samples:
            samples[key].append(b[key] - a[key])
    a, b = metrics(control, range(n)), metrics(treatment, range(n))
    return dict(absolute=a, progress=b, delta={key: b[key] - a[key] for key in a},
                paired_level_bootstrap_95={key: np.quantile(v, (.025, .975)).tolist() for key, v in samples.items()})


@torch.inference_mode()
def evaluate(checkpoint, seeds, *, progress, budget):
    agent = load_frozen_agent(checkpoint, allow_hierarchical=True)
    if agent.cfg.env.env_id != "MiniHack-Room-5x5-v0" or agent.update != budget or (
        agent.cfg.ac.director_goal_progress != progress
    ):
        raise ValueError(f"Unexpected frozen checkpoint configuration for {checkpoint}")
    actor = agent.actor
    goals = actor.decode_goals(torch.arange(actor.num_goals))
    pairs = list(itertools.combinations(range(len(goals)), 2))
    i, j = np.asarray(pairs).T
    results = {}
    for mode, sampled in (("greedy", False), ("sampled", True)):
        goal_rows = []
        returns = evaluate_minihack(
            agent.cfg, agent.world, actor, seeds, sample_actions=sampled, sample_latents=sampled,
            sampling_seed=19, goal_diagnostics=goal_rows,
        )
        levels = {}
        with TimeLimit(gym.make(agent.cfg.env.env_id), max_episode_steps=agent.cfg.env.max_episode_steps) as env:
            for index, (seed, task) in enumerate(zip(seeds, returns)):
                rng = np.random.default_rng(seed)
                prefix = rng.integers(agent.world.net_cfg.action_dim, size=index % actor.interval).tolist()
                if prepared_start(agent, env, seed, prefix) is None:
                    # Preserve the level for natural evaluation and use the initial
                    # state for intervention if the preselected prefix ended it.
                    prefix = []
                uniforms = rng.random(actor.interval) if sampled else [None] * actor.interval
                rows = [branch(agent, env, seed, prefix, goals, code, uniforms) for code in range(len(goals))]
                p = np.asarray(rows[0]["probabilities"][0])
                delta = np.asarray([np.asarray(row["real"][-1]) - row["real"][0] for row in rows])
                endpoint = [row["path"][-1] for row in rows]
                natural = [row for row in goal_rows if row["seed"] == seed]
                matched = np.diag(delta)
                counts = np.bincount([row["goal_id"] for row in natural], minlength=len(goals))
                levels[seed] = dict(
                    task=task, counts=counts.tolist(),
                    endpoints=mean(endpoint[a] != endpoint[b] for a, b in pairs),
                    tv=(np.abs(p[i] - p[j]).sum(1) * .5).tolist(),
                    target_progress=float(matched.mean()),
                    specificity=float((matched - (delta.sum(0) - matched) / (len(goals) - 1)).mean()),
                    goal_completion_rate=mean(row["achieved"] for row in natural),
                    trajectories=rows,
                )
        # Maintain an explicit per-level ordering for paired bootstrap.
        results[mode] = [levels[seed] for seed in seeds]
    return results


def initial_tensors(run):
    payload = torch.load(run / "initial.pt", map_location="cpu", weights_only=False)
    return [value for key in ("world", "actor", "critic", "slow_critic") for value in payload[key].values()]


def report(output, jobs, spec):
    seeds = list(range(spec["diagnostic_seed_start"], spec["diagnostic_seed_start"] + spec["diagnostic_levels"]))
    result = {"levels": seeds, "pairs": {}, "incomplete": []}
    for seed in spec["arms"]["absolute"]:
        pair = {r["arm"]: r for r in jobs if r["seed"] == seed}
        if len(pair) != 2:
            continue
        full = all(j["state"] == "completed" and j["updates"] == spec["updates"] and (
            j["env_steps"] == spec["updates"] * spec["num_envs"] * spec["collect_steps"]
        ) for j in pair.values())
        paths = {key: Path(j["run"]) for key, j in pair.items()}
        if not full:
            result["incomplete"].extend(dict(arm=arm, seed=seed, state=j["state"], updates=j.get("updates", 0))
                                        for arm, j in pair.items() if j["state"] != "completed")
        budget = spec["updates"] if full else spec["fallback_update"]
        filename = "latest.pt" if full else f"matched_{budget}.pt"
        checkpoints = {key: run / filename for key, run in paths.items()}
        if not all(p.is_file() for p in checkpoints.values()):
            continue
        # A fallback is descriptive and never substitutes for primary acceptance.
        a, b = (initial_tensors(paths[k]) for k in ("absolute", "progress"))
        if len(a) != len(b) or any(not torch.equal(x, y) for x, y in zip(a, b)):
            raise ValueError(f"Initialization mismatch for training seed {seed}")
        frozen = {key: evaluate(path, seeds, progress=key == "progress", budget=budget) for key, path in checkpoints.items()}
        write_json(output / f"frozen_seed{seed}.json", frozen)
        modes = {mode: uncertainty(frozen["absolute"][mode], frozen["progress"][mode], seed)
                 for mode in ("greedy", "sampled")}
        for summary in modes.values():
            point = summary["progress"]
            summary["passes_thresholds"] = (
                point["task"] >= .9 and point["target_progress"] > 0 and point["specificity"] > 0 and point["endpoints"] >= .1
                and point["tv"] >= .05 and point["effective_goals"] >= 2
            ) if full else None
        result["pairs"][str(seed)] = dict(
            budget=budget, primary=full, modes=modes,
            checkpoint_sha256={arm: digest(path) for arm, path in checkpoints.items()},
        )
    write_json(output / "progress_comparison.json", result)
    text = ["## Frozen goal interventions", "",
            "Greedy uses posterior modes and argmax actions. Sampled natural evaluation samples both. Sampled interventions share worker-action uniforms and hold the posterior mode fixed. Goal specificity is the increase toward the assigned code minus the increase toward that same code under alternative assignments.",
            "", "| Seed | Update | Budget | Policy | Reward | Task return | Target progress | Goal specificity | Distinct endpoints | Median action TV | Effective goals |",
            "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for seed, row in result["pairs"].items():
        for mode, comparison in row["modes"].items():
            for arm in ("absolute", "progress"):
                m = comparison[arm]
                text.append(f"| {seed} | {row['budget']} | {'Primary' if row['primary'] else 'Fallback'} | {mode} | {arm} | {m['task']:.3f} | {m['target_progress']:+.4f} | {m['specificity']:+.4f} | {m['endpoints']:.1%} | {m['tv']:.3f} | {m['effective_goals']:.2f} |")
    if result["incomplete"]:
        text += ["", "### Incomplete workers", ""]
        for row in result["incomplete"]:
            text.append(f"- {row['arm']}, seed {row['seed']}, {row['state']} at update {row['updates']}.")
    text += [
        "", "Per-seed paired 95% level-bootstrap intervals and threshold outcomes are in progress_comparison.json. Training-seed uncertainty is shown as separate paired estimates. Two seeds provide limited evidence for generalization across training runs.",
        "", "Incomplete workers are reported separately. Fallback checkpoints are assessed only at their shared predeclared update. They do not enter primary acceptance.",
        "", "Corridor R2 remains deferred.", "",
    ]
    return "\n".join(text)
