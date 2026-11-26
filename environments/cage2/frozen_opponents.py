"""Evaluate immutable CAGE2 Blue checkpoints with the existing greedy rollout."""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from host_experiment_support import ROOT, git, sha256
from learning_curve import evaluate_policy, write

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
from dreamerrl.models.actor import Actor
from dreamerrl.models.hierarchical_actor import HierarchicalActor
from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import LatentConfig, NetworkConfig

MANIFEST = Path(__file__).with_suffix(".json")
VENDOR = ROOT / "environments/cage2/vendor/cage-challenge-2"
OPPONENTS = {"B-line": B_lineAgent, "Meander": RedMeanderAgent, "Sleep": SleepAgent}


def env_schema(env):
    return env.obs_dim, env.action_dim, env.host_order, env.subnet_ids, env.action_mapping


def load_policy(item, spec, device):
    payload = torch.load(ROOT / item["path"], map_location="cpu", weights_only=False)
    cfg = payload["config"]
    if (
        payload["version"] != item["version"]
        or payload["completed_updates"] != spec["completed_updates"]
        or payload["total_env_steps"] != spec["env_steps"]
        or cfg.train.seed != item["training_seed"]
        or cfg.env.seed != item["training_seed"]
        or cfg.ac.hierarchical != (item["version"] == 5)
        or cfg.world.obs_encoder != "cage2_host"
        or cfg.env.finite_horizon
        or cfg.env.max_episode_steps != spec["horizon"]
    ):
        raise ValueError(f"Saved configuration or training budget mismatch for {item['id']}")

    env = Cage2BlueEnv(cfg.env, device)
    try:
        if env_schema(env)[:4] != (52, 145, tuple(cfg.world.cage2_host_order), tuple(cfg.world.cage2_subnet_ids)):
            raise ValueError(f"Blue input or action schema mismatch for {item['id']}")
        latent = LatentConfig(
            deter_size=cfg.world.deter_size, stoch_size=cfg.world.stoch_size, num_classes=cfg.world.num_classes
        )
        world_net = NetworkConfig(
            hidden_size=cfg.world.hidden_size, action_dim=env.action_dim,
            value_bins=cfg.world.value_bins, unimix=cfg.world.unimix,
        )
        actor_net = NetworkConfig(hidden_size=cfg.ac.actor_hidden, action_dim=env.action_dim, unimix=cfg.ac.unimix)
        world = WorldModel(obs_space=env.obs_space, dreamer_cfg=cfg, latent=latent, net=world_net, device=device)
        actor = (
            HierarchicalActor(
                latent=latent, net=actor_net, goals=cfg.ac.hierarchy_goals,
                goal_dim=cfg.ac.hierarchy_goal_dim, interval=cfg.ac.hierarchy_interval,
            )
            if cfg.ac.hierarchical else Actor(latent=latent, net=actor_net)
        ).to(device)
        for name, module in (("world", world), ("actor", actor)):
            module.load_state_dict(payload[name], strict=True)
            module.requires_grad_(False).eval()
        return SimpleNamespace(
            cfg=cfg, world=world, actor=actor, device=device, action_dim=env.action_dim, schema=env_schema(env)
        ), env.scenario
    finally:
        env.close()


def opponent_env_factory(policy, opponent):
    def make_env(cfg, device):
        env = Cage2BlueEnv(cfg, device, red_agent=OPPONENTS[opponent])
        if env_schema(env) != policy.schema:
            raise ValueError(f"Blue schema differs for {opponent} at seed {cfg.seed}")
        return env

    return make_env


def run_cell(policy, item, opponent, seeds, output):
    name = f"{item['id']}_{opponent.lower()}"
    log = output / f"{name}.jsonl"
    with log.open("x", buffering=1) as stream:
        report = evaluate_policy(
            policy, "actor", seeds, env_factory=opponent_env_factory(policy, opponent),
            on_episode=lambda row: stream.write(json.dumps(row, allow_nan=False) + "\n"),
        )
    returns = [row["total_return"] for row in report["episodes"]]
    return dict(
        checkpoint=item["id"], training_seed=item["training_seed"], opponent=opponent,
        mean_return=statistics.mean(returns), episode_sd=statistics.stdev(returns), episodes=len(returns),
        seconds=report["seconds"], log=log.name,
    )


def report_results(output, rows, spec):
    text = [
        "# Frozen CAGE2 Blue opponent evaluation", "",
        f"Greedy actions and posterior modes, native undiscounted Blue returns, {spec['horizon']} steps per episode.",
        f"All cells share {spec['episodes_per_cell']} simulator seeds starting at {spec['first_seed']}.",
        "Higher returns are better. B-line was the training opponent.",
    ]
    by_opponent = {}
    for opponent in spec["opponents"]:
        group = [row for row in rows if row["opponent"] == opponent]
        hierarchy = [row["mean_return"] for row in group if row["checkpoint"].startswith("hierarchical_")]
        mean = statistics.mean(hierarchy)
        sd = statistics.stdev(hierarchy)
        by_opponent[opponent] = dict(mean_return=mean, training_seed_sd=sd, training_seeds=len(hierarchy))
        text += [
            "", f"## {opponent}", "",
            "| Checkpoint | Training seed | Mean return | Episode SD | Episodes |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in group:
            text.append(
                f"| {row['checkpoint']} | {row['training_seed']} | {row['mean_return']:.3f} | "
                f"{row['episode_sd']:.3f} | {row['episodes']} |"
            )
        text += ["", f"Hierarchical training-seed mean {mean:.3f}, sample SD {sd:.3f} across {len(hierarchy)} checkpoint means."]
    text += [
        "", "## Interpreting variability", "",
        "Episode SD measures variation across simulator seeds for one fixed checkpoint. The same simulator "
        "seeds are paired across cells. Hierarchical training-seed SD is the descriptive sample SD of the "
        "three checkpoint means on that shared seed set. The flat policy has one training seed, so its "
        "training-seed SD cannot be estimated.",
        "", "## Official protocol differences", "",
        "The vendored submission evaluator runs 100 episodes for each of the three opponents at 30, 50 "
        "and 100 steps and reports mean and sample SD per combination. This study uses the original "
        "training horizon of 30 steps and a freshly seeded simulator for each episode. The submission "
        "script reuses one simulator per cell, resetting it between episodes without an explicit "
        "paired seed schedule. Official leaderboard validation sums the per-combination means over "
        "1,000 episodes per combination. These checkpoint results are a local generalization study.",
        "", "## Reproducibility", "",
        "See provenance.json for checkpoint and dependency fingerprints, saved configurations and seed "
        "schedule. Each checkpoint and opponent has its own episode JSONL log.", "",
    ]
    write(output / "report.json", dict(cells=rows, hierarchical_by_opponent=by_opponent))
    (output / "report.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ready-channel")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT / "runs":
        parser.error("Output must be a new directory directly under runs/")
    output.mkdir()
    status = 1
    try:
        with (output / "console.log").open("x", buffering=1) as console:
            with redirect_stdout(console), redirect_stderr(console):
                try:
                    run_evaluation(output, args.ready_channel)
                except KeyboardInterrupt:
                    status = 130
                    traceback.print_exc()
                except Exception:
                    traceback.print_exc()
                else:
                    status = 0
    finally:
        (output / "exit_status").write_text(f"{status}\n")
        if args.ready_channel and not (output / "started.json").is_file():
            subprocess.run(["tmux", "wait-for", "-S", args.ready_channel], check=False)
    return status


def run_evaluation(output, ready_channel):
    spec = json.loads(MANIFEST.read_text())
    seeds = list(range(spec["first_seed"], spec["first_seed"] + spec["episodes_per_cell"]))
    if spec["horizon"] != 30:
        raise ValueError("The historical evaluator implements 30 steps")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    if git("diff", "--name-only", "HEAD", "--", "uv.lock", "pyproject.toml"):
        raise ValueError("Dependency declarations have local modifications")
    import CybORG

    if not Path(CybORG.__file__).resolve().is_relative_to(VENDOR):
        raise ValueError("Installed CybORG is outside the locked vendor checkout")
    vendor_revision = git("rev-parse", "HEAD", root=VENDOR)
    if vendor_revision != git("rev-parse", "HEAD:environments/cage2/vendor/cage-challenge-2"):
        raise ValueError("Vendor revision differs from the recorded gitlink")
    if git("status", "--porcelain", "--untracked-files=no", root=VENDOR):
        raise ValueError("CybORG vendor source has local modifications")
    checkpoints = spec["checkpoints"]
    for item in checkpoints:
        git("merge-base", "--is-ancestor", item["source"], "HEAD")
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Checkpoint fingerprint mismatch for {item['id']}")
    policies = {}
    configs = {}
    for item in checkpoints:
        policy, scenario = load_policy(item, spec, device)
        policies[item["id"]] = policy
        configs[item["id"]] = policy.cfg.to_dict()
        print(f"Verified {item['id']} checksum and strict world/actor reload", flush=True)
    baseline_tag = git("rev-parse", "cage2_baseline_v1")
    write(output / "provenance.json", dict(
        manifest=spec, manifest_sha256=sha256(MANIFEST), source_revision=git("rev-parse", "HEAD"),
        baseline_tag=baseline_tag, python=sys.version, platform=platform.platform(), device=str(device),
        torch_threads=torch.get_num_threads(), pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        uv=subprocess.check_output(["uv", "--version"], text=True).strip(),
        packages=sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
        lock_sha256=sha256(ROOT / "uv.lock"), cyborg_revision=vendor_revision, scenario_sha256=sha256(scenario),
        configs=configs, seeds=seeds, policy_seeds=[seed + 100_000 for seed in seeds],
        seed_audit="No matches for the new block in maintained configs, experiment worktrees and local JSON/CSV/JSONL history before launch.",
        command=sys.argv, policy="greedy, deterministic posterior modes",
    ))
    (output / "source.diff").write_text(git("diff", "--no-ext-diff", "--binary", "HEAD") + "\n")

    # Validate a full episode for each cell on a seed outside the reporting schedule.
    smoke_seed = spec["first_seed"] - 1
    for item in checkpoints:
        policy = policies[item["id"]]
        for opponent in spec["opponents"]:
            smoke = evaluate_policy(
                policy, "actor", [smoke_seed],
                env_factory=opponent_env_factory(policy, opponent),
            )
            if not math.isfinite(smoke["mean_return"]):
                raise ValueError(f"Non-finite smoke result for {item['id']} / {opponent}")
    write(output / "started.json", dict(cells=len(checkpoints) * len(OPPONENTS), smoke_seed=smoke_seed))
    if ready_channel:
        subprocess.run(["tmux", "wait-for", "-S", ready_channel], check=True)

    rows = []
    for item in checkpoints:
        for opponent in spec["opponents"]:
            row = run_cell(policies[item["id"]], item, opponent, seeds, output)
            rows.append(row)
            print(json.dumps(row), flush=True)
    for item in checkpoints:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Checkpoint fingerprint changed for {item['id']}")
    if baseline_tag != git("rev-parse", "cage2_baseline_v1"):
        raise ValueError("Baseline tag changed during evaluation")
    report_results(output, rows, spec)


if __name__ == "__main__":
    sys.exit(main())
