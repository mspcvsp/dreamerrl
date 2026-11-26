"""Offline hierarchical goal-to-future-latent probes for trusted POPGym checkpoints."""

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from latent_probe import episode_split, metrics, paired_episode_interval, ridge, spectral_rank

from dreamerrl.env.env_factory import make_env
from dreamerrl.models.hierarchical_actor import HierarchicalActor
from dreamerrl.training.trainer import DreamerTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def collect(trainer, episodes, seed, *, deterministic=False):
    """Use a fresh single-row environment and restore Python, NumPy, and Torch RNGs."""
    world, actor = trainer.world, trainer.actor
    if not isinstance(actor, HierarchicalActor) or not trainer.cfg.env.env_id.lower().startswith("popgym-"):
        raise ValueError("This collector requires a hierarchical POPGym model")
    device = trainer.device
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    training_modes = [(module, module.training) for model in (world, actor) for module in model.modules()]
    posterior = world.posterior.deterministic_latent_for_tests
    saved_python, saved_numpy = random.getstate(), np.random.get_state()
    dataset = []
    env = None
    try:
        world.eval()
        actor.eval()
        world.posterior.deterministic_latent_for_tests = True
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            env = make_env(replace(trainer.cfg.env, num_envs=1, parallel=False, seed=seed), device)
            for episode in range(episodes):
                out = env.reset(seed=seed + episode)
                state, policy = world.init_state(1), actor.initial_state(1, device)
                action = torch.zeros(1, env.action_dim, device=device)
                observations, latents, goals, decisions = [], [], [], []
                total = 0.0
                for step in range(trainer.cfg.env.max_episode_steps + 1):
                    observation = out["state"] if step == 0 else out["transition_state"]
                    if trainer.cfg.env.finite_horizon:
                        clock = observation.new_full((1, 1), 1 - step / trainer.cfg.env.max_episode_steps)
                        observation = torch.cat((observation, clock), -1)
                    state = world.observe_step(
                        state, observation, action, is_first=torch.full((1,), step == 0, device=device),
                    )["post"]
                    observations.append(observation[0].cpu().numpy())
                    latents.append(torch.cat((state.h, state.z.flatten(1)), -1)[0].cpu().numpy())
                    if bool(out["is_last"][0]):
                        break
                    if step == trainer.cfg.env.max_episode_steps:
                        raise RuntimeError("POP Gym episode exceeded the configured horizon")
                    chosen = actor.step(state.h, state.z, policy, deterministic=deterministic)
                    policy = chosen.state
                    goals.append(int(policy.goal_id[0]))
                    decisions.append(bool(chosen.manager_decision[0]))
                    action = torch.nn.functional.one_hot(chosen.action, env.action_dim).float()
                    out = env.step(chosen.action)
                    total += float(out["reward"][0])
                dataset.append(dict(
                    episode=episode, observation=np.asarray(observations), latent=np.asarray(latents),
                    goals=np.asarray(goals), decisions=np.asarray(decisions), episode_return=total,
                ))
    finally:
        try:
            if env is not None:
                env.venv.close()
        finally:
            random.setstate(saved_python)
            np.random.set_state(saved_numpy)
            world.posterior.deterministic_latent_for_tests = posterior
            for module, training in training_modes:
                module.training = training
    return dataset


def future_pairs(dataset, lag):
    """Pair decisions with later posteriors inside the same episode, including its final observation."""
    if lag < 1:
        raise ValueError("Prediction lag must be positive")
    rows = []
    for row in dataset:
        for step in np.flatnonzero(row["decisions"]):
            if step + lag < len(row["latent"]):
                rows.append((
                    row["episode"], step, row["goals"][step], row["observation"][step],
                    row["latent"][step], row["latent"][step + lag],
                ))
    if not rows:
        raise ValueError("No eligible decision pairs for this lag")
    return tuple(np.asarray(column) for column in zip(*rows, strict=True))


def usage(dataset, embeddings):
    goal_ids = np.concatenate([row["goals"][row["decisions"]] for row in dataset])
    counts = np.bincount(goal_ids, minlength=len(embeddings))
    probability = counts / max(1, counts.sum())
    active = probability[probability > 0]
    entropy = float(-np.sum(active * np.log(active)))
    norms = np.linalg.norm(embeddings, axis=1)
    normalized = embeddings / np.maximum(norms[:, None], 1e-12)
    distances = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=-1)
    i, j = np.triu_indices(len(embeddings), 1)
    cosine = (normalized @ normalized.T)[i, j]
    redundant = (cosine >= 0.95) | (distances[i, j] <= 1e-6)
    return dict(
        decision_counts=counts.tolist(), entropy_nats=entropy,
        normalized_entropy=entropy / np.log(len(embeddings)) if len(embeddings) > 1 else 0.0,
        effective_goals=float(np.exp(entropy)) if len(goal_ids) else 0.0,
        possible_collapse=bool(len(goal_ids) and probability.max() >= 0.9),
        embedding_norms=norms.tolist(), codebook_rank=spectral_rank(embeddings),
        redundant_embedding_pairs=np.stack((i[redundant], j[redundant]), 1).tolist(),
        max_embedding_cosine=float(cosine.max()) if cosine.size else None,
    )


def error_interval(actual, predicted, episode, sampled_episodes):
    _, index = np.unique(episode, return_inverse=True)
    counts = np.bincount(index)[sampled_episodes].sum(1)
    squared = np.bincount(index, weights=np.square(actual - predicted))[sampled_episodes].sum(1)
    return np.quantile(np.sqrt(squared / counts), (0.025, 0.975)).tolist()


def analyze(dataset, embeddings, *, lag, seed, draws=1000) -> dict[str, Any]:
    episode, timestep, goals, observation, latent, future = future_pairs(dataset, lag)
    if len(np.unique(episode)) < 5:
        raise ValueError("Need eligible pairs from at least five distinct episodes")
    split, groups = episode_split(episode, seed)
    test = split["test"]
    episodes = len(groups["test"])
    sampled_episodes = np.random.default_rng(seed).integers(episodes, size=(draws, episodes))
    # Fixed axes fit only on training episodes. Predict future posterior changes along these axes.
    delta = np.asarray(future - latent, dtype=np.float64)
    centered = delta - delta[split["fit"]].mean(0)
    _, spectrum, axes = np.linalg.svd(centered[split["fit"]], full_matrices=False)
    axes = axes[: min(4, int((spectrum > 1e-8).sum()))]
    timestep = timestep[:, None]
    one_hot = np.eye(len(embeddings))[goals]
    controls = np.column_stack((latent, observation, timestep))
    features = dict(
        timestep=timestep, observation=observation, latent=latent, controls=controls,
        goal_only=one_hot, controls_and_goal=np.column_stack((controls, one_hot)),
    )
    targets = centered @ axes.T
    results = []
    for index, target in enumerate(targets.T):
        predictions = {
            name: ridge(values, target, split, (0.001, 0.01, 0.1, 1.0, 10.0))
            for name, values in features.items()
        }
        model_scores = {
            name: dict(
                alpha=model["alpha"], **metrics(target[test], model["predictions"][test]),
                rmse_ci95=error_interval(target[test], model["predictions"][test], episode[test], sampled_episodes),
            )
            for name, model in predictions.items()
        }
        first = predictions["controls_and_goal"]["predictions"][test]
        reference = predictions["controls"]["predictions"][test]
        results.append(dict(
            component=index, scores=model_scores,
            added_goal_interval=paired_episode_interval(target[test], first, reference, episode[test], seed, draws),
        ))
    returns = [row["episode_return"] for row in dataset]
    return dict(
        method="Frozen observational rollout; linear prediction of posterior change, not a controllability test.",
        lag=lag, pairs=len(episode), split_episodes={name: group.tolist() for name, group in groups.items()},
        split_pairs={name: int(mask.sum()) for name, mask in split.items()},
        components=results, target_status="variable" if results else "constant_fit_target",
        goal_usage=usage(dataset, embeddings),
        evaluation_returns=dict(mean=float(np.mean(returns)), episode_returns=returns),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted project checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory under runs/")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--lag", type=int, help="Defaults to the configured manager interval")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT_ROOT / "runs") or args.episodes < 5:
        parser.error("Use a fresh directory under runs/ and at least five episodes")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("version") != 5 or not payload["config"].env.env_id.lower().startswith("popgym-"):
        parser.error("A trusted hierarchical POPGym checkpoint is required")
    del payload
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    trainer = DreamerTrainer.from_checkpoint(args.checkpoint, cuda=False, log_dir=output)
    try:
        assert isinstance(trainer.actor, HierarchicalActor)
        dataset = collect(trainer, args.episodes, args.seed, deterministic=args.deterministic)
        report = analyze(
            dataset, trainer.actor.goal_embedding.weight.detach().cpu().numpy(),
            lag=trainer.actor.interval if args.lag is None else args.lag, seed=args.seed,
        )
        with args.checkpoint.open("rb") as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        report.update(
            checkpoint=str(args.checkpoint.resolve()), sha256=checksum.hexdigest(),
            training_seed=trainer.cfg.train.seed, environment=trainer.cfg.env.env_id,
            rollout_seed=args.seed, deterministic=args.deterministic,
        )
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


if __name__ == "__main__":
    main()
