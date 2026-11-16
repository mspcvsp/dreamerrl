"""Measure position decodability from frozen world-model representations."""

import argparse
from pathlib import Path

import torch

from dreamerrl.eval.location_probe import collect_locations, evaluate_probes, load_frozen_agent, validate_seed_split
from dreamerrl.training.run_monitor import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=16)
    parser.add_argument("--train-seed-start", type=int, default=4000000)
    parser.add_argument("--test-seed-start", type=int, default=5000000)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--max-per-cell", type=int, default=4)
    parser.add_argument("--policy", choices=["random", "sampled", "greedy"], default="random")
    parser.add_argument("--sample-latents", action="store_true", help="Sample posterior latents instead of using modes")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dataset-output", type=Path, help="Optionally retain CPU features and labels for re-fitting")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.train_episodes, args.test_episodes, args.max_steps, args.max_per_cell, args.epochs, args.threads) < 1:
        parser.error("Episode counts, collection limits, epochs, and threads must be positive")
    if min(args.train_seed_start, args.test_seed_start, args.seed) < 0:
        parser.error("Seeds must be nonnegative")
    train_seeds = list(range(args.train_seed_start, args.train_seed_start + args.train_episodes))
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_episodes))
    validate_seed_split(train_seeds, test_seeds)
    torch.set_num_threads(args.threads)
    agent = load_frozen_agent(args.checkpoint, sample_latents=args.sample_latents)
    initial = (
        load_frozen_agent(args.initial_checkpoint, sample_latents=args.sample_latents)
        if args.initial_checkpoint
        else None
    )
    if initial is not None and initial.update != 0:
        parser.error("--initial-checkpoint must be from update zero")
    splits = []
    for name, seeds in (("train", train_seeds), ("test", test_seeds)):
        data = collect_locations(
            agent,
            seeds,
            max_steps=args.max_steps,
            max_per_cell=args.max_per_cell,
            policy=args.policy,
            sampling_seed=args.seed,
            initial=initial,
        )
        splits.append(data)
        print(f"{name}: {len(data.positions)} samples, {len(seeds)} level seeds", flush=True)
    train, test = splits
    if args.dataset_output:
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"train": train, "test": test, "update": agent.update}, args.dataset_output)
    results = evaluate_probes(train, test, epochs=args.epochs, seed=args.seed)
    report = dict(
        checkpoint=agent.checkpoint,
        update=agent.update,
        world_model_frozen=all(not parameter.requires_grad for parameter in agent.world.parameters()),
        position_label_source="blstats[0:2], used only as probe targets",
        normalization="training mean/std; constant dimensions use scale 1",
        latent_mode="posterior samples" if args.sample_latents else "posterior mode",
        initial_checkpoint=initial.checkpoint if initial else None,
        settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        train=train.summary(),
        test=test.summary(),
        grid_shape=train.grid_shape,
        results=results,
        limitations="Position decodability, not room identity or proof of memory use. Encoder and initial-model "
        "controls separate observation availability and random features from learned recurrence.",
    )
    write_json(args.output, report)
    for name, scores in results.items():
        score = scores.get("test", scores)
        print(
            f"{name}: held-out Manhattan={score['mean_manhattan_cells']:.3f} cells, "
            f"exact={score['exact_cell_accuracy']:.1%}",
            flush=True,
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
