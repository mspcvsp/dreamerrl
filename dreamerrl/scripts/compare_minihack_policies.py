"""Matched action/latent sampling evaluations with movement diagnostics."""

import argparse
from pathlib import Path
from statistics import mean
from typing import TypedDict

import torch

from dreamerrl.eval.corridor_diagnostics import CORRIDOR_R2, CorridorEpisode, summarize_stair_opportunities
from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.eval.minihack_eval import EpisodeDiagnostics, compare_returns, evaluate_minihack
from dreamerrl.training.run_monitor import write_json


class _OptionalPolicyModeReport(TypedDict, total=False):
    stair_episodes: list[CorridorEpisode]


class PolicyModeReport(_OptionalPolicyModeReport):
    summary: dict[str, object]
    episodes: list[EpisodeDiagnostics]
    returns: list[float]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--stair-diagnostics", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.threads) < 1 or min(args.seed_start, args.sampling_seed) < 0:
        parser.error("Counts must be positive and seeds nonnegative")
    torch.set_num_threads(args.threads)
    agent = load_frozen_agent(args.checkpoint)
    if args.stair_diagnostics and agent.cfg.env.env_id != CORRIDOR_R2:
        parser.error("--stair-diagnostics requires MiniHack Corridor R2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    agent.world.to(args.device)
    agent.actor.to(args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    modes: dict[str, PolicyModeReport] = {}
    report: dict[str, object] = dict(
        checkpoint=agent.checkpoint,
        update=agent.update,
        seeds=seeds,
        sampling_seed=args.sampling_seed,
        device=args.device,
        modes=modes,
        limitations="One sampling replicate per level; paired returns are descriptive. No movement may be "
        "legitimate for OPEN/KICK/SEARCH. Frozen game turns track the actual no-progress penalty. "
        "Positions and game turns are diagnostic labels only. Terminated does not alone imply success.",
    )
    for sample_latents in (False, True):
        for sample_actions in (False, True):
            name = (
                f"{'sampled' if sample_actions else 'greedy'}_actions_{'sampled' if sample_latents else 'mode'}_latents"
            )
            episodes: list[EpisodeDiagnostics] = []
            stair_episodes: list[CorridorEpisode] | None = [] if args.stair_diagnostics else None
            returns = evaluate_minihack(
                agent.cfg,
                agent.world,
                agent.actor,
                seeds,
                sample_actions=sample_actions,
                sample_latents=sample_latents,
                sampling_seed=args.sampling_seed,
                episode_diagnostics=episodes,
                corridor_diagnostics=stair_episodes,
            )
            steps = sum(e["steps"] for e in episodes)
            counts = {key: sum(e["action_counts"][key] for e in episodes) for key in episodes[0]["action_counts"]}
            reversal_pairs = sum(max(e["steps"] - 1, 0) for e in episodes)
            summary: dict[str, object] = dict(
                mean_return=mean(returns),
                mean_episode_steps=mean(e["steps"] for e in episodes),
                truncated_episodes=sum(e["truncated"] for e in episodes),
                mean_unique_tiles=mean(e["unique_tiles"] for e in episodes),
                no_movement_rate=sum(e["stationary_steps"] for e in episodes) / steps,
                frozen_game_turn_rate=sum(e["frozen_game_turns"] for e in episodes) / steps,
                mean_longest_stationary_streak=mean(e["longest_stationary_streak"] for e in episodes),
                max_stationary_streak=max(e["longest_stationary_streak"] for e in episodes),
                two_step_return_rate=(
                    sum(e["two_step_returns"] for e in episodes) / reversal_pairs if reversal_pairs else 0
                ),
                action_counts=counts,
                action_frequencies={key: value / steps for key, value in counts.items()},
                action_names=episodes[0]["action_names"],
            )
            mode: PolicyModeReport = {"summary": summary, "episodes": episodes, "returns": returns}
            display_summary = summary.copy()
            if stair_episodes is not None:
                stairs = summarize_stair_opportunities(stair_episodes)
                summary["stair_opportunities"] = stairs
                display_summary["stair_opportunities"] = {
                    key: value if value is not None else "N/A" for key, value in stairs.items()
                }
                mode["stair_episodes"] = stair_episodes
            modes[name] = mode
            write_json(args.output, report)
            print(name, display_summary, flush=True)
    report["sampled_action_effect"] = {
        latent: compare_returns(
            modes[f"sampled_actions_{latent}_latents"]["returns"],
            modes[f"greedy_actions_{latent}_latents"]["returns"],
        )
        for latent in ("mode", "sampled")
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()
