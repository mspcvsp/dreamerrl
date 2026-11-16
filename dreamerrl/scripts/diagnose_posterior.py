"""Compare frozen checkpoints for posterior under-use without changing training."""

import argparse
from pathlib import Path

import torch

from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.eval.posterior_diagnostics import (
    collect_trace,
    conditional_sensitivity,
    filter_trace,
    posterior_usage,
    reconstruction_ablations,
)
from dreamerrl.training.run_monitor import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=6000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation-frames", type=int, default=64)
    parser.add_argument("--ablation-cells", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.episodes, args.max_steps, args.ablation_frames, args.ablation_cells, args.threads) < 1:
        parser.error("Counts and limits must be positive")
    if min(args.seed_start, args.seed) < 0:
        parser.error("Seeds must be nonnegative")
    torch.set_num_threads(args.threads)
    first = load_frozen_agent(args.checkpoint[0])
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    trace = collect_trace(first, seeds, args.max_steps, args.seed)
    report = {
        "seeds": seeds,
        "max_steps": args.max_steps,
        "observations": len(trace["obs"]),
        "trajectory_policy": "random, identical across models",
        "checkpoints": [],
        "limitations": "Marginal entropy gap includes history variation; it is not conditional mutual information. "
        "Observation swaps can be off-manifold. Decoder grid CE uses sampled cells; player-cell "
        "CE is a separate privileged diagnostic, not a training target.",
    }
    for index, path in enumerate(args.checkpoint):
        agent = first if index == 0 else load_frozen_agent(path)
        if agent.cfg.env != first.cfg.env or agent.cfg.world != first.cfg.world:
            parser.error("Compared checkpoints must share world and environment configurations")
        checkpoint = {"path": str(path.resolve()), "update": agent.update, "modes": {}}
        for mode in ("mode", "sampled"):
            states = filter_trace(agent, trace, sampled=mode == "sampled", seed=args.seed)
            usage = posterior_usage(states["q"], states["prior"], states["z"], agent.cfg.world.kl_cfg.free_nats)
            result = {
                "usage": usage,
                "sensitivity": conditional_sensitivity(agent.world, states, trace, args.seed),
                "reconstruction": reconstruction_ablations(
                    agent.world, states, trace, frames=args.ablation_frames, cells=args.ablation_cells, seed=args.seed
                ),
            }
            checkpoint["modes"][mode] = result
            print(
                f"update={agent.update} {mode}: KL={usage['raw_kl_total_mean_nats']:.5f} "
                f"H/logC={usage['posterior_normalized_entropy']:.4f} "
                f"obs-swap JS={result['sensitivity']['observation_swap_js_total_nats']:.5f}",
                flush=True,
            )
        report["checkpoints"].append(checkpoint)
        write_json(args.output, report)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
