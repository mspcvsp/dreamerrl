"""Score reconstruction/one-step forecast events against previous-label persistence."""

import argparse
from pathlib import Path

import torch

from dreamerrl.env.minihack.minihack_utils import glyph_scoring_schema
from dreamerrl.eval.field_events import json_metrics
from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.eval.minihack_events import score_minihack_events
from dreamerrl.eval.posterior_diagnostics import collect_trace, filter_trace
from dreamerrl.models.categorical import sample_onehot
from dreamerrl.models.glyph_model import GlyphDecoder
from dreamerrl.training.run_monitor import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=7000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sample-latents", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.episodes, args.max_steps, args.max_frames, args.threads) < 1 or min(args.seed, args.seed_start) < 0:
        parser.error("Counts must be positive and seeds nonnegative")
    torch.set_num_threads(args.threads)
    agent = load_frozen_agent(args.checkpoint)
    decoder = agent.world.decoder
    if not isinstance(decoder, GlyphDecoder):
        parser.error("Event decoding currently requires a spatial glyph checkpoint")
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    trace = collect_trace(agent, seeds, args.max_steps, args.seed)
    states = filter_trace(agent, trace, sampled=args.sample_latents, seed=args.seed)
    N = len(trace["obs"])
    indices = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))[: args.max_frames]
    frames = torch.zeros(1, N, dtype=torch.bool)
    frames[0, indices] = True
    targets = trace["obs"][:, : decoder.cells].long().unsqueeze(0)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(args.seed)
        prior_z = sample_onehot(states["prior"], deterministic=not args.sample_latents)
        reports = {}
        for name, z in (("posterior_reconstruction", states["z"]), ("one_step_prior", prior_z)):
            prediction = torch.zeros_like(targets)
            # Only selected frames are decoded; their previous true observations
            # remain available for a correctly aligned persistence comparison.
            for chunk in indices.split(16):
                prediction[0, chunk] = decoder.predict(decoder(states["h"][chunk], z[chunk]))
            reports[name] = score_minihack_events(
                prediction,
                targets,
                decoder.shape,
                is_first=trace["is_first"].unsqueeze(0),
                is_last=trace["is_last"].unsqueeze(0),
                player_positions=trace["positions"].unsqueeze(0),
                score_frames=frames,
            )
            summary = {
                key: value
                for key, value in reports[name].items()
                if key.endswith("accuracy") and any(part in key for part in ("player", "changed", "passability"))
            }
            print(name, json_metrics(summary), flush=True)
    write_json(
        args.output,
        dict(
            checkpoint=agent.checkpoint,
            update=agent.update,
            seeds=seeds,
            selected_frames=indices.tolist(),
            total_trace_frames=N,
            sampled_latents=args.sample_latents,
            player_glyph_ids=glyph_scoring_schema()[1],
            passability_radius=1,
            reports=json_metrics(reports),
            limitations="Posterior reconstruction sees the current observation; one-step prior "
            "uses real past observations and incoming actions. Terrain passability is an observable local proxy, "
            "not full "
            "movement legality. Player positions are diagnostic labels only. Rates with zero support are omitted.",
        ),
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
