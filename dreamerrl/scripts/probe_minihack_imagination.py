"""Compare posterior decoding, teacher-forced priors, and action-conditioned rollouts."""

import argparse
from pathlib import Path

import torch

from dreamerrl.eval.field_events import json_metrics
from dreamerrl.eval.imagination_probe import action_conditioned_rollout, endpoint_pair, rollout_anchors
from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.eval.minihack_events import score_minihack_events
from dreamerrl.eval.posterior_diagnostics import collect_trace, filter_trace
from dreamerrl.models.categorical import sample_onehot
from dreamerrl.models.glyph_model import GlyphDecoder
from dreamerrl.training.run_monitor import write_json


@torch.no_grad()
def compare_navigation(agent, trace, states, anchors, horizons, *, sampled=False, seed=42):
    decoder = agent.world.decoder
    if not isinstance(decoder, GlyphDecoder):
        raise ValueError("Navigation rollout scores require a spatial glyph decoder")
    first, last = trace["is_first"].unsqueeze(0), trace["is_last"].unsqueeze(0)
    predicted_states = action_conditioned_rollout(
        agent.world,
        states["h"].unsqueeze(0),
        states["z"].unsqueeze(0),
        trace["action"].unsqueeze(0),
        anchors,
        horizons,
        is_first=first,
        is_last=last,
        sampled=sampled,
        seed=seed,
    )
    endpoints = torch.unique(torch.cat([anchors[:, 1] + horizon for horizon in horizons]), sorted=True)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        teacher_z = sample_onehot(states["prior"][endpoints], deterministic=not sampled)
    caches = {}
    for name, z in (("posterior_reconstruction", states["z"][endpoints]), ("teacher_forced_one_step", teacher_z)):
        caches[name] = torch.cat(
            [
                decoder.predict(decoder(states["h"][endpoints[chunk]], z[chunk]))
                for chunk in torch.arange(len(endpoints)).split(16)
            ]
        )
    frames = torch.tensor([False, True]).expand(len(anchors), 2)
    pair_first = torch.tensor([True, False]).expand(len(anchors), 2)
    reports = {}
    for horizon in horizons:
        target_ids = anchors[:, 1] + horizon
        cache_ids = torch.searchsorted(endpoints, target_ids)
        truth = endpoint_pair(trace["obs"][None, :, : decoder.cells].long(), anchors, horizon)
        positions = endpoint_pair(trace["positions"].unsqueeze(0), anchors, horizon)
        pair_last = endpoint_pair(last, anchors, horizon)
        reference = torch.zeros_like(truth)
        reference[:, 1] = caches["posterior_reconstruction"][cache_ids]
        report = {}
        for name in ("posterior_reconstruction", "teacher_forced_one_step", "open_loop_prior"):
            prediction = torch.zeros_like(truth)
            if name == "open_loop_prior":
                state = predicted_states[horizon]
                prediction[:, 1] = torch.cat(
                    [decoder.predict(decoder(h, z)) for h, z in zip(state.h.split(16), state.z.split(16))]
                )
            else:
                prediction[:, 1] = caches[name][cache_ids]
            report[name] = score_minihack_events(
                prediction,
                truth,
                decoder.shape,
                is_first=pair_first,
                is_last=pair_last,
                player_positions=positions,
                score_frames=frames,
                reference_predictions=reference if name == "open_loop_prior" else None,
            )
        reports[horizon] = report
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--anchors", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=8000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sample-latents", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.horizons + [args.anchors, args.episodes, args.max_steps, args.threads]) < 1
        or min(args.seed_start, args.seed) < 0
    ):
        parser.error("Counts/horizons must be positive and seeds nonnegative")
    torch.set_num_threads(args.threads)
    agent = load_frozen_agent(args.checkpoint)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    trace = collect_trace(agent, seeds, args.max_steps, args.seed)
    states = filter_trace(agent, trace, sampled=args.sample_latents, seed=args.seed)
    anchors = rollout_anchors(
        trace["is_first"][None], trace["is_last"][None], max(args.horizons), args.anchors, args.seed
    )
    if len(anchors) == 0:
        parser.error("No complete within-episode rollouts at the requested maximum horizon")
    reports = compare_navigation(
        agent, trace, states, anchors, sorted(set(args.horizons)), sampled=args.sample_latents, seed=args.seed
    )
    write_json(
        args.output,
        dict(
            checkpoint=agent.checkpoint,
            update=agent.update,
            seeds=seeds,
            anchors=anchors.tolist(),
            horizons=sorted(set(args.horizons)),
            sampled_latents=args.sample_latents,
            reports=json_metrics(reports),
            limitations=(
                "Common anchors stay in one episode through the maximum horizon. Priors consume only anchor "
                "posteriors and recorded actions. Persistence holds the anchor label. Changes compare endpoint "
                "with anchor; reverted intermediate changes are not counted. Posterior and teacher-forced controls "
                "see later observations/history. Retention is undefined when the posterior decodes no cues correctly. "
                "This frozen diagnostic does not measure on-policy planning performance."
            ),
        ),
    )
    for horizon, report in reports.items():
        print(
            "horizon",
            horizon,
            {
                name: {
                    key: value.item()
                    for key, value in score.items()
                    if key
                    in (
                        "glyph_player_paired_accuracy",
                        "glyph_changed_nonterminal_accuracy",
                        "glyph_nearby_passability_paired_accuracy",
                    )
                }
                for name, score in report.items()
            },
            flush=True,
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
