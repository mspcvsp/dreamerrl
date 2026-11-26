"""Read-only Room 5x5 audit, run with the v1.1 tagged tree on PYTHONPATH."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from torch.nn.functional import one_hot

from dreamerrl.env.minihack.minihack_utils import extract_glyphs
from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.models.goal_autoencoder import goal_features
from dreamerrl.models.hierarchical_actor import policy_action

TAG = "dreamerv3-director-room5x5-v1.1"
CHECKSUM = "497115a718e1b9becc401b64f2ffea5182d5e658b0bd7dc5f5020d78ee7a86f2"
SEEDS = range(839200, 839240)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def stats(values):
    values = np.asarray(values)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=float(values.max())) if len(values) else None


def feature(state):
    return goal_features(state.h, state.z)[0]


def scores(state, goals):
    return torch.nn.functional.cosine_similarity(feature(state)[None], goals).clamp(-1, 1).tolist()


def position(obs):
    return [int(v) for v in obs["blstats"][:2]]


def reset(agent, env, seed):
    env.unwrapped.seed(core=seed, disp=seed, reseed=False, lgen=seed)
    obs, _ = env.reset(seed=seed)
    return obs, observe(agent, agent.world.init_state(1), obs, None)


def encode_action(agent, action):
    return one_hot(torch.tensor([action]), agent.world.net_cfg.action_dim).float()


def observe(agent, state, obs, action):
    first = action is None
    previous = torch.zeros(1, agent.world.net_cfg.action_dim) if first else encode_action(agent, action)
    glyphs = torch.as_tensor(extract_glyphs(obs).reshape(1, -1), dtype=torch.float32)
    return agent.world.observe_step(state, glyphs, previous, is_first=torch.tensor([first]))["post"]


def prepared_start(agent, env, seed, prefix):
    obs, state = reset(agent, env, seed)
    for action in prefix:
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            return None
        state = observe(agent, state, obs, action)
    return obs, state


def compare_probabilities(actor, state, goals, phase):
    h, z = state.h.expand(len(goals), -1), state.z.expand(len(goals), -1, -1)
    probabilities = actor.worker_logits(h, z, goals, torch.full((len(goals),), phase)).softmax(-1)
    return probabilities.numpy()


def select(p, uniform):
    return int(p.argmax() if uniform is None else min(np.searchsorted(p.cumsum(), uniform), len(p) - 1))


def branch(agent, env, seed, prefix, goals, code, uniforms):
    obs, state = prepared_start(agent, env, seed, prefix)
    actor = agent.actor
    prior = state
    real, imagined = [scores(state, goals)], [scores(prior, goals)]
    route, actions, probs, positions = [], [], [], [position(obs)]
    for step, random in enumerate(uniforms):
        p = compare_probabilities(actor, state, goals, actor.interval - step)
        probs.append(p.tolist())
        action = select(p[code], random)
        obs, _, term, trunc, _ = env.step(action)
        state = observe(agent, state, obs, action)
        prior = agent.world.imagine_step(prior, actor.worker, action=encode_action(agent, action),
                                         deterministic_imagination=True)
        route.append(position(obs))
        positions.append(position(obs))
        actions.append(action)
        real.append(scores(state, goals))
        imagined.append(scores(prior, goals))
        if term or trunc:
            break
    after = []
    policy_state = None
    if not (term or trunc):
        for _ in range(2):
            action, _, policy_state = policy_action(actor, state, policy_state, deterministic=True)
            obs, _, term, trunc, _ = env.step(int(action.item()))
            state = observe(agent, state, obs, int(action.item()))
            after.append(scores(state, goals))
            if term or trunc:
                break
    return dict(code=code, real=real, imagined=imagined, after=after, probabilities=probs,
                path=route, actions=actions, start_position=positions[0], ended=bool(term or trunc))


def spatial_trace(agent, env, seed, rng, goal_scores, rows, features, codes, goal_vectors):
    obs, state = reset(agent, env, seed)
    start = np.array(position(obs))
    for t in range(17):
        end = np.array(position(obs))
        rows.append((seed, t, *start, *(end - start)))
        features.append(feature(state).numpy().copy())
        goal_scores.append(scores(state, goal_vectors))
        codes.append(int(agent.actor.goal_autoencoder.encoder(feature(state)).argmax()))
        action = int(rng.integers(agent.world.net_cfg.action_dim))
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
        state = observe(agent, state, obs, action)


def summarise(blocks):
    pair = np.triu_indices(8, 1)
    result = {}
    for mode in ("greedy", "sampled"):
        groups = [b[mode] for b in blocks]
        rows = [r for block in groups for r in block]
        tv = []
        max_delta = []
        argmax_fraction = []
        for block in groups:
            p = np.asarray(block[0]["probabilities"][0])
            a, b = p[pair[0]], p[pair[1]]
            tv.extend(np.abs(a - b).sum(1) / 2)
            max_delta.extend(np.abs(a - b).max(1))
            argmax_fraction.extend(a.argmax(1) != b.argmax(1))
        result[mode] = dict(
            starts=len(groups), pairs_per_comparison=len(tv), start_tv=stats(tv), start_max_probability_change=stats(max_delta),
            start_argmax_change=float(np.mean(argmax_fraction)),
            action_sequence_change=float(np.mean([g[i]["actions"] != g[j]["actions"] for g in groups for i, j in zip(*pair)])),
            path_change=float(np.mean([g[i]["path"] != g[j]["path"] for g in groups for i, j in zip(*pair)])),
            endpoint_change=float(np.mean([g[i]["path"][-1] != g[j]["path"][-1] for g in groups for i, j in zip(*pair)])),
            branch_state_tv=stats([
                v for r in rows for p in np.asarray(r["probabilities"])[1:]
                for v in np.abs(p[pair[0]] - p[pair[1]]).sum(1) / 2
            ]),
            post_interval_change=stats([r["after"][-1][r["code"]] - r["real"][-1][r["code"]] for r in rows if r["after"]]),
        )
        for domain in ("real", "imagined"):
            curves = [np.asarray(r[domain]) for r in rows]
            selected = [c[:, r["code"]] for c, r in zip(curves, rows)]
            gaps = [c[-1, r["code"]] - np.delete(c[-1], r["code"]).mean() for c, r in zip(curves, rows)]
            result[mode][domain] = dict(
                initial=stats([c[0] for c in selected]), peak=stats([max(c[1:]) for c in selected]),
                final=stats([c[-1] for c in selected]), mean_final_minus_initial=float(np.mean([c[-1]-c[0] for c in selected])),
                mean_target_minus_other_at_final=float(np.mean(gaps)),
                peak_90=float(np.mean([max(c[1:]) >= .9 for c in selected])),
                initial_90=float(np.mean([c[0] >= .9 for c in selected])),
                newly_crossed_90=float(np.mean([c[0] < .9 <= max(c[1:]) for c in selected])),
                code7=dict(initial=stats([c[0] for c,r in zip(selected,rows) if r["code"] == 7]),
                           final=stats([c[-1] for c,r in zip(selected,rows) if r["code"] == 7])),
            )
    return result


def spatial_report(rows, features, goal_scores, codes, goals, probes):
    rows, x, similarity = np.asarray(rows), np.asarray(features), np.asarray(goal_scores)
    masks, groups = probes.episode_split(rows[:, 0], 839200)
    control = rows[:, 1:4]
    results = {}
    for axis, y in zip(("x", "y"), rows[:, 4:].T):
        results[axis] = {}
        for name, matrix in (("start_position_and_time", control),
                             ("plus_decoded_goal_similarities", np.column_stack((control, similarity))),
                             ("plus_full_latent", np.column_stack((control, x)))):
            fit = probes.ridge(matrix, y, masks, [.001, .01, .1, 1.])
            results[axis][name] = probes.metrics(y[masks["test"]], fit["predictions"][masks["test"]])
    # Nearest visited state is descriptive. Use test levels only and report the
    # relative position because absolute room placement can change by seed.
    held = np.flatnonzero(masks["test"])
    nearest = []
    for code, sim in enumerate(similarity.T):
        top = held[np.argsort(-sim[held])[:20]]
        nearest.append(dict(code=code, similarity=stats(sim[top]), relative_positions=rows[top, 4:].tolist(),
                            unique_displacements=len(np.unique(rows[top, 4:], axis=0))))
    unit = goals / np.maximum(np.linalg.norm(goals, axis=1)[:, None], 1e-8)
    pair = (unit @ unit.T)[np.triu_indices(len(goals), 1)]
    return dict(probe=results, level_counts={k: len(v) for k,v in groups.items()}, nearest=nearest,
                encoder_code_counts=np.bincount(codes, minlength=len(goals)).tolist(),
                goal_pair_cosine=stats(pair), pairs_above_098=int((pair > .98).sum()),
                decoded_effective_rank=probes.spectral_rank(goals)["effective_rank"])


def markdown(report):
    g, s = (report["sensitivity"][m] for m in ("greedy", "sampled"))
    spatial = report["spatial"]
    summaries = [
        "# Director Room 5x5 goal audit", "",
        f"Frozen checkpoint update 2,216. {g['starts']} paired starts, all eight codes per start.",
        "", "## Root-cause assessment", "",
        f"Across identical start states, changing the goal shifted the worker action distribution by mean total variation {g['start_tv']['mean']:.4f} "
        f"(95th percentile {g['start_tv']['p95']:.4f}, maximum {g['start_tv']['max']:.4f}). Argmax differed for {g['start_argmax_change']:.1%} of code pairs.",
        f"Greedy spatial paths differed for {g['path_change']:.1%} of pairs. Common-random-number sampled paths differed for {s['path_change']:.1%}, with endpoint differences in {s['endpoint_change']:.1%}. "
        "The sampled intervention varies only action sampling. Posterior modes and the initial RNG seed are shared.",
        f"Decoded effective rank was {spatial['decoded_effective_rank']:.2f}, with {spatial['pairs_above_098']} of 28 decoded pairs above 0.98 cosine. "
        f"Hard encoder assignments on visited states were {spatial['encoder_code_counts']}. This occupancy measures the encoder, while manager concentration was established by the earlier policy evaluation.",
        "", "## Implementation audit", "",
        "- Replay posterior h and flattened z are detached and reconstructed with unweighted feature MSE. The straight-through code is hard argmax. The entropy bonus acts on mean soft probabilities, so high soft entropy can coexist with hard assignment collapse. Reconstruction and entropy do not constrain decoded goals to be distinct, spatial, or reachable.",
        "- The manager samples a categorical code during stochastic execution and optimizes task return plus policy entropy. It receives no credit for behavioral diversity. The worker consumes h, z, the latched decoded goal and remaining phase. Its critic conditions on code ID and phase, so a decoder update can change a code target while the critic still sees the same ID.",
        "- The autoencoder has a separate optimizer. The world receives gradients from its own loss, and the worker never backpropagates into the decoder. Actor losses use the exact goal latched during imagination. The collector latches goals across optimizer steps and clears them at episode end, including truncation.",
        "- The intrinsic reward uses the post-action predicted RSSM state for the goal that conditioned that action. Worker discounting cuts at the goal boundary. No off-by-one reward or goal reset defect was identified. Manager tail bootstrap uses the task critic even when imagination ends mid-interval, an approximation.",
        "- The autoencoder objective is added only when the optional diagnostics mapping is present. The trainer supplies it, so this is an API coupling defect with no evidence that it caused this checkpoint. Training imagines from replay posterior states and uses predicted continuation. Evaluation follows real end flags and can summarize peak similarity after only one action. These metrics answer different questions.",
        "", "## Similarity and temporal alignment", "",
        "| Worker | Domain | Initial mean | Peak mean | Final mean | Final minus initial | Final target minus other | Initial ≥ 0.90 | Newly crosses 0.90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, item in report["sensitivity"].items():
        for domain in ("real", "imagined"):
            r = item[domain]
            summaries.append(f"| {mode} | {domain} | {r['initial']['mean']:.3f} | {r['peak']['mean']:.3f} | {r['final']['mean']:.3f} | {r['mean_final_minus_initial']:+.3f} | {r['mean_target_minus_other_at_final']:+.3f} | {r['initial_90']:.1%} | {r['newly_crossed_90']:.1%} |")
    summaries += [
        "", "Imagination replays the real action sequence open-loop from the identical starting posterior. This comparison isolates model prediction error from policy sampling differences. Raw traces contain similarity to all eight codes at every step and, when an episode survives, after two further actions under a new greedy interval.",
        f"For goal 7, greedy real mean initial similarity was {g['real']['code7']['initial']['mean']:.3f}, and final similarity was {g['real']['code7']['final']['mean']:.3f}. High similarity near the start can earn reward without targeted progress.",
        "", "## Spatial information", "",
        "Held-out ridge probes predict displacement from the initial position. The controls are initial coordinates and timestep. Goal similarities and the full visited RSSM are evaluated against the same level-disjoint split.",
        "", "| Features | X RMSE | Y RMSE |", "|---|---:|---:|",
    ]
    for key, x in spatial["probe"]["x"].items():
        summaries.append(f"| {key} | {x['rmse']:.3f} | {spatial['probe']['y'][key]['rmse']:.3f} |")
    summaries += [
        "", "Nearest visited states per decoded code are in report.json. A probe gain indicates spatial information in the representation and does not establish that the worker can reach the decoded goal.",
        "", "## Recommendations for review", "",
        "| Priority | Change | Research value | Effort | Risk |", "|---|---|---|---|---|",
        "| 1 | Replace absolute cosine reward with signed next-minus-current cosine for the same latched goal | Tests incidental reward directly | Low | Low–medium |",
        "| 2 | Learn goals from future states conditioned on the start and reachable interval | Addresses unreachable prototypes and common features | Medium | Medium |",
        "| 3 | Regularize hard occupancy and decoded redundancy, then test manager exploration separately | Tests representation collapse and manager concentration | Medium | Medium |",
        "| 4 | Stabilize latent targets with a slower goal model | Tests moving decoder/RSSM targets | Medium | Medium |",
        "| 5 | Strengthen goal injection in worker layers | Tests underconditioning after the target is informative | Medium | Medium–high |",
        "", "Change only the worker reward in the next comparison, preserving the task objective, goal interval, codes and evaluation. Signed progress is a targeted test, and manager diversity may still require a separate change.",
        "", "## Proposed Room 5x5 acceptance criteria", "",
        "- Compare paired independent training seeds at a fixed interaction budget and retain at least +0.90 mean held-out task return.",
        "- Require positive goal-specific final-minus-initial similarity and a positive advantage over matched alternative codes. Pre-register thresholds and use paired confidence intervals over held-out starts.",
        "- Require at least 10% distinct endpoints under forced code changes and median start action TV of at least 0.05. Report greedy and common-random-number sampled results separately.",
        "- Require at least two effectively used manager goals and evidence that goal-conditioned features improve held-out spatial prediction over the initial-state and timestep controls. Keep Corridor R2 pending these checks.",
    ]
    return "\n".join(summaries) + "\n"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "checkpoint", "manifest", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--ready-channel")
    args = parser.parse_args()
    import dreamerrl

    checkout = args.checkout.resolve()
    def git(*items):
        return subprocess.check_output(["git", "-C", checkout, *items], text=True).strip()

    manifest = json.loads(args.manifest.read_text())
    if (git("rev-parse", "HEAD") != manifest["source_commit"]
        or git("rev-parse", TAG + "^{}") != manifest["source_commit"]
        or git("rev-parse", TAG) != manifest["tag_object"]
        or not Path(dreamerrl.__file__).resolve().is_relative_to(checkout)
        or git("diff", "--name-only", "HEAD", "--", "dreamerrl", "environments/cage2/latent_probe.py", "uv.lock")
        or digest(args.checkpoint) != CHECKSUM):
        raise ValueError("Checkpoint or tagged source provenance mismatch")
    for name in ("dreamerrl", "environments/cage2/latent_probe.py", "uv.lock"):
        for relative in git("ls-files", "--", name).splitlines():
            if digest(checkout / relative) != manifest["source_fingerprints"][relative]:
                raise ValueError(f"Source fingerprint mismatch: {relative}")
    sys.path.insert(0, str(checkout / "environments/cage2"))
    import latent_probe
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    agent = load_frozen_agent(args.checkpoint, allow_hierarchical=True)
    if agent.update != 2216 or agent.cfg.env.env_id != "MiniHack-Room-5x5-v0" or agent.actor.num_goals != 8:
        raise ValueError("Unexpected checkpoint")
    assert not any(p.requires_grad for m in (agent.actor, agent.world) for p in m.parameters())
    goals = agent.actor.decode_goals(torch.arange(8))
    blocks, spatial, x, scores_seen, codes = [], [], [], [], []
    with TimeLimit(gym.make(agent.cfg.env.env_id), max_episode_steps=agent.cfg.env.max_episode_steps) as env:
        for index, seed in enumerate(SEEDS):
            rng = np.random.default_rng(seed)
            prefix = rng.integers(agent.world.net_cfg.action_dim, size=index % 4).tolist()
            if prepared_start(agent, env, seed, prefix) is not None:
                uniforms = rng.random(agent.actor.interval)
                blocks.append(dict(seed=seed, prefix=prefix, **{
                    mode: [branch(agent, env, seed, prefix, goals, code, random) for code in range(8)]
                    for mode, random in (("greedy", [None] * agent.actor.interval), ("sampled", uniforms))
                }))
            spatial_trace(agent, env, seed, rng, scores_seen, spatial, x, codes, goals)
            if index == 0:
                dump(args.output / "provenance.json", dict(
                    tag=TAG, revision=manifest["source_commit"], tag_object=manifest["tag_object"],
                    checkpoint_sha256=CHECKSUM, manifest_sha256=digest(args.manifest), entry_sha256=digest(__file__),
                    lock_sha256=digest(checkout / "uv.lock"), seeds=list(SEEDS), torch=torch.__version__, python=sys.version,
                ))
                print("Verified checkpoint, tagged source, and first complete paired sweep", flush=True)
                dump(args.output / "started.json", dict(first_seed=seed, matched_goals=8))
                if args.ready_channel:
                    subprocess.run(["tmux", "wait-for", "-S", args.ready_channel], check=True)
    result = dict(sensitivity=summarise(blocks), spatial=spatial_report(spatial, x, scores_seen, codes, goals.numpy(), latent_probe))
    if digest(args.checkpoint) != CHECKSUM:
        raise ValueError("Checkpoint fingerprint changed")
    dump(args.output / "paired_traces.json", blocks)
    np.savez_compressed(args.output / "visited_states.npz", context=spatial, features=x, similarity=scores_seen, encoder_codes=codes)
    dump(args.output / "report.json", result)
    (args.output / "report.md").write_text(markdown(result))


if __name__ == "__main__":
    main()
