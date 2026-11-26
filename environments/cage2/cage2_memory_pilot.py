"""Bounded archive parity and offline active-session labels for a frozen flat agent."""
from pathlib import Path
import json

import numpy as np
import torch

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
from dreamerrl.eval.memory.adapters.cage2 import CAGE2Replay, active_host_labels  # noqa: F401 (legacy import)
from dreamerrl.eval.memory.distributions import paired_categorical, posterior_summary
from dreamerrl.eval.memory.dreamer import CheckpointSpec, DreamerAdapter, file_sha256, module_sha256, verified_payload
from dreamerrl.eval.memory.legacy_ridge import metrics, ridge
from dreamerrl.eval.memory.presentation import MIST, NAVY, ORANGE, TEAL, plt, save_figure, style
from dreamerrl.eval.memory.reporting import write_json
from dreamerrl.eval.memory.replay import replay_coverage
from dreamerrl.eval.memory.statistics import binary_metrics, binary_probe, bootstrap_mean, grouped_split, l2_distance
from dreamerrl.eval.memory.trajectory import filter_trajectory
from dreamerrl.utils.transforms import symlog


def archive_parity(folder):
    features = folder / "frozen_features.npz"
    with np.load(features, allow_pickle=False) as data:
        observation, target, ids, timestep = (data[k] for k in ("observation", "target", "episode", "timestep"))
        q, z, rewards = (data[k] for k in ("posterior_probs", "z", "reward"))
        if observation.shape != (10800, 52) or q.shape != (10800, 32, 30):
            raise ValueError("Unexpected pilot archive schema")
        if not (np.array_equal(z.argmax(-1), q.argmax(-1)) and np.all(z.sum(-1) == 1)):
            raise ValueError("Archived hard states do not match posterior modes")
        episodes = np.unique(ids)
        if not (np.array_equal(ids, np.repeat(episodes, 30)) and np.array_equal(timestep, np.tile(np.arange(30), len(episodes)))):
            raise ValueError("Archive frames do not contain 30 ordered steps per episode")
        expected_target = np.cumsum(rewards.reshape(-1, 30)[:, ::-1], axis=1)[:, ::-1].flatten()
        np.testing.assert_array_equal(expected_target, target)
        config = json.loads((folder / "resolved_analysis_config.json").read_text())
        split, groups = grouped_split(ids, config["split_seed"])
        expected = json.loads((folder / "episode_splits.json").read_text())
        if {k: v.tolist() for k, v in groups.items()} != expected:
            raise ValueError("Archive split mismatch")
        model = ridge(observation, target, split, config["ridge_alphas"])
        measured = metrics(target[split["test"]], model["predictions"][split["test"]])
        historical = next(row for row in json.loads((folder / "linear_probe_comparison.json").read_text())
                          if row["features"] == "observation")
        if model["alpha"] != historical["alpha"]:
            raise ValueError("Historical regularization differs")
        np.testing.assert_allclose([measured[k] for k in ("r2", "mae", "rmse")],
                                   [historical["test_" + k] for k in ("r2", "mae", "rmse")], atol=1e-10, rtol=1e-10)
    return dict(features=str(features), sha256=file_sha256(features), episodes=len(episodes),
                observations=len(ids), matched_metrics=measured, alpha=model["alpha"],
                missing=["encoder", "raw_logits", "sampled_states", "offline_session_labels"])


def coverage_by_host(truth, split, hosts):
    return {host: {fold: dict(positive_frames=int(truth[mask, :, i].sum()),
                positive_episodes=int(truth[mask, :, i].any(1).sum()),
                negative_episodes=int((~truth[mask, :, i]).any(1).sum()))
            for fold, mask in split.items()} for i, host in enumerate(hosts)}


def choose_host(coverage):
    # Select using fitting support only; no optimization on test prevalence.
    support = {h: min(v["fit"]["positive_episodes"], v["fit"]["negative_episodes"]) for h, v in coverage.items()}
    return max(support, key=support.get) if max(support.values(), default=0) else None


def probe_table(features, target, groups, split, seed, variants):
    y = target.flatten()
    result, scores = {}, {}
    if not all(np.unique(y[split[k]]).size == 2 for k in ("fit", "validation")):
        return result, scores
    test = split["test"]
    for name, value in features.items():
        x = value.reshape(len(y), value.shape[2], -1).numpy()
        result[name] = {}
        for family in ("linear", "rbf") if name in variants else ("linear",):
            fit = binary_probe(x, y, split, nonlinear=family == "rbf", balanced=True)
            predictions = fit.pop("scores")[test]
            scores[name, family] = predictions
            result[name][family] = dict(fit, **binary_metrics(predictions, y[test], groups[test], seed))
    return result, scores


def compare_probes(scores, target, groups, seed):
    """Paired bootstrap for incremental balanced accuracy on held-out episodes."""
    results = {}
    ids = np.unique(groups)
    index = np.random.default_rng(seed).integers(len(ids), size=(2000, len(ids)))
    for first, reference in (("observation_time_h", "observation_time"),
                             ("observation_time_hz", "observation_time"),
                             ("observation_time_hz", "observation_time_fixed_hz")):
        if (first, "linear") not in scores or np.unique(target).size != 2:
            continue
        correct = [((scores[k, "linear"] >= 0) == target[:, None]).mean(1) for k in (first, reference)]
        diff = correct[0] - correct[1]
        row = np.array([[diff[(groups == g) & target].sum(), ((groups == g) & target).sum(),
                         diff[(groups == g) & ~target].sum(), ((groups == g) & ~target).sum()] for g in ids])
        totals = row.sum(0)
        a, p, b, n = row[index].sum(1).T
        valid = (p > 0) & (n > 0)
        results[f"{first} - {reference}"] = dict(
            delta=float((totals[0]/totals[1] + totals[2]/totals[3])/2),
            ci95=np.quantile((a[valid]/p[valid] + b[valid]/n[valid])/2, [.025, .975]).tolist() if valid.any() else None,
            valid_class_bootstrap_draws=int(valid.sum()))
    return results


def reconstruction(model, stages, obs, test):
    h, z = stages["h"][test], stages["z"][test]
    native = obs[test]
    if not ((native == 0) | (native == 1)).all():
        raise ValueError("Expected binary native BlueTable")
    decoded = model.world.decoder(h.flatten(0, 1), z.flatten(0, 1)).reshape_as(native)
    correct = (decoded >= np.log(2) / 2) == native.bool()
    error = (decoded - symlog(native)).square().unflatten(-1, (13, 4))
    correct = correct.unflatten(-1, (13, 4))
    visible_compromise = native.unflatten(-1, (13, 4))[..., 2:]
    correct_compromise = correct[..., 2:].all(-1)
    compromised = visible_compromise[..., 1] == 1
    return dict(all_bit_accuracy=float(correct.float().mean()), all_symlog_mse=float(error.mean()),
        visible_activity_pair_exact=float(correct[..., :2].all(-1).float().mean()),
        visible_compromise_pair_exact=float(correct_compromise.float().mean()),
        visible_compromise_symlog_mse=float(error[..., 2:].mean()),
        visible_positive_compromise_pair_exact=float(correct_compromise[compromised].float().mean()) if compromised.any() else None,
        visible_positive_frames=int(compromised.sum()))


def peak_reward(episode):
    # Align with a recorded nonzero reward in the latter half when available.
    start = len(episode["obs"]) // 2
    candidates = torch.nonzero(episode["reward"][start:]).flatten()
    return start + int(candidates[0]) if len(candidates) else len(episode["obs"]) - 1


@torch.inference_mode()
def run(root, spec, out, should_stop):
    cache = out / "results.json"
    if cache.is_file() and (out / "report.md").is_file():
        previous = json.loads(cache.read_text())
        inputs = [
            (Path(previous["checkpoint_path"]), previous["checkpoint"]["sha256"]),
            (Path(previous["source_cohort"]) / "provenance.json", previous["source_cohort_sha256"]),
            (Path(previous["action_archive"]), previous["action_archive_sha256"]),
            (Path(previous["archive_parity"]["features"]), previous["archive_parity"]["sha256"]),
        ]
        if any(file_sha256(path) != digest for path, digest in inputs):
            raise ValueError("Cached CAGE2 inputs changed")
        return True
    archive = archive_parity(root / spec["host_pilot_features"])
    write_json(out / "archive_parity.json", archive)
    if should_stop():
        return False
    cohort = root / spec["matched"]
    provenance = json.loads((cohort / "provenance.json").read_text())
    member = next(row for row in provenance["checkpoints"] if row["id"] == spec["checkpoint_id"])
    path = root / member["path"]
    checkpoint = CheckpointSpec(path, member["sha256"], member["training_seed"], 16899, 507450)
    payload = verified_payload(checkpoint)
    cfg = payload["config"]
    if cfg.world.obs_encoder != "cage2_host" or cfg.env.finite_horizon or cfg.env.max_episode_steps != 30:
        raise ValueError("Checkpoint has incompatible Blue encoder or horizon")
    replay = payload["replay"]
    endpoints = {id(ep): peak_reward(ep) for ep in replay["episodes"]}
    coverage = dict(final_stored_row=replay_coverage(replay),
                    late_reward_or_endpoint=replay_coverage(replay, delayed_at=lambda ep: endpoints[id(ep)]),
                    episodes_with_late_reward=sum(bool(ep["reward"][len(ep["obs"])//2:].any()) for ep in replay["episodes"]))
    write_json(out / "replay_audit.json", coverage)
    reference_mapping = root / spec["action_mapping"]
    actions_mapping = json.loads(reference_mapping.read_text())
    action_archive = cohort / f"{member['id']}_b-line.jsonl"
    rows = {r["seed"]: r for line in action_archive.read_text().splitlines() if (r := json.loads(line))}
    seeds = np.random.default_rng(spec["split_seed"]).choice(sorted(provenance["seeds"]), spec["episodes"], replace=False)
    adapter = CAGE2Replay(cfg, rows, provenance["scenario_sha256"], actions_mapping)
    env = Cage2BlueEnv(cfg.env, torch.device("cpu"))
    try:
        adapter._check_env(env)
        model = DreamerAdapter.from_payload(payload, env.obs_space, env.action_dim)
    finally:
        env.close()
    del payload, replay
    if should_stop():
        return False
    ids = [int(x) for x in seeds]
    hashes = {key: module_sha256(getattr(model, key)) for key in ("world", "actor", "critic")}
    episodes = adapter.collect(ids)
    observed = adapter.model_input(torch.stack([ep.observations for ep in episodes]))
    actions = torch.stack([ep.actions for ep in episodes])
    labels = [adapter.labels(seed) for seed in ids]
    truth = np.stack([label.values for label in labels])
    if not all(label.valid.all() for label in labels):
        raise ValueError("Unexpected missing per-host labels")
    states, mode = filter_trajectory(model, observed, actions)
    fixed_states, fixed = filter_trajectory(model, observed, actions, history=1)
    # Batched inference is numerically close to the archived single-episode evaluator.
    readouts = [model.readouts(state) for state in states]
    actor = torch.stack([row["action_probability"] for row in readouts], dim=1)
    chosen = torch.stack([row["action_logits"].argmax(-1) for row in readouts], dim=1)
    mismatches = chosen != actions
    # Recheck any mismatched episode using the original batch size and rounding path.
    for i in mismatches.any(1).nonzero().flatten().tolist():
        singleton, _ = filter_trajectory(model, observed[i:i+1], actions[i:i+1])
        chosen[i] = torch.stack([model.readouts(s)["action_logits"].argmax(-1) for s in singleton]).flatten()
    np.testing.assert_array_equal(chosen, actions)
    split, group_ids = grouped_split(seeds, spec["split_seed"])
    for state_a, state_b in zip(states[:1], fixed_states[:1], strict=True):
        if not torch.equal(state_a.z, state_b.z):
            raise ValueError("Reset state differs between factual and fixed-history controls")
    fixed_readouts = [model.readouts(s) for s in fixed_states]
    fixed_actor = torch.stack([row["action_probability"] for row in fixed_readouts], dim=1)
    fixed_actions = torch.stack([row["action_logits"].argmax(-1) for row in fixed_readouts], dim=1)
    value, control_value = [torch.stack([row["value"] for row in sequence], dim=1)
                            for sequence in (readouts, fixed_readouts)]
    test = split["test"]
    analysis_seed = spec["rng_seed"]
    readout = dict(actor_total_variation=bootstrap_mean((actor - fixed_actor).abs()[test].sum(-1).mean(1).numpy()/2, analysis_seed),
                   changed_greedy_fraction=bootstrap_mean((actions != fixed_actions)[test].float().mean(1).numpy(), analysis_seed),
                   critic_absolute_change=bootstrap_mean((value-control_value)[test].abs().mean(1).numpy(), analysis_seed))
    latent = dict(argmax=posterior_summary(mode["posterior"][test], mode["prior"][test], mode["z"][test]),
        history_pair_before_unimix=paired_categorical(mode["pre_unimix"][test], fixed["pre_unimix"][test]),
        history_pair_after_unimix=paired_categorical(mode["posterior"][test], fixed["posterior"][test],
                                                  mode["z"][test], fixed["z"][test]),
        history_pair_l2={stage: bootstrap_mean(l2_distance(mode[stage][test].flatten(0, 1),
                             fixed[stage][test].flatten(0, 1)).reshape(test.sum(), -1).mean(1).numpy(), analysis_seed)
                         for stage in ("encoder", "raw_logits", "pre_unimix", "posterior", "z", "h", "hz")})
    if should_stop():
        return False
    samples = []
    rng = torch.Generator().manual_seed(analysis_seed)
    for draw in range(spec["posterior_draws"]):
        _, sampled = filter_trajectory(model, observed, actions, mode="sample", rng=rng)
        samples.append({k: sampled[k] for k in ("posterior", "prior", "z", "h", "hz")})
    raw_samples = {key: torch.stack([draw[key] for draw in samples], dim=2) for key in samples[0]}
    latent["sample"] = posterior_summary(*[raw_samples[k][test] for k in ("posterior", "prior", "z")])
    host_stats = coverage_by_host(truth, split, cfg.world.cage2_host_order)
    host = choose_host(host_stats)
    features = {stage: mode[stage].unsqueeze(2) for stage in ("observation", "encoder", "raw_logits", "pre_unimix", "posterior", "z", "h", "hz")}
    features.update({f"sample_{key}": raw_samples[key] for key in ("z", "h", "hz")})
    time = torch.eye(observed.shape[1]).expand(len(seeds), -1, -1).unsqueeze(2)
    features.update(time=time, observation_time=torch.cat((features["observation"], time), dim=-1),
                    fixed_h=fixed["h"].unsqueeze(2), fixed_hz=fixed["hz"].unsqueeze(2))
    for name in ("h", "hz", "fixed_hz"):
        features["observation_time_" + name] = torch.cat((features["observation_time"], features[name]), -1)
    frame_split = {name: np.repeat(mask, observed.shape[1]) for name, mask in split.items()}
    frame_ids = np.repeat(seeds, observed.shape[1])
    visible = observed.unflatten(-1, (13, 4))[..., 3].bool().any(-1).numpy()
    visible_table, _ = probe_table({k: features[k] for k in ("observation", "encoder", "posterior", "z", "sample_z")},
                                   visible, frame_ids, frame_split, analysis_seed, {"encoder", "posterior"})
    truth_table, increments = {}, {}
    if host is not None:
        target = truth[..., cfg.world.cage2_host_order.index(host)]
        truth_table, scores = probe_table(features, target, frame_ids, frame_split, analysis_seed,
                                        {"observation_time", "posterior", "observation_time_hz"})
        increments = compare_probes(scores, target.flatten()[frame_split["test"]], frame_ids[frame_split["test"]], analysis_seed)
    if file_sha256(path) != member["sha256"] or hashes != {k: module_sha256(getattr(model, k)) for k in hashes}:
        raise ValueError("Frozen checkpoint state changed")
    record = dict(archive_parity=archive, checkpoint=member, checkpoint_path=str(path), source_cohort=str(cohort),
                  source_cohort_sha256=file_sha256(cohort / "provenance.json"), action_archive=str(action_archive),
                  action_archive_sha256=file_sha256(action_archive), action_mapping_sha256=file_sha256(reference_mapping),
                  checkpoint_config_sha256=file_sha256(path.parent / "resolved_config.json"),
                  replay=coverage, behavioral_invariants=dict(episodes=len(seeds), decision_frames=actions.numel(),
                  greedy_actions_match=True, rewards_match=True, archived_nonmissing_reward_components_match=True, matched_boundaries=True,
                  module_hashes=hashes), split={k: v.tolist() for k, v in group_ids.items()},
                  host_coverage=host_stats, selected_host=host, hidden_session_probes=truth_table,
                  hidden_session_paired_deltas=increments, visible_compromise_probes=visible_table,
                  history_readout_sensitivity=readout, categorical=latent,
                  reconstruction=reconstruction(model, mode, observed, test),
                  posterior_draws=spec["posterior_draws"])
    np.savez_compressed(out / "offline_session_labels.npz", seed=seeds, labels=truth,
                       host_order=np.array(cfg.world.cage2_host_order))
    write_json(out / "results.json", record)
    render(out, record)
    return True


def render(out, record):
    style()
    table = record["hidden_session_probes"]
    rows = [(stage, family["linear"]) for stage, family in table.items()
            if family["linear"]["balanced_accuracy"] is not None]
    fig, ax = plt.subplots(figsize=(8, max(3, 2 + .28*len(rows))))
    for i, (stage, score) in enumerate(rows):
        x = score["balanced_accuracy"]
        ci = score["balanced_accuracy_ci95"]
        if ci:
            ax.hlines(i, *ci, color=TEAL, lw=2)
        ax.scatter(x, i, color=NAVY if "sample" not in stage else ORANGE, s=25, zorder=2)
    ax.axvline(.5, color=MIST, ls="--")
    ax.set(xlim=(0, 1.03), yticks=range(len(rows)), yticklabels=[k.replace("_", " ") for k, _ in rows],
           xlabel="Held-out balanced accuracy")
    if not rows:
        ax.text(.5, .5, "Insufficient class support in held-out episodes", ha="center", transform=ax.transAxes)
    save_figure(fig, out / "hidden_host_probes")
    replay = record["replay"]["final_stored_row"]
    host = record["selected_host"] or "Unavailable"
    r = record["reconstruction"]
    lines = ["# CAGE2 frozen memory diagnostics readiness", "",
             "**NO-GO for broad evaluation pending review of this bounded validation and a larger predeclared test split.**", "",
             f"The archive regression recovered the historical observation-only return probe across {record['archive_parity']['episodes']} episodes.",
             f"The matched flat checkpoint reproduced archived greedy actions, rewards, available reward components and truncation on "
             f"{record['behavioral_invariants']['episodes']} B-line seeds.", "",
             f"Supervised replay uses {replay['seq_len']} rows with {replay['burn_in']} burn-in rows. "
             "The replay audit counts legal starts reaching the final row and a late nonzero reward when present. "
             "Joint start coverage is a structural gradient-path check. Burn-in is detached.", "",
             f"Host selected using fit support only **{host}**. The figure is a pilot estimate on held-out episodes. "
             "The privileged target is an active Red session. BlueTable compromise is a separate visible label.", "",
             f"Aggregate bit accuracy was {r['all_bit_accuracy']:.6g}. Exact visible-compromise field accuracy was "
             f"{r['visible_compromise_pair_exact']:.6g}. Positive visible-compromise reconstruction is reported separately in JSON.", "",
             "The factual and fixed-history runs hold the current observation and its incoming action constant. Posterior divergence "
             "and actor/critic changes quantify sensitivity to carrying prior history. This model-side contrast cannot establish "
             "a causal effect on simulator outcomes. A null hidden-session comparison with few class-supporting episodes is inconclusive.", "",
             "![Episode-disjoint host-session estimates](hidden_host_probes.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
