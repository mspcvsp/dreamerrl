"""Run the single approved CAGE2 recent-session common-support study."""
from collections import Counter
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
from dreamerrl.eval.memory.adapters.cage2 import CAGE2Replay
from dreamerrl.eval.memory.dreamer import CheckpointSpec, DreamerAdapter, file_sha256, module_sha256, verified_payload
from dreamerrl.eval.memory.matching import ContrastFrame, exact_contrasts
from dreamerrl.eval.memory.reporting import write_json
from dreamerrl.eval.memory.statistics import binary_probe, exact_binomial_ci95
from dreamerrl.eval.memory.trajectory import filter_trajectory
from environments.cage2.history_common_support_report import render

PLAN_REVISION = "0d04596"
TIE_TOLERANCE = 1e-8
CONTROL_NAMES = ("current", "fixed_carry")
PROBE_NAMES = (*CONTROL_NAMES, "recurrent", "raw_history")


def eligible_frames(episodes, truth, target):
    frames = []
    excluded = 0
    for ep in episodes:
        labels = truth[ep.episode_id]
        bits = ep.observations.numpy()
        if bits.shape != (30, 52) or not np.isin(bits, [0, 1]).all() or labels.shape != (30,):
            raise ValueError("Unexpected observation or truth schema")
        last = None
        for t in range(target["last_timestep"] + 1):
            if labels[t]:
                last = t
            if t < target["first_timestep"]:
                continue
            if labels[t]:
                excluded += 1
                continue
            age = None if last is None else t - last
            key = (t, np.packbits(bits[t].astype(np.uint8)).tobytes(), int(ep.actions[t-1]))
            frames.append(ContrastFrame(ep.episode_id, t, key, age in target["lookback_frames"], age))
    return frames, excluded


def coverage(frames, matchable, pairs, active_excluded, episode_count):
    selected = [row for pair in pairs for row in pair]

    def counts(rows, axis):
        bins = {}
        for row in rows:
            index = getattr(row, axis)
            key = "never" if index is None else str(index)
            bins.setdefault(key, Counter())[str(int(row.positive))] += 1
        return {key: {"negative": values["0"], "positive": values["1"]} for key, values in bins.items()}

    def stratum(rows):
        return dict(frames=len(rows), episodes=len({row.episode for row in rows}),
                    by_elapsed_time=counts(rows, "time"), by_time_since_last_active=counts(rows, "age"))

    return dict(episodes_replayed=episode_count, active_current_frames_excluded=active_excluded,
                eligible_without_opposite_episode=len(frames) - len(matchable),
                qualifying_episode_disjoint_pairs=len(pairs), eligible=stratum(frames),
                with_exact_opposite_episode=stratum(matchable), selected=stratum(selected))


def paired_metrics(pairs, negative_scores, positive_scores):
    margins = np.asarray(positive_scores) - np.asarray(negative_scores)
    correct = int((margins > 0).sum())
    objectives = dict(correct=correct, ties=int((margins == 0).sum()), pairs=len(pairs),
                      correct_fraction=correct / len(pairs), exact_ci95=exact_binomial_ci95(correct, len(pairs)),
                      median_margin=float(np.median(margins)))
    rows = [dict(negative_episode=a.episode, positive_episode=b.episode, elapsed_time=b.time,
                 negative_age=a.age, positive_age=b.age, negative_score=float(neg), positive_score=float(pos))
            for (a, b), neg, pos in zip(pairs, negative_scores, positive_scores, strict=True)]
    return {**objectives, "scores": rows}


def check_controls(features):
    result = {}
    for name in CONTROL_NAMES:
        discrepancy = np.abs(features[name][0::2] - features[name][1::2])
        maximum = float(discrepancy.max()) if discrepancy.size else None
        result[name] = dict(feature_max_abs=maximum, score_max_abs=None,
                            feature_ties=None if maximum is None else bool(maximum <= TIE_TOLERANCE))
    return result


@torch.inference_mode()
def representations(model, episodes, pairs):
    obs = torch.stack([ep.observations for ep in episodes])
    actions = torch.stack([ep.actions for ep in episodes])
    states, mode = filter_trajectory(model, obs, actions)
    choices = torch.stack([model.readouts(state)["action_logits"].argmax(-1) for state in states], dim=1)
    for i in (choices != actions).any(1).nonzero().flatten().tolist():
        singleton, _ = filter_trajectory(model, obs[i:i+1], actions[i:i+1])
        choices[i] = torch.stack([model.readouts(state)["action_logits"].argmax(-1) for state in singleton]).flatten()
    if not torch.equal(choices, actions):
        raise ValueError("Frozen greedy actions differ from the archived decisions")
    if not pairs:
        return None
    _, fixed = filter_trajectory(model, obs, actions, history=1)
    index = {episode.episode_id: i for i, episode in enumerate(episodes)}
    rows = [row for pair in pairs for row in pair]
    ei = torch.tensor([index[row.episode] for row in rows])
    t = torch.tensor([row.time for row in rows])
    current = torch.cat((obs[ei, t], F.one_hot(t, 30), F.one_hot(actions[ei, t-1], model.action_dim)), dim=-1)
    past_time = t[:, None] - torch.tensor([4, 3, 2, 1])
    previous = torch.cat((torch.full_like(actions[:, :1], model.action_dim), actions[:, :-1]), dim=1)
    raw_past = torch.cat((obs[ei[:, None], past_time],
                          F.one_hot(previous[ei[:, None], past_time], model.action_dim + 1)), dim=-1).flatten(1)
    values = (current, torch.cat((current, fixed["h"][ei, t]), -1),
              torch.cat((current, mode["h"][ei, t]), -1), torch.cat((current, raw_past), -1))
    return {key: value.numpy() for key, value in zip(PROBE_NAMES, values, strict=True)}


def readout_metrics(pairs, features, predictors, controls):
    results = {}
    for name, probe in predictors.items():
        scores = probe.predict(features[name])
        if name in CONTROL_NAMES:
            maximum = float(np.abs(scores[0::2] - scores[1::2]).max())
            controls[name].update(score_max_abs=maximum, score_ties=maximum <= TIE_TOLERANCE)
        else:
            results[name] = paired_metrics(pairs, scores[0::2], scores[1::2])
    return results


def assert_controls(controls):
    for name, metrics in controls.items():
        if metrics["feature_ties"] is False or metrics.get("score_ties") is False:
            raise ValueError(f"{name} control broke its prespecified tie tolerance")


def compare_fit_predictors(fit, validation, penalties):
    lengths = (len(fit["current"]), len(validation["current"]))
    fit_mask = np.arange(sum(lengths)) < lengths[0]
    split = dict(fit=fit_mask, validation=~fit_mask, test=np.zeros_like(fit_mask))
    targets = np.tile([False, True], sum(lengths)//2)
    models, tuning = {}, {}
    for name in PROBE_NAMES:
        features = np.concatenate((fit[name], validation[name]))
        result = binary_probe(features[:, None], targets, split, balanced=True, penalties=penalties, freeze=True)
        models[name] = result["predictor"]
        np.testing.assert_allclose(models[name].predict(features), result["scores"][:, 0], atol=1e-10, rtol=1e-10)
        tuning[name] = dict(penalty=result["penalty"], validation_balanced_accuracy=result["validation_accuracy"])
    return models, tuning


def run(root, spec, out, should_stop):
    maximum = spec["maximum"]
    groups = spec["splits"]
    seeds = [seed for group in groups.values() for seed in group]
    if len(seeds) != maximum["archived_episodes"] or len(set(seeds)) != len(seeds) or set(seeds) & set(spec["excluded_pilot_seeds"]):
        raise ValueError("Pinned episode roles are invalid")
    if len(seeds) * 30 != maximum["native_steps"]:
        raise ValueError("Pinned native step budget and episode plan disagree")
    artifacts = {Path(key).name: root / key for key in spec["artifact_sha256"]}
    for key, expected in spec["artifact_sha256"].items():
        if file_sha256(root / key) != expected:
            raise ValueError(f"Pinned input hash differs for {key}")
    checkpoint = spec["checkpoint"]
    path = root / checkpoint["path"]
    payload = verified_payload(CheckpointSpec(path, checkpoint["sha256"], checkpoint["training_seed"],
                                               16899, 507450, version=checkpoint["version"]))
    config = payload["config"]
    provenance = json.loads(artifacts["provenance.json"].read_text())
    row = next(item for item in provenance["checkpoints"] if item["id"] == checkpoint["id"])
    if row != checkpoint or spec["opponent"] != "B-line" or config.env.finite_horizon:
        raise ValueError("Pinned checkpoint and cohort metadata disagree")
    archive = {}
    with artifacts["flat_seed0_b-line.jsonl"].open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["seed"] in archive:
                raise ValueError("Duplicate archived episode")
            archive[record["seed"]] = record
    mapping_path = root / "runs/cage2_blue_learning_seed0_20261123/action_mapping.json"
    mapping_hash = file_sha256(mapping_path)
    previous = json.loads(artifacts["results.json"].read_text())
    if previous["action_mapping_sha256"] != mapping_hash:
        raise ValueError("Blue action mapping differs from the validated pilot")
    mapping = json.loads(mapping_path.read_text())
    adapter = CAGE2Replay(config, archive, provenance["scenario_sha256"], mapping)
    env = Cage2BlueEnv(config.env, torch.device("cpu"))
    try:
        adapter._check_env(env)
        model = DreamerAdapter.from_payload(payload, env.obs_space, env.action_dim, load_critic=False)
    finally:
        env.close()
    del payload
    weights = {name: module_sha256(getattr(model, name)) for name in ("world", "actor")}
    outcomes = dict(plan_revision=PLAN_REVISION, checkpoint=checkpoint, pinned_artifacts=spec["artifact_sha256"],
                    target=spec["target"], seed_splits=groups, acceptance=spec["acceptance"],
                    independent_pairs=spec["independent_pairs"], cohorts={}, test_opened=False,
                    episodes_replayed=0, native_steps=0, study_status="incomplete")

    def save():
        write_json(out / "results.json", outcomes)

    def finish(name):
        outcomes["study_status"] = name
        outcomes["frozen_module_hashes"] = weights
        outcomes["archive_action_mapping_sha256"] = mapping_hash
        if file_sha256(path) != checkpoint["sha256"] or any(module_sha256(getattr(model, k)) != v for k, v in weights.items()):
            raise ValueError("Frozen checkpoint or module state changed")
        save()
        render(out, outcomes)
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > maximum["output_mib"] * 2**20:
            raise RuntimeError("Study output exceeded pinned maximum")
        return True

    def collect(fold):
        episodes = []
        for seed in groups[fold]:
            if should_stop():
                save()
                render(out, outcomes)
                return None
            episodes.extend(adapter.collect([seed]))
            outcomes["episodes_replayed"] += 1
            outcomes["native_steps"] += len(episodes[-1].actions)
        truth = {ep.episode_id: adapter.labels(ep.episode_id) for ep in episodes}
        if not all(value.valid.all() for value in truth.values()):
            raise ValueError("Offline session labels are incomplete")
        host = tuple(config.world.cage2_host_order).index(spec["target"]["host"])
        truth_rows = {seed: labelled.values[:, host] for seed, labelled in truth.items()}
        frames, excluded = eligible_frames(episodes, truth_rows, spec["target"])
        pairs, matchable = exact_contrasts(frames)
        features = representations(model, episodes, pairs)
        controls = check_controls(features) if features is not None else {
            name: dict(feature_ties=None, score_max_abs=None) for name in CONTROL_NAMES}
        assert_controls(controls)
        outcomes["cohorts"][fold] = dict(
            coverage=coverage(frames, matchable, pairs, excluded, len(episodes)), controls=controls,
            archived_greedy_reward_component_boundary_parity=True,
            matching_rows=[dict(negative_episode=a.episode, positive_episode=b.episode, elapsed_time=a.time,
                                negative_age=a.age, positive_age=b.age) for a, b in pairs])
        np.savez_compressed(out / f"{fold}_aligned.npz", episode_ids=[ep.episode_id for ep in episodes],
                            observations=np.stack([ep.observations.numpy() for ep in episodes]),
                            actions=np.stack([ep.actions.numpy() for ep in episodes]),
                            rewards=np.stack([ep.rewards.numpy() for ep in episodes]),
                            offline_host_active=np.stack([truth_rows[ep.episode_id] for ep in episodes]))
        save()
        return pairs, features

    save()
    development = {}
    for fold in ("fit", "validation"):
        cohort = collect(fold)
        if cohort is None:
            return False
        development[fold] = cohort
    if any(len(development[fold][0]) < spec["independent_pairs"][f"{fold}_min"] for fold in development):
        return finish("insufficient_development_common_support")
    models, tuning = compare_fit_predictors(development["fit"][1], development["validation"][1], spec["probe"]["penalties"])
    for fold, (pairs, features) in development.items():
        controls = outcomes["cohorts"][fold]["controls"]
        outcomes["cohorts"][fold]["probes"] = readout_metrics(pairs, features, models, controls)
        assert_controls(controls)
    outcomes["tuning"] = tuning
    if outcomes["cohorts"]["validation"]["probes"]["recurrent"]["correct_fraction"] < spec["probe"]["validation_min_correct_fraction"]:
        return finish("development_history_recovery_gate_failed")
    lock = out / "locked_probes.json"
    write_json(lock, dict(tuning=tuning, models={name: probe.to_record() for name, probe in models.items()}))
    outcomes["probe_lock_sha256"] = file_sha256(lock)
    if should_stop():
        save()
        render(out, outcomes)
        return False
    outcomes["test_opened"] = True
    save()
    cohort = collect("test")
    if cohort is None:
        return False
    pairs, features = cohort
    if len(pairs) < spec["independent_pairs"]["test_min"]:
        if pairs:
            # Verify the controls with the already locked models. Descriptive
            # primary scoring is withheld when independent support fails.
            controls = outcomes["cohorts"]["test"]["controls"]
            for name in CONTROL_NAMES:
                scores = models[name].predict(features[name])
                maximum = float(np.abs(scores[0::2] - scores[1::2]).max())
                controls[name].update(score_max_abs=maximum, score_ties=maximum <= TIE_TOLERANCE)
            assert_controls(controls)
        return finish("insufficient_test_common_support")
    controls = outcomes["cohorts"]["test"]["controls"]
    metrics = readout_metrics(pairs, features, models, controls)
    assert_controls(controls)
    outcomes["cohorts"]["test"]["probes"] = metrics
    primary = metrics["recurrent"]
    passed = primary["correct_fraction"] >= spec["acceptance"]["test_min_correct_fraction"] and primary["exact_ci95"][0] > .5
    return finish("bounded_history_recovery_supported" if passed else "history_recovery_acceptance_not_met")
