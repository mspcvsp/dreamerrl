"""Offline return-history baselines from the existing CAGE2 feature archive."""
import json

import numpy as np

from dreamerrl.eval.memory.dreamer import file_sha256
from dreamerrl.eval.memory.history import AttentionEvidence, fit_history_probe, past_windows, project_fit_scaled
from dreamerrl.eval.memory.legacy_ridge import metrics, paired_episode_interval
from dreamerrl.eval.memory.presentation import MIST, NAVY, ORANGE, TEAL, plt, save_figure, style
from dreamerrl.eval.memory.reporting import write_json
from dreamerrl.eval.memory.statistics import grouped_split


def run(root, spec, out, should_stop):
    archive = root / spec["archive"]
    archive_manifest = archive / "resolved_analysis_config.json"
    config = json.loads(archive_manifest.read_text())
    archive_path = archive / "frozen_features.npz"
    fingerprint = file_sha256(archive_path)
    pilot_path = root / spec["pilot"]
    pilot = json.loads(pilot_path.read_text())
    if pilot["archive_parity"]["sha256"] != fingerprint:
        raise ValueError("History input differs from the independently validated archive")
    support = pilot["host_coverage"][pilot["selected_host"]]
    episode_counts = {k: len(v) for k, v in pilot["split"].items()}
    delta = pilot["hidden_session_paired_deltas"]["observation_time_hz - observation_time_fixed_hz"]
    evidence = AttentionEvidence(episode_counts["fit"], episode_counts["validation"], episode_counts["test"],
                                 min(support["test"]["positive_episodes"], support["test"]["negative_episodes"]) >= 10,
                                 tuple(delta["ci95"]) if delta["ci95"] is not None else None)
    if "transformer" in spec["kinds"] and not evidence.permits_fit:
        raise ValueError("The independent hidden-session pilot does not support transformer fitting")
    with np.load(archive_path, allow_pickle=False) as data:
        ids, time, target = (data[k] for k in ("episode", "timestep", "target"))
        split, groups = grouped_split(ids, config["split_seed"])
        expected = json.loads((archive / "episode_splits.json").read_text())
        if {k: values.tolist() for k, values in groups.items()} != expected:
            raise ValueError("History study must preserve episode-disjoint historical splits")
        features = dict(observation=data["observation"], posterior=data["posterior_probs"].reshape(len(ids), -1),
                        hz=np.column_stack((data["h"], data["z"].reshape(len(ids), -1))))
    test = split["test"]
    seed = spec["rng_seed"]
    time_lookup = np.array([target[split["fit"] & (time == t)].mean() for t in range(time.max() + 1)])[time]
    clock = time.astype(np.float32)[:, None] / time.max()
    result = dict(archive=str(archive_path), archive_sha256=fingerprint, pilot=str(pilot_path),
                  pilot_sha256=file_sha256(pilot_path), target="undiscounted return-to-go",
                  episode_counts={k: len(v) for k, v in groups.items()}, history_steps=spec["history_frames"],
                  time_lookup=metrics(target[test], time_lookup[test]),
                  unavailable_representations=["encoder"], transformer_fitted=False,
                  hidden_session_gate=dict(episode_counts=episode_counts, support=support["test"],
                                           history_increment=delta, permits_fit=evidence.permits_fit),
                  representations={})
    for label, frames in features.items():
        if should_stop():
            return False
        projected = project_fit_scaled(frames, split["fit"], spec["projected_features"], seed)
        windows, valid = past_windows(np.column_stack((projected, clock)), ids, time, spec["history_frames"])
        records, predictions = {}, {}
        for kind in spec["kinds"]:
            if should_stop():
                return False
            values = fit_history_probe(kind, windows, valid, target, split, seed=seed,
                                       budget=spec["parameter_budget"], learning_rates=spec["learning_rates"],
                                       epochs=spec["epochs"], attention_evidence=evidence)
            prediction = values.pop("predictions")[test]
            predictions[kind] = prediction
            records[kind] = dict(**values, **metrics(target[test], prediction))
            result["transformer_fitted"] |= kind == "transformer"
        comparisons = {}
        for kind in ("fixed_mlp", "gru", "transformer"):
            if kind in predictions and "current_mlp" in predictions:
                first, reference = predictions[kind], predictions["current_mlp"]
                paired = paired_episode_interval(target[test], first, reference, ids[test], seed)
                paired["delta_r2"] = metrics(target[test], first)["r2"] - metrics(target[test], reference)["r2"]
                comparisons[f"{kind}_minus_current_mlp"] = paired
        result["representations"][label] = dict(probes=records, paired_history_increment=comparisons)
        write_json(out / "results.json", result)
    render(out, result)
    return True


def render(out, result):
    style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {"linear": (NAVY, "o"), "current_mlp": (MIST, "D"), "fixed_mlp": (TEAL, "s"),
              "gru": (ORANGE, "^"), "transformer": (ORANGE, "*")}
    reps = list(result["representations"])
    for kind, (c, marker) in styles.items():
        values = [r["probes"].get(kind) for r in result["representations"].values()]
        where = [i for i, value in enumerate(values) if value]
        if where:
            ax.scatter(where, [values[i]["r2"] for i in where], color=c, marker=marker, label=kind.replace("_", " "))
    ax.axhline(result["time_lookup"]["r2"], ls="--", lw=1, c=NAVY, label="time lookup")
    ax.set(xticks=range(len(reps)), xticklabels=["Observation", "Posterior probabilities", "Combined RSSM"],
           ylabel="Episode-disjoint test R²")
    ax.legend(frameon=False)
    save_figure(fig, out / "history_baselines")
    kinds = list(dict.fromkeys(kind for row in result["representations"].values() for kind in row["probes"]))
    labels = [kind.replace("_", " ") + " R²" for kind in kinds]
    lines = ["# Archived CAGE2 temporal-history baselines", "",
             "Identical targets and episode splits were used across models. The target is archived return-to-go. "
             "The archive has no encoder embeddings and no offline hidden-session labels.",
             "", "| Representation | " + " | ".join(labels) + " |",
             "|---|" + "---:|" * len(kinds)]
    for name, row in result["representations"].items():
        probes = row["probes"]
        values = [f"{probes[kind]['r2']:.4f}" if kind in probes else "—" for kind in kinds]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += ["", "Paired R² change versus the current-state MLP, with a 95% episode-bootstrap interval.", ""]
    for name, row in result["representations"].items():
        for contrast, difference in row["paired_history_increment"].items():
            lines.append(f"- {name}, {contrast.replace('_', ' ')} {difference['delta_r2']:+.4f} "
                         f"{difference['delta_r2_ci95']}")
    lines += ["", f"The time-only lookup reached test R² {result['time_lookup']['r2']:.4f}.",
              "Projection, scaling, parameter cap and target are shared across architectures within representation. "
              "The same time feature was available to every probe. Confidence intervals resample test episodes "
              "with fitted probes fixed."]
    if not result["hidden_session_gate"]["permits_fit"]:
        lines += ["", "**The bounded hidden-session pilot did not clear the attention gate.** It had five held-out "
                  "episodes and its fixed-history comparison interval included zero."]
    lines += ["", "Return forecasting measures recoverability in archived histories. It leaves policy use, "
              "hidden compromise estimation, and the value of attention after the RSSM unresolved. The next "
              "comparison needs encoder features and a well-supported offline target with weaker observation and "
              "time baselines.", "", "![Temporal history baselines](history_baselines.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
