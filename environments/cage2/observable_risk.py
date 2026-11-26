"""Offline forecast of defender-visible evidence burden, not hidden compromise or damage."""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from latent_clustering import sha, write
from latent_probe import metrics, paired_episode_interval, ridge

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "environments/cage2/observable_risk_config.json"


def observable_risk(observations):
    """Verified BlueTable bit pairs. Unknown is reported separately, never inferred safe."""
    if observations.shape[-1] != 52 or not np.isin(observations, [0, 1]).all():
        raise ValueError("Expected native binary BlueTable 13x4 observations")
    hosts = observations.reshape(*observations.shape[:-1], 13, 4)
    if np.any((hosts[..., 0] == 0) & (hosts[..., 1] == 1)):
        raise ValueError("Invalid Activity code 01")
    exploit = (hosts[..., 0] == 1) & (hosts[..., 1] == 1)
    compromised = hosts[..., 3] == 1
    unknown = (hosts[..., 2] == 1) & (hosts[..., 3] == 0)
    return (exploit | compromised).mean(-1), unknown.mean(-1)


def future_targets(risk, episode, timestep, horizon):
    target = np.full(len(risk), np.nan)
    for ep in np.unique(episode):
        index = np.flatnonzero(episode == ep)
        index = index[np.argsort(timestep[index])]
        assert np.array_equal(timestep[index], np.arange(30))
        for step in range(30 - horizon):
            target[index[step]] = risk[index[step + 1 : step + 1 + horizon]].mean()
    return target, np.isfinite(target)


def calibration(actual, predicted) -> dict[str, Any]:
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (predicted >= lower) & (predicted <= upper if index == 9 else predicted < upper)
        if mask.any():
            bins.append(
                dict(
                    lower=lower,
                    upper=upper,
                    samples=int(mask.sum()),
                    predicted_mean=float(predicted[mask].mean()),
                    actual_mean=float(actual[mask].mean()),
                )
            )
    variance = float(np.var(predicted))
    slope = (
        float(np.mean((predicted - predicted.mean()) * (actual - actual.mean())) / variance)
        if variance > 1e-12
        else None
    )
    return dict(
        mean_bias=float(np.mean(predicted - actual)),
        slope=slope,
        intercept=float(actual.mean() - slope * predicted.mean()) if slope is not None else None,
        bins=bins,
        interpretation="Calibration of predicted mean risk burden, not event probability",
    )


def confidence(actual, predicted, episodes, seed, draws):
    ids = np.unique(episodes)
    stats = np.array(
        [
            [
                sum(episodes == ep),
                actual[episodes == ep].sum(),
                np.square(actual[episodes == ep]).sum(),
                np.square(predicted[episodes == ep] - actual[episodes == ep]).sum(),
                np.abs(predicted[episodes == ep] - actual[episodes == ep]).sum(),
            ]
            for ep in ids
        ]
    )
    totals = stats[np.random.default_rng(seed).integers(len(ids), size=(draws, len(ids)))].sum(1)
    denominator = totals[:, 2] - totals[:, 1] ** 2 / totals[:, 0]
    valid = denominator > 1e-12
    return dict(
        r2_ci95=np.quantile(1 - totals[valid, 3] / denominator[valid], [0.025, 0.975]).tolist()
        if valid.any()
        else None,
        mae_ci95=np.quantile(totals[:, 4] / totals[:, 0], [0.025, 0.975]).tolist(),
        rmse_ci95=np.quantile(np.sqrt(totals[:, 3] / totals[:, 0]), [0.025, 0.975]).tolist(),
    )


def main(run):
    config = json.loads(CONFIG.read_text())
    source = ROOT / config["dataset_directory"]
    provenance = json.loads((source / "manifest.json").read_text())
    completed = json.loads((source / "complete.json").read_text())
    freeze = json.loads((source / "frozen_model_verification.json").read_text())
    assert completed["success"] and completed["checkpoint_unchanged"]
    assert freeze["weights_and_buffers_unchanged"] and freeze["optimizer_updates_added"] == 0
    assert sha(Path(provenance["checkpoint"])) == config["checkpoint_sha256"] == provenance["checkpoint_sha256"]
    assert provenance["pilot_source_commit"] == config["pilot_commit"]
    assert (
        subprocess.check_output(["git", "rev-parse", config["pilot_tag"] + "^{}"], text=True).strip()
        == config["pilot_commit"]
    )
    assert sha(ROOT / "uv.lock") == provenance["uv_lock_sha256"]
    # Bind to the dataset hashes independently recorded by the preceding clustering job.
    prior = json.loads((ROOT / "runs/cage2_latent_clustering_host_pilot_20261124/manifest.json").read_text())
    assert all(sha(source / name) == value for name, value in prior["input_sha256"].items())
    blue = ROOT / "environments/cage2/vendor/cage-challenge-2"
    assert (
        subprocess.check_output(["git", "-C", str(blue), "rev-parse", "HEAD"], text=True).strip()
        == provenance["cyborg_revision"]
    )
    assert not subprocess.check_output(["git", "-C", str(blue), "status", "--porcelain"], text=True).strip()
    splits = json.loads((source / "episode_splits.json").read_text())
    assert not any(
        set(splits[a]) & set(splits[b]) for a, b in [("fit", "validation"), ("fit", "test"), ("validation", "test")]
    )
    data = np.load(source / "frozen_features.npz", allow_pickle=False)
    assert data["h"].shape == (10800, 200)
    episode, timestep = data["episode"], data["timestep"]
    risk, unknown = observable_risk(data["observation"])
    (run / "source.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], text=True))
    write(
        run / "manifest.json",
        dict(
            original_provenance=provenance,
            input_sha256=prior["input_sha256"],
            source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    Path(__file__),
                    CONFIG,
                    ROOT / "environments/cage2/latent_probe.py",
                    ROOT / "environments/cage2/latent_clustering.py",
                    ROOT / "environments/cage2/observable_risk_job.py",
                    ROOT / "scripts/run_observable_risk.sh",
                )
            },
        ),
    )
    write(run / "resolved_config.json", config)
    write(run / "episode_splits.json", splits)
    write(
        run / "target_summary.json",
        dict(
            meaning=config["target"],
            observed_fraction_mean=float(risk.mean()),
            unknown_fraction_mean=float(unknown.mean()),
            nonzero_observations=int((risk > 0).sum()),
            source_observations=len(risk),
            no_model_loaded=True,
            no_optimizer_steps=True,
        ),
    )
    print("PROVENANCE_VERIFIED: frozen 10800-row dataset; checkpoint unchanged; no model loaded", flush=True)
    rows, comparisons, calibrations, predictions = [], {}, {}, {}
    h, z = data["h"], data["z"].reshape(len(risk), -1)
    time = np.eye(30)[timestep]
    obs = data["observation"]
    control = np.column_stack((obs, time, risk, unknown))
    families = dict(
        time=timestep[:, None],
        time_lookup=time,
        observation=obs,
        h=h,
        z=z,
        hz=np.column_stack((h, z)),
        observation_time=np.column_stack((obs, time)),
        observable_control=control,
        h_control=np.column_stack((h, control)),
        z_control=np.column_stack((z, control)),
        hz_control=np.column_stack((h, z, control)),
    )
    for horizon in config["horizons"]:
        target, valid = future_targets(risk, episode, timestep, horizon)
        masks = {key: np.isin(episode[valid], ids) for key, ids in splits.items()}
        y = target[valid]
        pred = {"persistence": risk[valid], "fit_mean": np.full(len(y), y[masks["fit"]].mean())}
        for name, features in families.items():
            result = ridge(features[valid], y, masks, config["ridge_alphas"])
            pred[name] = np.clip(result["predictions"], 0, 1)
            np.savez_compressed(
                run / f"h{horizon}_{name}_model.npz",
                mean=result["mean"],
                scale=result["scale"],
                coefficients=result["coefficients"],
                intercept=result["ymean"],
                alpha=result["alpha"],
            )
            write(run / f"h{horizon}_{name}_validation.json", result["validation_path"])
            print(f"FIT H={horizon} features={name} alpha={result['alpha']}", flush=True)
        test = masks["test"]
        for name, prediction in pred.items():
            row = dict(
                horizon=horizon, features=name, **metrics(y[test], prediction[test]), episodes=len(splits["test"])
            )
            row.update(
                confidence(
                    y[test], prediction[test], episode[valid][test], config["bootstrap_seed"], config["bootstrap_draws"]
                )
            )
            for split in ("fit", "validation"):
                row[split + "_metrics"] = metrics(y[masks[split]], prediction[masks[split]])
            rows.append(row)
            calibrations[f"H{horizon}_{name}"] = calibration(y[test], prediction[test])
        for name in ("h_control", "z_control", "hz_control"):
            comparisons[f"H{horizon}_{name}_vs_observable_control"] = paired_episode_interval(
                y[test],
                pred[name][test],
                pred["observable_control"][test],
                episode[valid][test],
                config["bootstrap_seed"],
                config["bootstrap_draws"],
            )
        predictions[f"h{horizon}_target"] = y[test]
        predictions[f"h{horizon}_episode"] = episode[valid][test]
        for name, prediction in pred.items():
            predictions[f"h{horizon}_{name}"] = prediction[test]
    write(run / "comparison.json", rows)
    with (run / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(run / "calibration.json", calibrations)
    write(run / "paired_information.json", comparisons)
    np.savez_compressed(run / "heldout_predictions.npz", **predictions)
    plot(run, predictions, calibrations, config["primary_horizon"])
    primary = comparisons[f"H{config['primary_horizon']}_hz_control_vs_observable_control"]
    interval = primary["delta_r2_ci95"]
    lines = [
        "Offline observable defensive evidence-risk forecast",
        config["target"],
        "horizon features test_R2 MAE RMSE R2_CI95",
        *[f"{r['horizon']} {r['features']} {r['r2']} {r['mae']:.5f} {r['rmse']:.5f} {r['r2_ci95']}" for r in rows],
        f"Primary H=5 combined-vs-observable/time/persistence-feature control delta_R2 CI95: {interval}",
        "Latents add linear predictive information under this protocol."
        if interval and interval[0] > 0
        else "No established incremental linear information over the strong observable/time control.",
        "Do not add a training objective yet: forecastability does not establish that its gradient improves defense.",
        "This risk is remembered defender-visible evidence, not true compromise probability, "
        "hidden attacker progress, or damage.",
        "Unknown is not healthy; misses, stale beliefs, scan-only threats and unknown compromises are not measured.",
        "Future observations appear only in targets. Inputs contain no future "
        "actions/rewards/observations or hidden state.",
        "Forecast is under the frozen greedy policy, not action-conditioned counterfactual "
        "risk or a policy-invariant value.",
        "Calibration describes a fractional evidence burden, not calibrated host compromise probabilities.",
        "Preprocessing fits on fitting episodes; alpha chosen by validation MSE (before "
        "clipping); test scored after clipping.",
        "Intervals resample complete test episodes; not uncertainty across independent trained agents.",
        "Existing test episodes were already inspected by prior analyses; development "
        "evidence, not fresh confirmation.",
        "No model loaded or modified, no auxiliary training loss implemented.",
    ]
    (run / "final_report.txt").write_text("\n".join(lines) + "\n")
    assert sha(Path(provenance["checkpoint"])) == config["checkpoint_sha256"]
    assert all(sha(source / name) == value for name, value in prior["input_sha256"].items())
    write(
        run / "complete.json",
        dict(success=True, checkpoint_unchanged=True, dataset_unchanged=True, no_model_updates=True),
    )
    print("OBSERVABLE_RISK_COMPLETE", flush=True)


def plot(run, predictions, calibrations, horizon):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("persistence", "observable_control", "hz_control"):
        actual, pred = predictions[f"h{horizon}_target"], predictions[f"h{horizon}_{name}"]
        axes[0].scatter(actual, pred, s=3, alpha=0.15, label=name)
        bins = calibrations[f"H{horizon}_{name}"]["bins"]
        axes[1].plot([b["predicted_mean"] for b in bins], [b["actual_mean"] for b in bins], marker="o", label=name)
    for axis in axes:
        axis.plot([0, 1], [0, 1], "k--")
        axis.legend()
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    axes[0].set(xlabel="Observed future evidence burden", ylabel="Predicted", title=f"Held-out H={horizon}")
    axes[1].set(xlabel="Predicted bin mean", ylabel="Observed bin mean", title="Regression calibration")
    fig.tight_layout()
    fig.savefig(run / "risk_predictions_calibration.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    main(parser.parse_args().run_dir.resolve())
