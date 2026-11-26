"""Frozen CAGE2 posterior probes, episode-level scoring and cautious latent diagnostics."""

import argparse
import csv
import json
import random
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from host_experiment_support import ROOT, sha256
from latent_probe import episode_split, metrics, paired_episode_interval, ridge, spectral_rank
from learner_cycle import state_hash
from learning_curve import write

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
from dreamerrl.training.trainer import DreamerTrainer

CONFIG = ROOT / "environments/cage2/latent_analysis_config.json"


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_audit(config, checkpoint):
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint fingerprint differs from completed pilot")
    summary = json.loads((checkpoint.parent / "final_summary.json").read_text())
    manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    assert summary["checkpoint_sha256"] == config["checkpoint_sha256"]
    assert summary["update"] == 16899 and summary["env_steps"] == 507450
    assert abs(summary["mean_return"] - (-28.64)) < 1e-8
    actual = subprocess.check_output(["git", "rev-parse", config["source_tag"] + "^{}"], cwd=ROOT, text=True).strip()
    assert actual == config["source_commit"] == summary["source_commit"] == manifest["source_commit"]
    assert (
        subprocess.check_output(["git", "rev-parse", config["source_tag"]], cwd=ROOT, text=True).strip()
        == manifest["tag_object"]
    )
    vendor = ROOT / "environments/cage2/vendor/cage-challenge-2"
    assert (
        subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
        == manifest["dependencies"]["cyborg_revision"]
    )
    assert not subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain"], text=True).strip()
    for path, expected in manifest["source_fingerprints"].items():
        if path.startswith("dreamerrl/") or path == "uv.lock":
            assert sha256(ROOT / path) == expected, "Frozen-model source/dependencies differ: " + path
    return summary, manifest


@torch.inference_mode()
def collect(trainer, config, run):
    fields = {
        key: []
        for key in (
            "h",
            "z",
            "posterior_probs",
            "prior_probs",
            "observation",
            "timestep",
            "episode",
            "action",
            "reward",
        )
    }
    episodes = []
    started = time.perf_counter()
    for episode in range(config["episodes"]):
        seed = config["simulator_seed_start"] + episode
        policy_seed = seed + config["policy_seed_offset"]
        random.seed(policy_seed)
        np.random.seed(policy_seed)
        torch.manual_seed(policy_seed)
        env = Cage2BlueEnv(replace(trainer.cfg.env, seed=seed), trainer.device)
        initial = env.reset(seed=seed)
        state = trainer.world.init_state(1)
        action = torch.zeros(1, trainer.action_dim, device=trainer.device)
        observed = initial["state"]
        rewards = []
        try:
            for step in range(config["horizon"]):
                result = trainer.world.observe_step(
                    state, observed, action, is_first=torch.tensor([step == 0], device=trainer.device)
                )
                state = result["post"]
                fields["h"].append(state.h[0].cpu().numpy().copy())
                fields["z"].append(state.z[0].cpu().numpy().astype(np.uint8))
                fields["posterior_probs"].append(result["post_stats"]["probs"][0].cpu().numpy().copy())
                fields["prior_probs"].append(result["prior_stats"]["probs"][0].cpu().numpy().copy())
                fields["observation"].append(observed[0].cpu().numpy().copy())
                fields["timestep"].append(step)
                fields["episode"].append(episode)
                chosen = int(trainer.actor(state.h, state.z).argmax(-1).item())
                output = env.step(torch.tensor([chosen], device=trainer.device))
                reward = env.journal[-1]["reward"]
                fields["action"].append(chosen)
                fields["reward"].append(reward)
                rewards.append(reward)
                if step < config["horizon"] - 1:
                    assert not output["is_last"].item()
                else:
                    assert output["is_last"].item() and not output["is_terminal"].item()
                observed = output["state"]
                action = torch.nn.functional.one_hot(
                    torch.tensor([chosen], device=trainer.device), trainer.action_dim
                ).float()
            episodes.append(
                dict(
                    episode=episode,
                    simulator_seed=seed,
                    policy_seed=policy_seed,
                    steps=len(rewards),
                    episode_return=sum(rewards),
                )
            )
        finally:
            env.close()
        if episode == 0 or (episode + 1) % 20 == 0:
            print(
                "COLLECT",
                episode + 1,
                "/",
                config["episodes"],
                "seconds",
                round(time.perf_counter() - started, 1),
                flush=True,
            )
            write(
                run / "progress.json",
                dict(
                    phase="collect",
                    episodes=episode + 1,
                    total_episodes=config["episodes"],
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
    data = {key: np.asarray(value) for key, value in fields.items()}
    reward = data["reward"].reshape(config["episodes"], config["horizon"])
    data["target"] = np.cumsum(reward[:, ::-1], axis=1)[:, ::-1].reshape(-1)
    np.savez_compressed(run / "frozen_features.npz", **data)
    write(run / "episodes.json", episodes)
    return data


def diagnose(data, split, run):
    output = {}
    utilization = []
    for name, selected in split.items():
        q = data["posterior_probs"][selected].astype(float)
        p = data["prior_probs"][selected].astype(float)
        z = data["z"][selected]
        usage = z.mean(0)
        entropy = -(q * np.log(np.maximum(q, 1e-12))).sum(-1)
        marginal = -(usage * np.log(np.maximum(usage, 1e-12))).sum(-1)
        mean_q = q.mean(0)
        soft_marginal = -(mean_q * np.log(np.maximum(mean_q, 1e-12))).sum(-1)
        kl = (q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(p, 1e-12)))).sum(-1)
        h = data["h"][selected]
        variance = h.var(0)
        output[name] = dict(
            posterior_entropy_mean_nats=float(entropy.mean()),
            maximum_factor_entropy_nats=float(np.log(q.shape[-1])),
            posterior_prior_kl_mean_per_factor=float(kl.mean()),
            posterior_prior_kl_mean_per_sample=float(kl.sum(-1).mean()),
            categories_used_per_factor=(usage > 0).sum(-1).tolist(),
            hard_usage_effective_categories=np.exp(marginal).tolist(),
            constant_mode_factors=int(((usage > 0).sum(-1) == 1).sum()),
            posterior_entropy_per_factor=entropy.mean(0).tolist(),
            soft_posterior_marginal_entropy=soft_marginal.tolist(),
            posterior_variation_jensen_gap=(soft_marginal - entropy.mean(0)).tolist(),
            h_variance=variance.tolist(),
            h_near_constant_units=int((variance < 1e-8).sum()),
            h_spectrum=spectral_rank(h),
        )
        if name == "fit":
            for factor, row in enumerate(usage):
                for category, value in enumerate(row):
                    utilization.append(dict(factor=factor, category=category, frequency=float(value)))
    means = np.stack([data["h"][split["fit"] & (data["timestep"] == t)].mean(0) for t in range(30)])
    output["time_residualized_h_test"] = spectral_rank(
        data["h"][split["test"]] - means[data["timestep"][split["test"]]]
    )
    output["interpretation"] = (
        "Hard-mode utilization is policy/sample dependent; low usage is not sufficient evidence of posterior collapse. "
        "Compare conditional entropy, prior KL, state variance and time-controlled probe results."
    )
    write(run / "latent_diagnostics.json", output)
    csv_write(run / "latent_utilization.csv", utilization)
    return output


def plots(data, split, models, diagnostics, run, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = split["test"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    names = ["time_lookup", "observation", "h", "z", "hz", "hz+observation+time"]
    actual = data["target"][test]
    for axis, name in zip(axes.flat, names):
        pred = models[name]["predictions"][test]
        axis.scatter(actual, pred, s=5, alpha=0.18)
        lo, hi = float(actual.min()), float(actual.max())
        axis.plot([lo, hi], [lo, hi], color="black", linewidth=0.7)
        axis.set(title=name, xlabel="Actual remaining return", ylabel="Predicted remaining return")
    figure.tight_layout()
    figure.savefig(run / "predicted_vs_actual.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    pca_metadata = {}
    representative = np.unique(data["episode"][test])[: config["pca_trajectory_count"]]
    for axis, name in zip(axes, ("h", "hz")):
        values = (
            data["h"] if name == "h" else np.concatenate([data["h"], data["z"].reshape(len(data["h"]), -1)], axis=1)
        )
        model = models[name]
        coordinates = ((values - model["mean"]) / model["scale"]) @ model["pca_vectors"]
        spectrum = model["eigenvalues"]
        pca_metadata[name] = dict(
            standardization="fitting-only",
            fitting_episodes=np.unique(data["episode"][split["fit"]]).tolist(),
            explained_variance_fraction=(spectrum[-2:][::-1] / max(float(spectrum.sum()), 1e-12)).tolist(),
            representative_test_episodes=representative.tolist(),
        )
        for episode in representative:
            selected = data["episode"] == episode
            axis.plot(coordinates[selected, 0], coordinates[selected, 1], alpha=0.5, linewidth=0.7)
            axis.scatter(
                coordinates[selected, 0], coordinates[selected, 1], c=data["timestep"][selected], s=5, vmin=0, vmax=29
            )
        axis.set(title=name + "; PCA fitted only on fitting episodes", xlabel="PC1", ylabel="PC2")
    figure.tight_layout()
    figure.savefig(run / "pca_trajectories.png", dpi=150)
    write(run / "pca_metadata.json", pca_metadata)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(data["z"][split["fit"]].mean(0), aspect="auto", vmin=0, vmax=1)
    axes[0].set(xlabel="Category", ylabel="Categorical factor", title="Fitting-set hard-mode usage")
    eigen = np.asarray(diagnostics["fit"]["h_spectrum"]["eigenvalues"])[::-1]
    axes[1].plot(np.arange(1, len(eigen) + 1), eigen)
    axes[1].set(
        xlabel="Eigenvalue rank", ylabel="Variance", title="Deterministic state covariance spectrum", yscale="log"
    )
    figure.tight_layout()
    figure.savefig(run / "latent_utilization.png", dpi=150)
    plt.close(figure)


def analyze(data, config, run):
    split, ids = episode_split(data["episode"], config["split_seed"])
    write(run / "episode_splits.json", {key: values.tolist() for key, values in ids.items()})
    time_fields = np.eye(30)[data["timestep"]]
    h = data["h"]
    z = data["z"].reshape(len(h), -1)
    obs = data["observation"]
    families = {
        "time_linear": data["timestep"][:, None],
        "observation": obs,
        "h": h,
        "z": z,
        "hz": np.concatenate([h, z], 1),
        "observation+time": np.concatenate([obs, time_fields], 1),
        "h+time": np.concatenate([h, time_fields], 1),
        "h+observation+time": np.concatenate([h, obs, time_fields], 1),
        "z+time": np.concatenate([z, time_fields], 1),
        "hz+time": np.concatenate([h, z, time_fields], 1),
        "hz+observation+time": np.concatenate([h, z, obs, time_fields], 1),
    }
    target = data["target"]
    by_time = np.array([target[split["fit"] & (data["timestep"] == t)].mean() for t in range(30)])
    models: dict[str, dict[str, Any]] = {"time_lookup": dict(predictions=by_time[data["timestep"]], alpha=None)}
    rows = []
    coef = {}
    validation = {}
    for name, features in families.items():
        print("FIT", name, "dimensions", features.shape[1], flush=True)
        write(run / "progress.json", dict(phase="fit", family=name, dimensions=features.shape[1]))
        model = ridge(features, target, split, config["ridge_alphas"])
        models[name] = model
        validation[name] = model["validation_path"]
        for field in ("mean", "scale", "coefficients", "ymean", "alpha"):
            coef[name.replace("+", "_") + "_" + field] = model[field]
    for name, model in models.items():
        row = dict(
            features=name, alpha=model["alpha"], dimensions=30 if name == "time_lookup" else families[name].shape[1]
        )
        for part, selected in split.items():
            values = metrics(target[selected], model["predictions"][selected])
            row.update({part + "_" + key: value for key, value in values.items()})
            row[part + "_episodes"] = len(ids[part])
        residual = metrics(
            target[split["test"]] - by_time[data["timestep"][split["test"]]],
            model["predictions"][split["test"]] - by_time[data["timestep"][split["test"]]],
        )
        row["time_residual_test_r2"] = residual["r2"]
        rows.append(row)
    csv_write(run / "linear_probe_comparison.csv", rows)
    write(run / "linear_probe_comparison.json", rows)
    write(run / "validation_paths.json", validation)
    np.savez_compressed(run / "linear_probe_coefficients.npz", **coef)
    comparisons = {}
    for first, reference in [
        ("h+time", "time_lookup"),
        ("z+time", "time_lookup"),
        ("hz+time", "time_lookup"),
        ("hz+observation+time", "observation+time"),
        ("h+observation+time", "observation+time"),
    ]:
        a, b = models[first]["predictions"][split["test"]], models[reference]["predictions"][split["test"]]
        comparisons[first + " vs " + reference] = paired_episode_interval(
            target[split["test"]],
            a,
            b,
            data["episode"][split["test"]],
            config["bootstrap_seed"],
            config["bootstrap_draws"],
        )
    for key, interval in comparisons.items():
        first, reference = key.split(" vs ")
        a = metrics(target[split["test"]], models[first]["predictions"][split["test"]])
        b = metrics(target[split["test"]], models[reference]["predictions"][split["test"]])
        r2_a, r2_b = a["r2"], b["r2"]
        interval["delta_r2"] = None if r2_a is None or r2_b is None else r2_a - r2_b
        interval["delta_mae"] = a["mae"] - b["mae"]
    write(run / "incremental_information.json", comparisons)
    diagnostics = diagnose(data, split, run)
    plots(data, split, models, diagnostics, run, config)
    lookup = {row["features"]: row for row in rows}
    extra = comparisons["hz+observation+time vs observation+time"]
    interval = extra["delta_r2_ci95"]
    evidence = (
        "The combined latent adds detectable linear predictive information beyond observation and timestep controls."
        if interval and interval[0] > 0
        else "This analysis does not establish incremental linear predictive information "
        "beyond observation and timestep controls."
    )
    lines = [
        "Frozen CAGE2 host-aware RSSM linear probes",
        "",
        "Target: undiscounted remaining return from the current decision through step 30. "
        "Features are pre-action, terminal frame excluded.",
        "Fitting/validation/test split by complete episode; standardization uses fitting data only; "
        "validation selects ridge alpha.",
        "",
        "Features | dimensions | test R2 | test MAE | test RMSE | test samples",
    ]
    for name in [
        "time_linear",
        "time_lookup",
        "observation",
        "h",
        "z",
        "hz",
        "observation+time",
        "h+time",
        "z+time",
        "hz+time",
        "hz+observation+time",
        "h+observation+time",
    ]:
        row = lookup[name]
        lines.append(
            f"{name} | {row['dimensions']} | {row['test_r2']} | {row['test_mae']:.4f} | "
            f"{row['test_rmse']:.4f} | {row['test_samples']}"
        )
    lines += [
        "",
        evidence,
        "Weak probe scores indicate difficulty decoding features linearly, "
        "not proof that the representation lacks information.",
        f"Paired episode-bootstrap 95% interval for incremental R2: {interval}.",
        f"Hard-mode constant factors (fitting set): {diagnostics['fit']['constant_mode_factors']}/32.",
        f"h effective rank (test): {diagnostics['test']['h_spectrum']['effective_rank']:.3f}; "
        f"after time residualization: {diagnostics['time_residualized_h_test']['effective_rank']:.3f}.",
        f"Mean posterior entropy: {diagnostics['test']['posterior_entropy_mean_nats']:.4f} nats/factor; "
        "maximum log(30).",
        "Neither low categorical usage nor a small rank alone proves posterior collapse; "
        "interpret with prior KL and conditional probes.",
        "These are correlations under one frozen checkpoint and its greedy policy, "
        "not causal evidence or independent-training-run uncertainty.",
        "Latent increments can reflect nonlinear recoding as well as history; "
        "prediction alone does not prove causal memory.",
        "PCA is descriptive; representative test episodes were chosen by fixed IDs, "
        "not by their returns. No UMAP dependency was added.",
    ]
    (run / "final_report.txt").write_text("\n".join(lines) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    run = args.run_dir.resolve()
    assert run.is_dir() and not (run / "frozen_features.npz").exists()
    checkpoint = ROOT / config["checkpoint"]
    summary, pilot_manifest = source_audit(config, checkpoint)
    write(run / "resolved_analysis_config.json", config)
    manifest = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        pilot_source_tag=config["source_tag"],
        pilot_source_commit=config["source_commit"],
        pilot_evaluation_return=summary["mean_return"],
        analysis_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        analysis_source_sha256={
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                Path(__file__),
                CONFIG,
                ROOT / "environments/cage2/latent_probe.py",
                ROOT / "environments/cage2/latent_analysis_job.py",
                ROOT / "scripts/run_latent_analysis.sh",
            ]
        },
        uv_lock_sha256=sha256(ROOT / "uv.lock"),
        cyborg_revision=pilot_manifest["dependencies"]["cyborg_revision"],
        features=(
            "h_t (200), z_t (32x30 flattened), combined concat (1160); no rewards/actions/successors in probe inputs"
        ),
    )
    (run / "analysis_source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
    write(run / "manifest.json", manifest)
    torch.set_num_threads(2)
    started = time.perf_counter()
    trainer = DreamerTrainer.from_checkpoint(checkpoint, cuda=True, log_dir=run / "loader_logs")
    assert trainer.completed_updates == 16899 and trainer.total_env_steps == 507450
    assert trainer.cfg.world.obs_encoder == "cage2_host" and trainer.cfg.env.max_episode_steps == 30
    assert trainer.device.type == "cuda"
    print(
        "CHECKPOINT_VERIFIED",
        config["checkpoint_sha256"],
        "parameters frozen; collecting pre-action states",
        flush=True,
    )
    before = {name: state_hash(getattr(trainer, name)) for name in ("world", "actor", "critic", "slow_critic")}
    replay_before = (trainer.replay.size, len(trainer.replay.episodes), trainer.completed_updates)
    for name in before:
        module = getattr(trainer, name)
        module.eval()
        module.requires_grad_(False)
    trainer.world.posterior.deterministic_latent_for_tests = True
    assert not any(p.requires_grad for name in before for p in getattr(trainer, name).parameters())
    write(
        run / "freeze_start.json",
        dict(
            all_parameters_frozen=True,
            model_hashes=before,
            completed_optimizer_updates=trainer.completed_updates,
            replay_size=trainer.replay.size,
        ),
    )
    print(
        "CHECKPOINT_VERIFIED",
        config["checkpoint_sha256"],
        "parameters frozen; collecting pre-action states",
        flush=True,
    )
    try:
        data = collect(trainer, config, run)
        after = {name: state_hash(getattr(trainer, name)) for name in before}
        assert before == after and replay_before == (
            trainer.replay.size,
            len(trainer.replay.episodes),
            trainer.completed_updates,
        )
        write(
            run / "frozen_model_verification.json",
            dict(
                all_parameters_frozen=True,
                weights_and_buffers_unchanged=True,
                replay_unchanged=True,
                optimizer_updates_added=0,
                model_hashes=after,
            ),
        )
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
    del trainer
    torch.cuda.empty_cache()
    rows = analyze(data, config, run)
    assert sha256(checkpoint) == config["checkpoint_sha256"]
    write(
        run / "complete.json",
        dict(
            success=True,
            episodes=config["episodes"],
            transitions=len(data["target"]),
            elapsed_seconds=time.perf_counter() - started,
            checkpoint_unchanged=True,
            probe_count=len(rows),
        ),
    )
    print("ANALYSIS_COMPLETE", run, flush=True)


if __name__ == "__main__":
    main()
