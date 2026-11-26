"""Fit-only NumPy K-means on frozen RSSM states; no model loading or training."""

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "environments/cage2/latent_clustering_config.json"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def distances(x, centers):
    return np.maximum((x * x).sum(1)[:, None] + (centers * centers).sum(1) - 2 * x @ centers.T, 0)


def kmeans(x, k, seed, iterations=60):
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))].copy()]
    for _ in range(1, k):
        nearest = distances(x, np.asarray(centers)).min(1)
        index = rng.choice(len(x), p=nearest / nearest.sum()) if nearest.sum() else rng.integers(len(x))
        centers.append(x[index].copy())
    centers = np.asarray(centers)
    previous = None
    for _ in range(iterations):
        d = distances(x, centers)
        labels = d.argmin(1)
        if previous is not None and np.array_equal(previous, labels):
            break
        previous = labels
        centers = np.stack(
            [
                x[labels == cluster].mean(0) if np.any(labels == cluster) else x[d.min(1).argmax()]
                for cluster in range(k)
            ]
        )
    return centers


def ari(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)

    def choose(values):
        return float((values * (values - 1) / 2).sum())

    total = len(a) * (len(a) - 1) / 2
    observed = choose(table)
    expected = choose(table.sum(0)) * choose(table.sum(1)) / max(total, 1)
    maximum = (choose(table.sum(0)) + choose(table.sum(1))) / 2
    return (observed - expected) / (maximum - expected) if maximum > expected else 1.0


def silhouette(x, labels):
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(x):
        return None
    d = np.sqrt(distances(x, x))
    np.fill_diagonal(d, 0)
    values = []
    for row, group in enumerate(labels):
        same = labels == group
        if same.sum() == 1:
            values.append(0.0)
            continue
        inside = d[row, same].sum() / (same.sum() - 1)
        outside = min(d[row, labels == other].mean() for other in unique if other != group)
        values.append((outside - inside) / max(inside, outside, 1e-15))
    return float(np.mean(values))


def db_index(x, labels):
    groups = np.unique(labels)
    if len(groups) < 2:
        return None
    centers = np.stack([x[labels == group].mean(0) for group in groups])
    spread = np.array(
        [np.linalg.norm(x[labels == group] - center, axis=1).mean() for group, center in zip(groups, centers)]
    )
    d = np.sqrt(distances(centers, centers))
    np.fill_diagonal(d, np.inf)
    return float(((spread[:, None] + spread) / np.maximum(d, 1e-15)).max(1).mean())


def nmi(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    joint = table / table.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    expected = pa[:, None] * pb
    valid = joint > 0
    information = (joint[valid] * np.log(joint[valid] / expected[valid])).sum()

    def entropy(p):
        return -(p[p > 0] * np.log(p[p > 0])).sum()

    denominator = np.sqrt(entropy(pa) * entropy(pb))
    return float(information / denominator) if denominator else 0.0


def prepare(h, timestep, fit, variance_target):
    mean, scale = h[fit].mean(0), h[fit].std(0)
    scale = np.where(scale > 1e-8, scale, 1)
    standardized = (h - mean) / scale
    eigenvalues, vectors = np.linalg.eigh(standardized[fit].T @ standardized[fit] / fit.sum())
    eigenvalues, vectors = np.maximum(eigenvalues[::-1], 0), vectors[:, ::-1]
    cumulative = np.cumsum(eigenvalues) / max(eigenvalues.sum(), 1e-15)
    components = min(len(eigenvalues), int(np.searchsorted(cumulative, variance_target)) + 1)
    time_means = np.stack([h[fit & (timestep == step)].mean(0) for step in range(30)])
    residual = h - time_means[timestep]
    residual_mean, residual_scale = residual[fit].mean(0), residual[fit].std(0)
    residual_scale = np.where(residual_scale > 1e-8, residual_scale, 1)
    features = {
        "h": standardized,
        "pca_h": standardized @ vectors[:, :components],
        "residual_h": (residual - residual_mean) / residual_scale,
    }
    parameters = dict(
        mean=mean,
        scale=scale,
        pca_vectors=vectors,
        pca_eigenvalues=eigenvalues,
        time_means=time_means,
        residual_mean=residual_mean,
        residual_scale=residual_scale,
    )
    return features, parameters


def persistence(labels, episodes, timesteps):
    order = np.lexsort((timesteps, episodes))
    labels, episodes = labels[order], episodes[order]
    adjacent = episodes[1:] == episodes[:-1]
    stays = labels[1:] == labels[:-1]
    streaks = []
    length = 1
    for index in range(1, len(labels)):
        if episodes[index] == episodes[index - 1] and labels[index] == labels[index - 1]:
            length += 1
        else:
            streaks.append(length)
            length = 1
    streaks.append(length)
    return dict(
        stay_probability=float(stays[adjacent].mean()),
        mean_streak=float(np.mean(streaks)),
        median_streak=float(np.median(streaks)),
        adjacent_pairs=int(adjacent.sum()),
    )


def characterize(x, labels, episode, timestep, target, k, seed):
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(100):
        shuffled = labels.copy()
        for step in np.unique(timestep):
            mask = timestep == step
            shuffled[mask] = rng.permutation(shuffled[mask])
        null.append(persistence(shuffled, episode, timestep)["stay_probability"])
    observed = persistence(labels, episode, timestep)
    observed["time_conditioned_shuffle_stay_mean"] = float(np.mean(null))
    observed["time_conditioned_shuffle_stay_ci95"] = np.quantile(null, [0.025, 0.975]).tolist()
    observed["excess_stay"] = observed["stay_probability"] - float(np.mean(null))
    clusters = []
    for group in range(k):
        mask = labels == group
        if not mask.any():
            clusters.append(dict(cluster=group, samples=0, episodes=0))
            continue
        clusters.append(
            dict(
                cluster=group,
                samples=int(mask.sum()),
                occupancy=float(mask.mean()),
                episodes=int(len(np.unique(episode[mask]))),
                timestep_histogram=np.bincount(timestep[mask], minlength=30).tolist(),
                timestep_mean=float(timestep[mask].mean()),
                return_quantiles=np.quantile(target[mask], [0, 0.25, 0.5, 0.75, 1]).tolist(),
                return_mean=float(target[mask].mean()),
                return_std=float(target[mask].std()),
                within_cluster_variance=float(x[mask].var(0).sum()),
            )
        )
    return dict(clusters=clusters, persistence=observed, timestep_nmi=nmi(labels, timestep))


def analyze(run):
    config = json.loads(CONFIG.read_text())
    source = ROOT / config["dataset_directory"]
    manifest = json.loads((source / "manifest.json").read_text())
    complete = json.loads((source / "complete.json").read_text())
    frozen = json.loads((source / "frozen_model_verification.json").read_text())
    assert complete["success"] and complete["checkpoint_unchanged"]
    assert frozen["weights_and_buffers_unchanged"] and frozen["optimizer_updates_added"] == 0
    assert manifest["checkpoint_sha256"] == config["checkpoint_sha256"] == sha(Path(manifest["checkpoint"]))
    assert manifest["pilot_source_commit"] == config["source_commit"]
    assert (
        subprocess.check_output(["git", "rev-parse", config["source_tag"] + "^{}"], text=True).strip()
        == config["source_commit"]
    )
    assert sha(ROOT / "uv.lock") == manifest["uv_lock_sha256"]
    splits = json.loads((source / "episode_splits.json").read_text())
    data = np.load(source / "frozen_features.npz", allow_pickle=False)
    h = data["h"].astype(np.float64)
    episode, timestep, target = data["episode"], data["timestep"], data["target"]
    masks = {key: np.isin(episode, ids) for key, ids in splits.items()}
    assert [len(splits[key]) for key in ("fit", "validation", "test")] == [216, 72, 72]
    assert not any(set(splits[a]) & set(splits[b]) for a, b in itertools.combinations(splits, 2))
    assert h.shape == (10800, 200) and np.isfinite(h).all()
    for ep in np.unique(episode):
        index = np.flatnonzero(episode == ep)
        assert np.array_equal(timestep[index], np.arange(30))
        np.testing.assert_allclose(target[index], np.cumsum(data["reward"][index][::-1])[::-1], atol=1e-5)
    provenance = dict(
        dataset_manifest=manifest,
        checkpoint_sha256=config["checkpoint_sha256"],
        input_sha256={
            name: sha(source / name)
            for name in ("frozen_features.npz", "episode_splits.json", "manifest.json", "complete.json")
        },
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        analysis_source_sha256={
            str(path.relative_to(ROOT)): sha(path)
            for path in (Path(__file__), CONFIG, ROOT / "environments/cage2/latent_clustering_job.py")
        },
    )
    assert all(
        subprocess.check_output(["git", "show", manifest["analysis_revision"] + ":" + path])
        == (ROOT / path).read_bytes()
        for path in manifest["analysis_source_sha256"]
    )
    write(run / "manifest.json", provenance)
    (run / "source.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], text=True))
    write(run / "resolved_config.json", config)
    write(run / "episode_splits.json", splits)
    print("PROVENANCE_VERIFIED dataset 10800 states; checkpoint unchanged; no Dreamer model loaded", flush=True)
    features, parameters = prepare(h, timestep, masks["fit"], config["pca_variance_target"])
    np.savez_compressed(run / "preprocessing.npz", **parameters)
    rng = np.random.default_rng(config["analysis_seed"])
    samples = {
        key: rng.choice(np.flatnonzero(mask), min(config["silhouette_sample_size"], mask.sum()), replace=False)
        for key, mask in masks.items()
    }
    write(run / "silhouette_sample_rows.json", {key: rows.tolist() for key, rows in samples.items()})
    rows, selected, details, assignments = [], {}, {}, {}
    for family, x in features.items():
        candidates = []
        for k in config["k_values"]:
            fits = [
                kmeans(x[masks["fit"]], k, seed, config["max_iterations"]) for seed in config["initialization_seeds"]
            ]
            inertia = [float(distances(x[masks["fit"]], centers).min(1).sum()) for centers in fits]
            best = int(np.argmin(inertia))
            labels = [distances(x, centers).argmin(1) for centers in fits]
            row: dict[str, Any] = dict(
                features=family,
                dimensions=x.shape[1],
                k=k,
                initialization_seed=config["initialization_seeds"][best],
                fit_inertia=inertia[best],
            )
            for split in ("fit", "validation", "test"):
                index = samples[split]
                row[split + "_silhouette"] = silhouette(x[index], labels[best][index])
                row[split + "_db"] = db_index(x[masks[split]], labels[best][masks[split]])
                row[split + "_occupied_clusters"] = int(len(np.unique(labels[best][masks[split]])))
                row[split + "_init_ari_mean"] = float(
                    np.mean(
                        [
                            ari(labels[a][masks[split]], labels[b][masks[split]])
                            for a, b in itertools.combinations(range(len(fits)), 2)
                        ]
                    )
                )
            row["test_min_cluster_size"] = int(np.bincount(labels[best][masks["test"]], minlength=k).min())
            rows.append(row)
            candidates.append((row, fits[best], labels[best]))
            details[f"{family}_k{k}"] = dict(
                initialization_fit_inertia=inertia,
                initialization_seeds=config["initialization_seeds"],
                **{
                    split: characterize(
                        x[masks[split]],
                        labels[best][masks[split]],
                        episode[masks[split]],
                        timestep[masks[split]],
                        target[masks[split]],
                        k,
                        config["analysis_seed"],
                    )
                    for split in ("fit", "validation", "test")
                },
            )
            print(
                f"FIT {family} K={k} validation silhouette={row['validation_silhouette']} "
                f"test={row['test_silhouette']}",
                flush=True,
            )
        chosen = max(candidates, key=lambda item: (item[0]["validation_silhouette"] or -1, -item[0]["k"]))
        row, centers, labels = chosen
        k = row["k"]
        bootstrap = []
        for seed in range(config["episode_bootstrap_refits"]):
            ids = np.random.default_rng(config["analysis_seed"] + seed).choice(
                splits["fit"], len(splits["fit"]), replace=True
            )
            index = np.concatenate([np.flatnonzero(episode == ep) for ep in ids])
            boot = kmeans(x[index], k, config["initialization_seeds"][0], config["max_iterations"])
            prediction = distances(x, boot).argmin(1)
            bootstrap.append(
                dict(
                    test_ari=ari(labels[masks["test"]], prediction[masks["test"]]),
                    per_test_episode_ari=[
                        ari(labels[episode == ep], prediction[episode == ep]) for ep in splits["test"]
                    ],
                )
            )
        selected[family] = dict(
            quality=row, episode_bootstrap_stability=bootstrap, characterization=details[f"{family}_k{k}"]["test"]
        )
        assignments[family] = labels
        np.savez_compressed(run / f"{family}_model.npz", centers=centers, labels=labels)
    with (run / "cluster_quality.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(run / "cluster_quality.json", rows)
    write(run / "cluster_characterization.json", details)
    write(run / "selected_clusters.json", selected)
    plot(run, features["h"] @ parameters["pca_vectors"][:, :2], assignments, episode, timestep, splits)
    lines = [
        "Frozen RSSM clustering (post-hoc descriptive analysis)",
        "Preprocessing, PCA, timestep means and K-means fit on 216 fitting episodes only.",
        "K selected by validation silhouette (72 episodes), not test outcomes. Test: 72 episodes /2160 states.",
        "Silhouette uses fixed 600-row samples; DB/occupancy use all split rows.",
        "features K test_silhouette test_DB init_ARI bootstrap_ARI time_NMI excess_stay",
    ]
    for family, result in selected.items():
        q, c = result["quality"], result["characterization"]
        boot = np.mean([value["test_ari"] for value in result["episode_bootstrap_stability"]])
        lines.append(
            f"{family} {q['k']} {q['test_silhouette']:.4f} {q['test_db']:.4f} "
            f"{q['test_init_ari_mean']:.4f} {boot:.4f} {c['timestep_nmi']:.4f} "
            f"{c['persistence']['excess_stay']:.4f}"
        )
    raw = selected["h"]["characterization"]["timestep_nmi"]
    residual = selected["residual_h"]["characterization"]["timestep_nmi"]
    residual_quality = selected["residual_h"]["quality"]
    residual_boot = np.mean([r["test_ari"] for r in selected["residual_h"]["episode_bootstrap_stability"]])
    residual_excess = selected["residual_h"]["characterization"]["persistence"]["excess_stay"]
    lines.append(
        "Residual clusters retain reproducible temporal structure beyond the mean timestep trajectory; "
        "semantic meaning is not established."
        if residual_quality["test_silhouette"] > 0.1 and residual_boot > 0.6 and residual_excess > 0.05
        else "Evidence for stable clusters beyond episode progression is limited under this residual control."
    )
    lines.extend(
        [
            f"Timestep NMI raw={raw:.4f}, residual={residual:.4f}.",
            "Compare residual quality/stability and time-conditioned persistence to judge structure "
            "beyond progression.",
            "Residualizing the mean per timestep does not remove timestep-dependent variance or "
            "nonlinear clock information.",
            "Return distributions are descriptive and confounded by time; no causal or simulator-state "
            "interpretation is established.",
            "ARI bootstrap refits hold preprocessing fixed; variability conditional on this agent and "
            "policy, not independent training seeds.",
            "K-means imposes Euclidean spherical partitions; scores do not establish natural clusters "
            "or semantic defense states.",
            "Test seeds were already used for linear probes; this is exploratory reuse, not fresh "
            "confirmatory evaluation.",
        ]
    )
    (run / "final_report.txt").write_text("\n".join(lines) + "\n")
    assert sha(Path(manifest["checkpoint"])) == config["checkpoint_sha256"]
    assert all(sha(source / name) == value for name, value in provenance["input_sha256"].items())
    write(
        run / "complete.json", dict(success=True, checkpoint_unchanged=True, dataset_unchanged=True, models=len(rows))
    )
    print("CLUSTERING_COMPLETE", flush=True)


def plot(run, projected, assignments, episode, timestep, splits):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    test = np.isin(episode, splits["test"])
    for column, (family, labels) in enumerate(assignments.items()):
        axes[0, column].scatter(*projected[test].T, c=labels[test], cmap="tab10", s=5, alpha=0.5)
        axes[0, column].set_title(family + " held-out clusters")
        for ep in sorted(splits["test"])[:8]:
            index = np.flatnonzero(episode == ep)
            index = index[np.argsort(timestep[index])]
            axes[1, column].plot(*projected[index].T, alpha=0.4, linewidth=0.6)
            axes[1, column].scatter(*projected[index].T, c=labels[index], cmap="tab10", s=9)
        axes[1, column].set_title("Eight fixed-ID test trajectories")
        for axis in axes[:, column]:
            axis.set_xlabel("Standardized h: fit PCA 1")
            axis.set_ylabel("Fit PCA 2")
    figure.tight_layout()
    figure.savefig(run / "pca_clusters.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    analyze(parser.parse_args().run_dir.resolve())
