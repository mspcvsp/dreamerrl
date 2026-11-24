"""Episode-aware NumPy ridge probing; no Dreamer parameters or optimizers are trained."""

from typing import TypedDict

import numpy as np


class ProbeMetrics(TypedDict):
    r2: float | None
    mae: float
    rmse: float
    samples: int


def metrics(actual, predicted) -> ProbeMetrics:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    return ProbeMetrics(
        r2=1 - float(np.square(error).sum()) / denominator if denominator > 1e-12 else None,
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.square(error).mean())),
        samples=len(actual),
    )


def episode_split(episodes, seed):
    ids = np.asarray(sorted(set(episodes.tolist())))
    order = np.random.default_rng(seed).permutation(ids)
    fit_end, val_end = int(0.6 * len(ids)), int(0.8 * len(ids))
    groups = dict(fit=order[:fit_end], validation=order[fit_end:val_end], test=order[val_end:])
    return {key: np.isin(episodes, values) for key, values in groups.items()}, groups


def ridge(features, targets, split, alphas):
    x = np.asarray(features, dtype=np.float64)
    mean, scale = x[split["fit"]].mean(0), x[split["fit"]].std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (x - mean) / scale
    training = standardized[split["fit"]]
    ymean = float(targets[split["fit"]].mean())
    covariance = training.T @ training / len(training)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0)
    response = eigenvectors.T @ (training.T @ (targets[split["fit"]] - ymean) / len(training))
    choices = []
    best = None
    for alpha in alphas:
        coefficients = eigenvectors @ (response / (eigenvalues + alpha))
        prediction = standardized[split["validation"]] @ coefficients + ymean
        loss = float(np.mean((prediction - targets[split["validation"]]) ** 2))
        choices.append(dict(alpha=alpha, validation_mse=loss))
        if best is None or loss < best[0]:
            best = (loss, float(alpha), coefficients)
    assert best is not None
    predictions = standardized @ best[2] + ymean
    return dict(
        alpha=best[1],
        predictions=predictions,
        mean=mean,
        scale=scale,
        coefficients=best[2],
        ymean=ymean,
        validation_path=choices,
        pca_vectors=eigenvectors[:, -2:][:, ::-1],
        eigenvalues=eigenvalues,
    )


def spectral_rank(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / max(1, len(values) - 1)), 0)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return dict(
            effective_rank=0.0,
            participation_ratio=0.0,
            variance_95_rank=0,
            total_variance=total,
            eigenvalues=eigenvalues.tolist(),
        )
    p = eigenvalues[eigenvalues > 0] / total
    return dict(
        effective_rank=float(np.exp(-np.sum(p * np.log(p)))),
        participation_ratio=float(total**2 / np.square(eigenvalues).sum()),
        variance_95_rank=int(np.searchsorted(np.cumsum(eigenvalues[::-1]) / total, 0.95) + 1),
        total_variance=total,
        eigenvalues=eigenvalues.tolist(),
    )


def paired_episode_interval(actual, first, reference, episode_ids, seed, draws=2000):
    ids = np.unique(episode_ids)
    sums = []
    for episode in ids:
        selected = episode_ids == episode
        y = actual[selected]
        sums.append(
            [
                len(y),
                y.sum(),
                np.square(y).sum(),
                np.square(first[selected] - y).sum(),
                np.square(reference[selected] - y).sum(),
                np.abs(first[selected] - y).sum(),
                np.abs(reference[selected] - y).sum(),
            ]
        )
    sums = np.asarray(sums)
    index = np.random.default_rng(seed).integers(len(ids), size=(draws, len(ids)))
    totals = sums[index].sum(1)
    denominator = totals[:, 2] - totals[:, 1] ** 2 / totals[:, 0]
    valid = denominator > 1e-12
    delta_r2 = (totals[valid, 4] - totals[valid, 3]) / denominator[valid]
    delta_mae = (totals[:, 5] - totals[:, 6]) / totals[:, 0]
    return dict(
        delta_r2_ci95=np.quantile(delta_r2, [0.025, 0.975]).tolist() if len(delta_r2) else None,
        delta_mae_ci95=np.quantile(delta_mae, [0.025, 0.975]).tolist(),
        episodes=len(ids),
        bootstrap_draws=draws,
    )
