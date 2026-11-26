"""Grouped probe fitting, uncertainty, and paired categorical diagnostics."""
import numpy as np
import torch

from dreamerrl.eval.memory.legacy_ridge import episode_split


def grouped_split(group_ids, seed):
    split, groups = episode_split(np.asarray(group_ids), seed)
    if not all(len(ids) for ids in groups.values()):
        raise ValueError("At least one independent group is required per fold")
    return split, groups


def _kernels(x, fit, nonlinear, bandwidths):
    x = np.asarray(x, dtype=np.float64).copy()
    x -= x[fit].mean(0)
    scale = np.sqrt(np.mean(np.sum(x[fit] ** 2, axis=1)))
    x /= scale if scale else 1
    gram = x @ x[fit].T
    if not nonlinear:
        return [(1.0, gram)]
    d2 = np.maximum(np.sum(x * x, axis=1)[:, None] + np.sum(x[fit] ** 2, axis=1) - 2 * gram, 0)
    fit_d2 = d2[fit]
    positive = fit_d2[fit_d2 > 1e-12]
    median = float(np.median(positive)) if positive.size else 1.
    return [(m, np.exp(-d2 / (2 * median * m))) for m in bandwidths]


def binary_probe(x, targets, split, *, nonlinear=False, balanced=False,
                 penalties=(.001, .03, 1.), bandwidths=(.5, 1., 2.)):
    """Fit one declared draw per row; evaluate every draw. Return fixed scores.

    x is (rows, draws, features). The masks must come from a group-disjoint
    split. The balanced variant weights fit and validation by class prevalence.
    """
    x, targets = np.asarray(x), np.asarray(targets, dtype=bool)
    n, repeats, width = x.shape
    if targets.shape != (n,) or not np.all(sum(np.asarray(mask, dtype=int) for mask in split.values()) == 1):
        raise ValueError("Labels and disjoint masks must cover probe rows")
    fit = np.flatnonzero(split["fit"]) * repeats
    val = np.flatnonzero(split["validation"])
    y = np.where(targets[fit // repeats], 1., -1.)
    val_y = targets[val]
    if len(np.unique(y)) != 2 or len(np.unique(val_y)) != 2:
        raise ValueError("Fit and validation both require positive and negative labels")
    best = None
    for bandwidth, kernel in _kernels(x.reshape(-1, width), fit, nonlinear, bandwidths):
        train = kernel[fit]
        if balanced:
            weights = np.where(y > 0, .5 / np.mean(y > 0), .5 / np.mean(y < 0))
            row_mean = kernel @ weights / len(fit)
            train_mean = weights @ train / len(fit)
            kernel = kernel - row_mean[:, None] - train_mean + weights @ train_mean / len(fit)
            diagonal = np.diag(1 / weights)
        else:
            kernel = kernel - kernel.mean(axis=1, keepdims=True) - train.mean(axis=0) + train.mean()
            diagonal = np.eye(len(fit))
        for penalty in reversed(penalties):
            # Preserve the historical solver order in the unweighted convention.
            inverse = np.linalg.solve(kernel[fit] + penalty * diagonal, np.eye(len(fit)))
            scores = (kernel @ (inverse @ y)).reshape(n, repeats)
            correct = (scores[val] >= 0) == val_y[:, None]
            accuracy = (correct[val_y].mean() + correct[~val_y].mean()) / 2 if balanced else correct.mean()
            if best is None or accuracy > best["validation_accuracy"]:
                best = dict(validation_accuracy=float(accuracy), penalty=penalty, bandwidth=bandwidth, scores=scores)
    return best


def bootstrap_mean(values, seed, draws=2000):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values):
        raise ValueError("Supply one value per independent group")
    resamples = np.random.default_rng(seed).integers(len(values), size=(draws, len(values)))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[resamples].mean(axis=1), [.025, .975]).tolist())


def paired_accuracy(scores, labels):
    """One score per pair, averaging any posterior draws before uncertainty."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.shape[:2] != labels.shape or labels.shape[1] != 2 or not np.all(labels.sum(1) == 1):
        raise ValueError("Each pair needs both labels")
    return ((scores >= 0) == labels[(...,) + (None,) * (scores.ndim - 2)]).reshape(len(labels), -1).mean(1)


def binary_metrics(scores, targets, groups, seed, draws=1000):
    """Balanced accuracy, tie-aware AP and episode-bootstrap accuracy."""
    scores, targets = np.asarray(scores), np.asarray(targets, dtype=bool)
    if scores.ndim != 2 or len(scores) != len(targets) or len(groups) != len(targets):
        raise ValueError("Expected aligned (row, draw) scores, labels and groups")
    correct = ((scores >= 0) == targets[:, None]).mean(1)
    ids = np.unique(groups)
    counts = np.asarray([[correct[(groups == g) & targets].sum(), ((groups == g) & targets).sum(),
                          correct[(groups == g) & ~targets].sum(), ((groups == g) & ~targets).sum()] for g in ids])
    total = counts.sum(0)
    result = dict(rows=len(targets), groups=len(ids), prevalence=float(targets.mean()), balanced_accuracy=None,
                  balanced_accuracy_ci95=None, average_precision=None)
    if not total[1] or not total[3]:
        return result
    result["balanced_accuracy"] = float((total[0] / total[1] + total[2] / total[3]) / 2)
    indices = np.random.default_rng(seed).integers(len(ids), size=(draws, len(ids)))
    bootstrap = counts[indices].sum(1)
    valid = (bootstrap[:, 1] > 0) & (bootstrap[:, 3] > 0)
    if valid.any():
        a, p, b, n = bootstrap[valid].T
        result["balanced_accuracy_ci95"] = np.quantile((a/p + b/n)/2, [.025, .975]).tolist()
    result["valid_class_bootstrap_draws"] = int(valid.sum())
    # Group ties at their shared threshold. Average AP across draw replicates.
    ap = []
    for draw in scores.T:
        order = np.argsort(-draw, kind="stable")
        endpoints = np.r_[np.flatnonzero(np.diff(draw[order]) != 0), len(order)-1]
        positives = targets[order].cumsum()[endpoints]
        ap.append(np.sum(np.diff(positives, prepend=0) * positives / (endpoints + 1)) / total[1])
    result["average_precision"] = float(np.mean(ap))
    return result


def entropy(probs):
    q = probs.double()
    return -torch.special.xlogy(q, q).sum(-1)


def categorical_kl(q, p):
    q, p = q.double(), p.double()
    return (torch.special.xlogy(q, q) - torch.special.xlogy(q, p)).sum(-1)


def posterior_summary(q, p, z, threshold=.01):
    """One row per state with trailing (factor, category) axes."""
    q, p, z = [x.reshape(-1, *x.shape[-2:]) for x in (q, p, z)]
    h = entropy(q).mean(0)
    factor_information = entropy(q.double().mean(0)) - h
    return dict(entropy_nats_by_factor=h.tolist(), kl_nats_by_factor=categorical_kl(q, p).mean(0).tolist(),
                information_nats_by_factor=factor_information.tolist(),
                active_factors=int((factor_information > threshold).sum()), active_threshold_nats=threshold,
                occupied_categories=z.ne(0).any(0).sum(-1).tolist())


def paired_categorical(q, other, z=None, other_z=None):
    if q.shape != other.shape:
        raise ValueError("Opposite distributions must align")
    q, other = q.double(), other.double()
    mean = (q + other)/2
    js = ((categorical_kl(q, mean) + categorical_kl(other, mean))/2).clamp_min(0)
    directional = torch.minimum(categorical_kl(q, other).sum(-1), categorical_kl(other, q).sum(-1))
    equal = q.argmax(-1).eq(other.argmax(-1))
    result = dict(js_nats_by_factor=js.reshape(-1, js.shape[-1]).mean(0).tolist(),
                  argmax_pair_equal_fraction=float(equal.all(-1).double().mean()),
                  mean_single_draw_bayes_accuracy_upper_bound=float((1 + (directional.clamp_min(0)/2).sqrt().clamp_max(1)).mean()/2))
    if z is not None and other_z is not None:
        result["realized_hard_pair_equal_fraction"] = float(z.eq(other_z).flatten(-2).all(-1).double().mean())
    return result


def l2_distance(left, right):
    return torch.linalg.vector_norm((left - right).double().flatten(1), dim=-1)
