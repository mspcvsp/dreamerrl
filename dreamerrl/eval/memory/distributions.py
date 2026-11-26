"""Categorical distribution distances for diagnostic inference."""
import torch


def entropy(probs):
    q = probs.double()
    return -torch.special.xlogy(q, q).sum(-1)


def categorical_kl(q, p):
    q, p = q.double(), p.double()
    return (torch.special.xlogy(q, q) - torch.special.xlogy(q, p)).sum(-1)


def categorical_tv(q, other):
    """Exact total variation along the categories of a single factor."""
    return (q.double() - other.double()).abs().sum(-1) / 2


def categorical_js(q, other):
    """Per-factor Jensen–Shannon divergence in nats, with equal weights."""
    q, other = q.double(), other.double()
    center = (q + other) / 2
    return ((categorical_kl(q, center) + categorical_kl(other, center)) / 2).clamp_min(0)


def single_categorical_bayes_accuracy(q, other):
    """Exact equal-prior accuracy for one categorical observation."""
    return (1 + categorical_tv(q, other)) / 2


def product_single_draw_bayes_upper_bound(q, other, *, method="pinsker"):
    """Per-pair equal-prior bound for a single draw of the whole latent.

    Inputs have trailing (factor, category) axes and are normalized within each
    factor. Factors are independent conditional on one of the two histories.
    The decoder may know the pair/history context, but receives one draw and
    must distinguish the two equiprobable conditions. The bound does not cover
    repeated draws, correlated factors, or a decoder that observes h as well.

    Pinsker uses the smaller directional product KL and preserves the original
    diagnostic convention. The affinity bound uses the product Bhattacharyya
    coefficient and remains finite with disjoint support.
    """
    if q.ndim < 2 or q.shape != other.shape or q.numel() == 0:
        raise ValueError("Paired categorical factors must align and be nonempty")
    q, other = q.double(), other.double()
    if method == "pinsker":
        directional = torch.minimum(categorical_kl(q, other).sum(-1), categorical_kl(other, q).sum(-1))
        tv_upper = (directional.clamp_min(0) / 2).sqrt().clamp_max(1)
    elif method == "affinity":
        factor_affinity = 1 - (q.sqrt() - other.sqrt()).square().sum(-1) / 2
        affinity = factor_affinity.clamp(0, 1).prod(-1)
        tv_upper = (1 - affinity.square()).clamp_min(0).sqrt()
    else:
        raise ValueError(f"Unknown Bayes bound method: {method}")
    return (1 + tv_upper) / 2


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
    """Report distribution, argmax and realized-state differences separately."""
    upper = product_single_draw_bayes_upper_bound(q, other)
    js = categorical_js(q, other)
    tv = categorical_tv(q, other)
    equal = q.argmax(-1).eq(other.argmax(-1))
    result = dict(js_nats_by_factor=js.reshape(-1, js.shape[-1]).mean(0).tolist(),
                  tv_by_factor=tv.reshape(-1, tv.shape[-1]).mean(0).tolist(),
                  argmax_pair_equal_fraction=float(equal.all(-1).double().mean()),
                  mean_single_draw_bayes_accuracy_upper_bound=float(upper.mean()))
    if z is not None and other_z is not None:
        result["realized_hard_pair_equal_fraction"] = float(z.eq(other_z).flatten(-2).all(-1).double().mean())
    return result
