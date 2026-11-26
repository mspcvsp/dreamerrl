"""Audit replay coverage using its actual lengths and truncated gradient graph."""
from collections import Counter

import torch


def replay_coverage(replay, *, delayed_at=None):
    """Count eligible start positions and context coverage by episode length.

    delayed_at is a zero-based decision-frame index or a callback taking an
    episode. When omitted, use the final stored row as the delayed endpoint.
    """
    seq, burn = replay["seq_len"], replay["burn_in"]
    lengths, starts, covered, both, detached, first_missing = (Counter() for _ in range(6))
    nonzero, valid_firsts = 0, 0
    for episode in replay["episodes"]:
        length = len(episode["obs"])
        target = delayed_at(episode) if callable(delayed_at) else delayed_at
        target = length - 1 if target is None else target
        if not 0 <= target < length:
            raise ValueError("Delayed endpoint lies outside episode")
        width = min(seq, length)
        lengths[length] += 1
        nonzero += int(torch.count_nonzero(episode["reward"]))
        reset = bool(episode["is_first"][0])
        valid_firsts += int(reset)
        for start in range(length - width + 1):
            starts[length] += 1
            if not start <= target < start + width:
                continue
            covered[length] += 1
            if reset and start == 0:
                both[length] += 1
            elif reset and start <= burn:
                detached[length] += 1
            else:
                first_missing[length] += 1
    return dict(seq_len=seq, burn_in=burn, episode_lengths=dict(lengths), reset_rows_observed=valid_firsts,
                eligible_starts=dict(starts), delayed_endpoint_supervised=dict(covered),
                reset_and_endpoint_jointly_supervised=dict(both), reset_available_only_through_detached_burn_in=dict(detached),
                endpoint_windows_without_valid_reset=dict(first_missing), nonzero_reward_rows=nonzero,
                unit="eligible episode/start combinations",
                gradient_note="Burn-in runs under no_grad. Joint coverage permits temporal gradient connectivity; it does not measure gradient strength.")


def prefix_gradient_norms(readout, supervised, *, burn_in=None):
    """Probe a single scalar readout's actual gradient to its observed prefix.

    Pass the tensors used in the evaluated graph, with axes (batch,time,...)
    and requires_grad enabled. This consumes the diagnostic graph without
    accumulating parameter gradients. A missing derivative indicates no graph
    path. A connected zero only describes local sensitivity at the chosen input.
    """
    prefixes = {"supervised": supervised}
    if burn_in is not None:
        prefixes["burn_in"] = burn_in
    if readout.ndim != 0 or not all(p.requires_grad and p.ndim >= 2 for p in prefixes.values()):
        raise ValueError("Supply a scalar readout and differentiable batch/time prefixes")
    derivatives = torch.autograd.grad(readout, tuple(prefixes.values()), allow_unused=True)
    result = {}
    for name, derivative in zip(prefixes, derivatives, strict=True):
        norm = None if derivative is None else derivative.reshape(*derivative.shape[:2], -1).norm(dim=-1)
        result[name] = dict(connected=derivative is not None, l2_by_frame=None if norm is None else norm.tolist())
    return result
