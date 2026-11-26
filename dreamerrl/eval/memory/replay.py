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
