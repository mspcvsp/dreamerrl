"""Fixed-history replay and interventions independent of simulator labels."""
import torch
import torch.nn.functional as F


def filter_trajectory(model, observations, actions, *, mode="argmax", rng=None, history=0, initial_z=None):
    """Filter aligned (batch,time,features) inputs; history=0 retains full history.

    With history=1 each frame is processed alone. Larger finite histories
    recompute that many frames. The first frame of a truncated prefix retains
    its incoming action and starts from zero carry.
    """
    n, length = observations.shape[:2]
    if actions.shape != (n, length) or history < 0 or (initial_z is not None and history):
        raise ValueError("Invalid fixed-history inputs")
    state = model.initial_state(n)
    zero = observations.new_zeros(n, model.action_dim)
    states, stages = [], []
    for t in range(length):
        start = max(0, t - history + 1) if history else t
        if history:
            state = model.initial_state(n)
        for k in range(start, t + 1):
            first = k == 0
            incoming = zero if first else F.one_hot(actions[:, k - 1].long(), model.action_dim).to(zero.dtype)
            state, features = model.filter_step(state, observations[:, k], incoming,
                torch.full((n,), first, device=observations.device), mode=mode, rng=rng)
        if t == 0 and initial_z is not None:
            state = model.with_z(state, initial_z)
            features = {**features, "z": state.z, "hz": torch.cat((state.h, state.z.flatten(1)), -1)}
        states.append(state)
        stages.append(features)
    return states, {key: torch.stack([stage[key] for stage in stages], dim=1) for key in stages[0]}
