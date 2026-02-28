"""
λ-return (time-major):

reward: (T, B)
value:  (T+1, B

G_t^λ blends TD(0) and Monte Carlo:

TD(0):        r_t + γ V_{t+1}
Monte Carlo:  r_t + γ r_{t+1} + γ² r_{t+2} + ...

λ mixes n-step returns with exponentially decaying weights:

G_t^λ = (1-λ)*G_t^{1-step}
        + λ(1-λ)*G_t^{2-step}
        + λ²(1-λ)*G_t^{3-step}
        + ...

λ = 0 → trust critic (low variance, high bias)
λ = 1 → trust rollout (high variance, low bias)

Time-major rollout:
------------------
t = 0      1      2      ...    T-1      T
|------|------|------|------|------|------|
s0     s1     s2     ...    s(T-1)  sT
r0     r1     r2     ...    r(T-1)

Values:
V(s0)  V(s1)  V(s2)  ...    V(s(T-1))  V(sT)
<----------- T+1 values ------------->

λ-return needs V(s_{t+1}) for every t, so value must be (T+1, B)
"""

from __future__ import annotations

import torch


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    discount: float,
    lam: float,
) -> torch.Tensor:
    """
    Time-major λ-return.

    Args:
        reward: (T, B) tensor of rewards.
        value:  (T+1, B) tensor of value predictions.
        discount: scalar discount factor γ.
        lam: scalar λ parameter.

    Returns:
        (T, B) tensor of λ-returns.
    """
    T, B = reward.shape
    ret = torch.zeros_like(reward)

    next_val = value[-1]  # (B,)
    for t in reversed(range(T)):
        delta = reward[t] + discount * next_val - value[t]
        next_val = value[t] + lam * delta
        ret[t] = next_val

    return ret
