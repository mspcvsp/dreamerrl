⭐ Dreamer‑V3 KL Invariants Cheat Sheet
These are the invariants that keep the RSSM stable, prevent latent collapse, and catch 99% of world‑model failures early.

They apply to:

Code
KL_dyn  = KL[ sg(q) || p ]
KL_rep  = KL[ q || sg(p) ]
KL_total = KL_dyn + KL_rep
All invariants are enforced after aggregation, i.e., KL has shape (B,).

1. Finite KL
Invariant:
KL must contain no NaN, no Inf.

Why:
Exploding logits, invalid probabilities, or broken gradients propagate silently unless caught here.

Check:

python
assert torch.isfinite(kl).all()
2. Non‑negative KL
Invariant:
KL ≥ 0 (within numerical tolerance).

Why:
Negative KL indicates invalid probability distributions or log‑softmax instability.

Check:

python
assert (kl >= -1e-6).all()
3. No KL explosion
Invariant:
KL must not exceed a configured maximum (typically 50–100).

Why:
Large KL destroys training signal, destabilizes the RSSM, and causes gradient spikes.

Check:

python
assert kl.mean() <= max_kl
4. No KL collapse (optional)
Invariant:
KL must not collapse to exactly zero unless explicitly allowed.

Why:
Zero KL means:

posterior collapse

dead latent factors

encoder ignoring observations

prior and posterior becoming identical

This is catastrophic for world‑model learning.

Check:

python
if require_nonzero:
    assert kl.mean() != 0
Note:
Tests and smoke runs should disable this (require_nonzero=False) because uniform priors/posteriors naturally produce KL=0.

5. Free bits applied after KL computation
Invariant:
Free bits clamp KL per batch element, not per factor.

Why:
Prevents the model from collapsing KL to zero while still allowing small KL values.

Check:

python
kl = torch.maximum(kl, free_bits)
6. KL_total = KL_dyn + KL_rep
Invariant:
Total KL is the sum of the two directional KLs.

Why:
This matches Dreamer‑V3’s structured KL geometry and ensures symmetric gradient flow.

Check:

python
kl_total = kl_dyn + kl_rep
⭐ Shape Summary (V3‑correct)
Quantity	Shape	Notes
q_probs, p_probs	(B, K, C)	factored categorical
categorical_kl	(B,)	aggregated over K and C
kl_dyn, kl_rep	(B,)	after aggregation
kl_total	(B,)	sum of dyn + rep


⭐ Where invariants belong
✔ Inside structured_kl
This is the only correct place.

✖ Not inside categorical_kl
That function must remain a pure mathematical primitive.

✖ Not inside observe_step
WorldModel should only attach KL values, not validate them.

⭐ When to disable invariants
Disable require_nonzero only for:

smoke tests

functional tests

zero‑input tests

initialization

debugging runs

Enable it for:

training

evaluation

long‑horizon rollouts
