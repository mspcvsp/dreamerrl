# Dreamer‑V3: Stochastic Latent Sampling (Intuition & Rationale)

The RSSM state is (h, z):

    h — deterministic recurrent state (transition model)

    z — discrete stochastic latent capturing unpredictable dynamics

Dreamer‑V3 uses factored categorical latents:

    z = [z¹, z², …, z^K], each zᵢ ∈ {1..num_classes}

Benefits of factored discrete latents:

    higher expressiveness than a single categorical

    stable per‑factor KL (prevents collapse)

    better long‑horizon credit assignment

    more stable imagination rollouts

# Why add Gumbel noise?

During training we need a differentiable approximation to categorical sampling.
The Gumbel‑Softmax trick provides this:

y = softmax((logits + gumbel_noise) / temperature)

y is a soft sample that is differentiable with respect to logits.
Why divide by temperature?

Temperature τ controls sample sharpness:

    τ < 1 → sharper, more discrete‑like samples

    τ > 1 → smoother, more exploratory samples

Lower τ encourages confident, stable latents during training.

# Why return a hard one‑hot z?

The RSSM forward pass uses a discrete latent:

z_hard = one_hot(argmax(y))

This ensures:

    stable imagination rollouts

    compatibility with prior/posterior KL

    clean inputs for decoder, reward head, and actor

# Straight‑through estimator:

z = z_hard + (y - y.detach())

keeps the forward pass discrete while allowing gradients to flow.

# Why deterministic_latent_for_tests?

For invariants (CPU/GPU determinism, batch‑size invariance, pure‑function behavior), we disable sampling entirely:

z = one_hot(argmax(probs))

This yields bit‑exact reproducibility for unit tests.
