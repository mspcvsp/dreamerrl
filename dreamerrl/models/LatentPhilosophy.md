# Latent contracts in this implementation

The state is a deterministic vector `h` and a factored one-hot tensor `z`.
Legacy configuration names put `z` at `(B, num_classes, stoch_size)`; the final
axis is the number of categories per factor. This naming is retained for existing
configurations. Sequence latents add a time axis: `(B, L, K, C)`.

The deterministic transition consumes previous `h`, previous `z`, and the action.
Observe computes that transition before its prior and posterior distributions.
The posterior also receives encoded observations. Episode starts clear previous
state and action for the affected batch rows.

Use categorical sampling with uniform mixing and straight-through probability
gradients; see [sampling](StochasticLatentSampling.md). This estimator does not by
itself prevent posterior collapse. Dynamics and representation KL use separate
stop-gradient directions, sum over factors, and apply the free-nats floor before
averaging examples. Current defaults are 1 nat and loss weights 1.0 / 0.1.

Keep factor/category axes intact for categorical probabilities, samples, and KL.
Flattening a latent for an MLP is valid and is used in the recurrent transition
and all prediction/actor heads. Each factor has its own projection coordinates;
permuting factors can change predictions even when category counts are identical. Categorical latent
capacity alone does not prove a network can learn to use that capacity.

The [bounded fidelity report](../../docs/dreamerv3_fidelity.md) records the pinned
reference, checked contracts, and remaining architecture/optimization differences.
