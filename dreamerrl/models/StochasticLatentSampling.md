# Stochastic latent sampling

The RSSM uses factored categorical probabilities with a 1% uniform mixture:

```python
probs = 0.99 * softmax(logits) + 0.01 / categories
hard = one_hot(Categorical(probs).sample())
z = hard + (probs - probs.detach())
```

Forward values are hard one-hot samples. Backward gradients follow the mixed
probabilities, not a noisy Gumbel-softmax relaxation. There is no temperature
parameter in the prior/posterior. This matches the pinned official
[one-hot estimator](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/embodied/jax/outs.py#L228-L253).

The deterministic test mode uses the probability argmax for the forward pass and
retains the same backward rule. It makes latent selection repeatable, but does not
promise bit-identical floating-point operations across devices or PyTorch versions.

Implementation: `categorical.py`, `prior.py`, and `posterior.py`.
Regression coverage: `tests/invariants/test_dreamer_fidelity.py`.
