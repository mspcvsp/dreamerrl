# Bounded DreamerV3 fidelity check

Checked 2026-11-14. Reference: official DreamerV3 commit
[`e01491fad6434b2245a3b8ca201dd7faedcc458c`](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c),
with the [April 2024 paper](https://arxiv.org/html/2301.04104v2) as an equations cross-check.
The current official implementation includes changes beyond early DreamerV3 versions;
this report states which reference behavior was selected.

## Result

The bounded check is complete for categorical distributions/gradients, imagined
actor/critic targets, continuation weighting, KL balancing, and training-state
persistence. The material gaps found in categorical mixing, sampling gradients,
and critic regularization have been corrected and tested. This is a simplified
DreamerV3 variant, not a claim to reproduce the official architecture or scores.

| Contract | Result |
| --- | --- |
| Prior/posterior and discrete actor distributions | 1% uniform mixture; probabilities remain positive at extreme logits. |
| Latent sampling gradient | Hard categorical sample; backward Jacobian follows mixed probabilities, without a noisy Gumbel-softmax relaxation. Deterministic test modes keep this backward rule. |
| Discrete actor objective | REINFORCE on detached imagined states; sampling, log-probabilities and entropy share the mixed policy distribution. |
| Critic returns | Original-unit lambda returns, online critic bootstraps, continuation-aware discounts and cumulative weights. |
| Slow critic | Detached two-hot supervision from its scalar prediction, loss scale 1; EMA rate 0.02 after each online critic update. It regularizes the critic and does not replace online return bootstraps. |
| KL | Sum categorical factors before a 1-nat floor; dynamics/representation stop-gradients stay separate. Default scales corrected to 1.0 / 0.1. |
| Scalar heads | Symlog two-hot training and symexp readout; Bernoulli continuation. |
| Actor normalization/defaults | EMA return percentiles and a scale floor of 1; entropy coefficient corrected to 3e-4. |
| Gated dynamics | Standard GRUCell over embedded previous latent/action inputs, with the deterministic state passed explicitly. |
| Persistence | Checkpoint v4 includes the GRU, factor-preserving heads, and the EMA critic. Fixed-batch continuation restores optimizer/RNG state. |

Distribution behavior is grounded in the reference
[categorical and one-hot outputs](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/embodied/jax/outs.py#L196-L253).
Critic behavior follows the reference
[imagined loss](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py#L353-L390),
[EMA update](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/embodied/jax/utils.py#L88-L118),
and [configured coefficients](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/configs.yaml#L79-L107).

## Explicit differences from the full reference

These are active design choices/omissions, not quarantined broken paths:

- The recurrent transition now uses a standard
  [PyTorch GRUCell](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html).
  Previous latent and action are concatenated, projected, normalized, and activated
  before the cell; previous deterministic state is its hidden input. It uses
  PyTorch's standard gate equations and GRU initialization, rather than the modern
  reference's customized block-gated core. Prediction heads and the actor preserve
  factor identity through flattened latent projections. Network sizes remain much
  smaller. See reference [RSSM](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/rssm.py#L126-L149)
  and [feature construction](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py#L46-L48).
- Adam, global gradient clipping, and separate optimizers are retained; the modern
  reference uses an AGC/RMS/momentum optimizer chain. Xavier/random prediction-head
  initialization is retained rather than the reference output scaling. These are
  optimization differences, so published hyperparameter robustness is not established.
- Reconstruction MSE is averaged across observation features. Reference output
  losses aggregate feature dimensions differently, changing its effective balance
  against KL. Smaller reward/value supports (41 bins over symlog [-10,10]) are
  retained; the reference uses 255 bins. Out-of-support targets still clamp to endpoints.
- Critic training defaults to imagined trajectories. Optional recorded-reward
  replay-value supervision is available as an experimental local extension; its
  behavior-policy lambda targets do not establish equivalence to reference replay
  supervision. Replay windows reconstruct up to 16 preceding transitions with current weights
  and detached gradients (configurable burn-in), rather than full episode context. Sampling is episode-based.
- Discount 0.99 and task-specific imagination horizons remain configurable local
  choices. Continuation predicts actual survival and discount is applied separately.
  Return-percentile EMA is initialized from the first batch rather than zero.
- New MiniHack runner jobs use learned glyph embeddings, spatial convolutions,
  and categorical grid reconstruction; the legacy scalar MLP remains selectable.
  This is a task-specific symbolic representation, not the reference image model.
  Novelty is an experimental observation-change proxy,
  not part of the reference DreamerV3 reward or objective.

The pinned [configuration](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/configs.yaml#L79-L110)
and [optimizer/loss definitions](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py#L314-L351)
identify the comparison profile. These differences prevent calling this a full
reference reproduction. Evaluate the MiniHack baseline across seeds before
attributing performance changes to the auxiliary objective; a passing fidelity
contract test does not establish task learning.

## Regression evidence and checkpoint compatibility

`tests/invariants/test_dreamer_fidelity.py` checks exact uniform mixing, positive
probabilities at extreme logits, actual prior/posterior one-hot samples and their
noise-independent probability gradients, actor sampling/scoring consistency,
numerical slow-critic cross-entropy, detached gradient boundaries, and EMA update
order/formula. Existing learning-contract tests cover returns, continuation cutoffs,
reset timing, KL floor/scales, and two-hot units. Checkpoint and same-seed tests now
include the slow critic and GRU parameters. `tests/invariants/test_rssm_gru.py`
checks the gate equations, retain/replace extremes over 40 transitions, flattened
versus factored inputs, and gradients through a multi-step sequence.

Checkpoint v4 is required for the standard GRU recurrence. Pooled-latent v1/v2
and MLP-RSSM v3 checkpoints are rejected with architecture-specific messages before
loading model or optimizer state. No implicit weight/optimizer
conversion is performed. New finite-task configurations also add one time feature
and must match the saved configuration when resuming.

Replay now retains short trials with unpadded, shared-length batches instead of
filtering them out. Collection chunk size no longer limits replay sequence length.
Optional `finite_horizon` training makes remaining time observable, trains finite
terminal targets, and bounds imagined returns at the known trial end. The default
continuing-task semantics remain appropriate for artificial time-limit cutoffs.
See [horizon handling](minihack_baseline.md#finite-trial-horizons-and-collection-chunks).

```bash
python -m pytest dreamerrl/tests/invariants/test_dreamer_fidelity.py dreamerrl/tests/functional/test_checkpoint_resume.py -q
python -m pytest dreamerrl/tests -q
```
