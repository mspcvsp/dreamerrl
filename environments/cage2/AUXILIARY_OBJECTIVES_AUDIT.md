> Historical pre-cleanup assessment at source `17e5e53`; built-in objectives and
> the old verification entry point were subsequently removed. Use Git history to
> reproduce this audit. This document records findings, not the maintained API.

# DreamerRL auxiliary-objective audit

Date: 2026-11-24. Audited source: `17e5e53a4fa2ef6c803f5c05bd2986df6821c0a6`.
Scope: static inspection plus synthetic CPU forward/backward checks in locked uv;
no RL experiment, model change, checkpoint modification or tag change.

## Assessment

Five optional objectives are wired into world-model learning and have working
representation gradients. None has demonstrated CAGE2 benefit in the evidence
reviewed. **Keep them disabled for the next baseline comparison.** The completed
host-aware pilot explicitly used `disable_aux_losses=true`, `aux_objectives=[]`
and `replay_value_scale=0`; its representation findings cannot establish benefits
or failures of these auxiliaries.

“Skill” currently predicts the incoming action, “reachability” predicts ordinary
continuation, and “resource” predicts the immediate recorded reward. These are not
option discovery, goal reachability, or distinct resource dynamics. Affordance
requires labels that production replay does not collect. Novelty is a one-step
observation-change proxy, not a novelty count, intrinsic reward, or curiosity
bonus. Multi-step reward supervision of the world-model representation is absent.

## Inventory and priorities

Ratings concern this implementation and the current CAGE2 research question;
they are judgments, not measured improvements. Effort estimates are engineering
time including targeted tests, excluding full controlled training comparisons.

| Objective/path | Expected research value | Implementation maturity | Technical risk | Effort before useful experiment | Experimental priority |
|---|---|---|---|---|---|
| Novelty/change prediction | Medium if changed observable fields matter | Functional, tested; semantics weak | Medium: scale, clock, policy dependence | 1–2 days for target/diagnostic review | First existing auxiliary to consider, only after offline validation |
| Affordance | Medium with genuinely observable, varying availability | Head functional; replay integration missing | High: privileged labels and schema | 2–4 days for safe label contract/replay | Conditional; defer without labels |
| Skill/action reconstruction | Low for useful options; possible action-memory diagnostic | Functional and tested as action reconstruction | Medium: trivial shortcut and misleading name | 0.5–1 day for accurate documentation/diagnostics; genuine options are a separate project | Low; diagnostic only |
| Resource/reward duplicate | Low incremental value | Functional, tested | Low–medium: redundant reward gradient and loss conflict | 0.5–1 day for controlled redundancy check | Low; do not enable by default |
| Reachability/continuation duplicate | Very low here | Functional, tested | Low–medium: constant targets, redundant gradient | <1 day for label-frequency check | Last; retain as diagnostic |
| Latent-cluster skill helper | Speculative | Unintegrated helper, not a learning algorithm | High: self-label collapse and unstable partitions | Several days for a defensible specification; hierarchical integration weeks | Defer |
| Extra `MultiRewardHead.aux` heads | No demonstrated value as implemented | Container only; no production auxiliary supervision | High if assumed trained | 1–2 days just to specify targets/losses | Do not enable |
| Existing replay-value loss | Medium for critic calibration, not representation learning | Implemented; boundary tests exist | Medium: policy/value target bias | 1–2 days diagnostic/calibration study | Separate critic investigation |
| Future multi-step reward objective (not implemented) | Medium–high hypothesis for action-conditioned temporal sufficiency | Absent | Medium–high: alignment, censoring, policy confounding | 2–4 days for minimal tested implementation | Highest new-objective candidate **only after** an offline temporal reward diagnostic |

## Exact losses and inputs

Write `s_t=[h_t,flatten(z_t)]`, `g(x)=sign(x) log(1+|x|)` and
`f_j(s)=W2_j ReLU(W1_j s+b1_j)+b2_j`. All five heads use one shared-form,
but **separately parameterized**, two-layer MLP. Scalar output size is 1;
skill and affordance output size is the discrete action count A.

Production replay is destination aligned: `(o_t,a_{t-1},r_t)` share row t.
The reset row has zero action/reward; thus an index named `action[t]` is the
incoming action, not the next action the actor should choose from `o_t`.
See [replay buffer](../../dreamerrl/replay_buffer/replay_buffer.py:84),
[collector](../../dreamerrl/training/trainer.py:669), and
[observation loop](../../dreamerrl/training/core/world_model_update.py:13).

| Name | Exact target | Loss/reduction | Consequence |
|---|---|---|---|
| `novelty` | `d_t=mean_i abs(o[t+1,i]-o[t,i])` | Mean squared error `(f_n(s_t)-g(d_t))²` over valid adjacent pairs | Predicts expected next change under replay behavior; no conditioning on the next action |
| `skill` | One-hot incoming action, or stored action vector | Mean squared error across B×L×A | Reconstructs a known transition input from the resulting state, not a temporally extended skill |
| `reachability` | `1-is_terminal[t]` | BCE-with-logits mean across B×L | Same label as main continuation, not distance/goal reachability |
| `affordance` | `action_mask[t,:]` | BCE-with-logits mean across B×L×A | Availability classification only; “usefulness” is not supplied by a validity mask |
| `resource` | `g(reward[t])` | Scalar MSE mean across B×L | Regresses the same immediate reward already supervised by the main distributional reward loss |

Definitions: [aux_objectives.py](../../dreamerrl/models/aux_objectives.py:12).
Heads/losses: [aux_heads.py](../../dreamerrl/models/aux_heads.py:8).
Production integration: [world_model_update.py](../../dreamerrl/training/core/world_model_update.py:144).

`AuxObjectiveConfig` contains only `name` and a target callback `fn(batch,gamma)`.
There are no per-objective weights/horizons; all current callbacks ignore gamma.
Defaults are `aux_objectives=[]`, `disable_aux_losses=false`, global
`aux_reward_scale=0.1` ([types.py](../../dreamerrl/utils/types.py:123)). With N selected
objectives:

```
L_world = L_reconstruction + L_reward + L_continuation
        + beta_KL*(beta_dyn*L_KL_dyn + beta_rep*L_KL_rep)
        + (0.1/N)*sum_j L_aux_j
```

The last term is absent for N=0/disabled. Defaults are `beta_KL=1`,
`beta_dyn=1`, `beta_rep=0.1`, free-nats floor 1 on the **summed** categorical KL
per example. These are repository settings, not a claim of exact reference-paper
hyperparameter equivalence. Reconstruction has its independent scale (default 1).
Consequently adding an objective dilutes each existing objective's coefficient;
one objective gets 0.1, five get 0.02 each. An all-invalid zero novelty objective
still counts in N. MSE and BCE magnitudes and action-dimension reductions are not
calibrated to equal gradient influence. Metrics log unweighted auxiliary losses,
so compare weighted gradient contributions before treating loss values as balance.

## Gradient paths

All targets are explicitly `.detach().float()` in the production loss. No
representation detach occurs between each auxiliary head and `h,z`. The
categorical sample is hard in the forward pass with the mixed-probability
straight-through Jacobian in backward ([categorical.py](../../dreamerrl/models/categorical.py:17)).

Isolated auxiliary backward checks confirmed finite nonzero gradients to:

- Its own head, `h`, and `z`.
- RSSM/GRU, posterior, and observation encoder (therefore the host-aware encoder
  receives the same path when selected).
- Earlier observation/history parameters through recurrence within the learning
  window; `is_first` resets sever the corresponding row's history.

Isolated auxiliary loss gives **zero** gradients to prior, decoder, main reward,
and continuation parameters. Prior learning remains via dynamics KL, with changed
posterior/GRU representations influencing its targets indirectly. Burn-in runs
under `no_grad` and its final state is detached, so auxiliaries do not backpropagate
into burn-in observations. The world optimizer includes all registered selected
heads. Global world gradient clipping can couple the effective updates of all
objectives ([trainer.py](../../dreamerrl/training/trainer.py:901)).

Actor and critic get no direct auxiliary gradient. Replay start reconstruction
and imagined trajectories in actor–critic learning run under `no_grad`;
REINFORCE/critic losses use detached model features. Auxiliary benefit must arise
through better representations/dynamics, not an implicit bonus or option policy.
See [actor_critic_update.py](../../dreamerrl/training/core/actor_critic_update.py:31)
and [imagination.py](../../dreamerrl/training/core/imagination.py:28).

## Correctness and maturity gaps

1. **Affordance cannot run through ordinary production replay as configured.**
   `ReplayBuffer.add/sample` and collector do not store `action_mask`.
   The callback correctly raises instead of inventing all-positive labels.
   Synthetic tests provide masks manually; this does not establish end-to-end
   CAGE2 support. Only an agent-visible native validity contract is acceptable;
   querying hidden compromise/process state to manufacture “useful actions” would
   leak privileged information. A static all-valid 145-action mask adds no signal.
2. **Novelty boundary masking is good for the current sampler, incomplete for
   arbitrary batches.** It excludes the final replay row, next `is_first`, and
   current `is_last`/`done`. It does not independently inspect `is_terminal` when
   both last/done are absent. Synthetic terminal-only batches retain a loss
   (0.48638); supplying matching `is_last` makes it zero. Current production replay
   supplies done/last and episode-contained windows, limiting practical exposure.
   Terminal destination observations may still contribute incoming change, which
   is valid prediction but can emphasize blank terminal artifacts in legacy grids.
3. **No generic learning-window padding/valid-field mask.** Only burn-in is padded
   and masked today; learning windows are length-bucketed and unpadded, so their
   scalar means are appropriate. A future padded sampler would need loss masks.
   Reset rows contribute zero-action/zero-reward targets and positive continuation;
   decide whether this is scientifically useful before comparing auxiliaries.
4. **Observation-schema inconsistency.** Generic model loss accepts `state` as an
   observation alias, but novelty directly requires `batch['obs']`; verified
   `KeyError('obs')` on an otherwise valid state-only batch. It assumes rank-3 flat
   observations and averages coordinate differences with no per-field validity.
5. **Raw categorical differences lack a general metric.** Numeric glyph-ID
   distance is arbitrary; encoded CAGE2 field changes can mix very different
   semantics and scales. An explicit clock, if present in an experiment, is also
   part of the change average. Its deterministic contribution can reward time
   tracking. Not all observable changes are novel, controllable, or valuable.
6. **Skill shortcut / policy dependence.** The incoming action already enters the
   GRU. Reconstructing it can spend capacity on action memory without improving
   future reward prediction. Targets vary with behavior policy; no option duration,
   skill prior, diversity term, manager, termination or skill-conditioned actor
   exists. A head-only 50-step optimizer test verifies learnability, not skill
   discovery or RL sample efficiency. Its MSE logits are not calibrated class
   probabilities or a standard categorical negative-log-likelihood.
7. **Resource and reachability redundancy.** Continuation duplicates the same BCE
   target; resource duplicates reward using a different statistical objective
   (MSE symlog scalar versus two-hot symlog distribution). Differences may change
   representation weighting but provide no new supervision. Most truncated CAGE2
   trajectories need not provide varied true-terminal labels; verify frequencies
   before interpreting near-perfect reachability accuracy.
8. **Experimental helper is not integrated.** `latent_cluster_skill_target` demands
   a caller-owned persistent projection and returns softmax assignments. The
   callback signature includes extra arguments, has no configured objective,
   persistent projector, training rule, temperature validation or collapse
   prevention. Do not treat post-hoc K-means as discovered executable skills.
9. **Extra reward container is dormant.** WorldModel sets `num_aux=0`; the production
   loss discards extra reward outputs. Configuring the container alone would leave
   those heads unsupervised. It is not multi-step reward prediction.
10. **Avoidable work and RNG coupling.** With auxiliaries enabled, all five heads
    are constructed then only selected heads registered. This consumes RNG even
    for discarded heads, including an empty selected list when the disable flag
    is false; this can change subsequent actor/critic initialization RNG relative
    to explicit disable. `observe_step` computes heads each timestep, then the
    training loss recomputes selected heads on stacked states. That first set of
    prediction graphs is unused by the auxiliary loss. Do not optimize this during
    the audit or silently change initialization for controlled comparisons.

BCE-with-logits, symlog scalar targets, straight-through categorical gradients,
shape checks, singleton handling, and differentiable zero for no valid novelty
pairs work in the checks. There is no auxiliary loss clamp hiding large losses.
Finite extreme-bias tests cover gradients at losses above 1; exhaustive AMP,
large-magnitude/NaN target and environment-specific label validation are absent.

## Checkpoint compatibility

Selected heads live under `aux_heads.<name>.fc1/fc2`; they are registered model
parameters and included in world state and optimizer checkpoints. An in-memory
same-configuration Torch serialization round trip passed bitwise parameter checks.
Loading enabled-head weights strictly into a disabled model correctly failed.
The production restart loader also compares saved world/actor/environment configs
before strict model and optimizer restore
([trainer.py](../../dreamerrl/training/trainer.py:384)).

Enabling heads on the original baseline is a new configuration: do not load its
optimizer state as though schemas match. Callbacks are Python functions, so saved
configs additionally depend on their import paths and source revision. The audit
verified model/config serialization, not a full auxiliary-enabled trainer restart
with restored Adam moments. Existing general restart tests do not establish that
specific combination. No historical checkpoint was loaded or rewritten here. Final preservation checks
rehashed both the pilot and original update-1 checkpoints and the original CAGE2
lockfile against recorded fingerprints; all matched. Tag objects also matched
the recorded baseline and pilot identifiers. Receipts are in `preservation.json`.

## Interpretation against the frozen pilot

Evidence: [linear report](../../runs/cage2_latent_analysis_host_pilot_20261124/final_report.txt),
[latent diagnostics](../../runs/cage2_latent_analysis_host_pilot_20261124/latent_diagnostics.json),
[clustering results](../../runs/cage2_latent_clustering_host_pilot_20261124/selected_clusters.json).
These files are generated evidence, never executable dependencies.

- Combined latent R²=0.76052 versus timestep lookup 0.59394 and observation+time
  0.71947. Combined latent+observation+time R²=0.76037; paired episode-bootstrap
  incremental R² interval [0.01690,0.06382]. There is useful predictive structure
  beyond these controls. A new objective must improve this and/or behavior,
  not just lower a training loss. Correlation does not prove causal memory.
- h effective rank 8.112/200; residual rank 6.588. Low-dimensional useful dynamics
  can explain these numbers; increasing rank is not itself a defensible objective.
- Test posterior entropy 0.8368 nats/factor versus maximum log(30)=3.4012;
  posterior–prior KL about 0.9996 nats/state, no constant hard-mode factors and no
  near-constant h units. Fitting modes use 2–6 categories/factor. Restricted greedy
  visitation and mode-based collection limit coverage; this is not evidence of
  total posterior collapse, and does not justify entropy maximization by itself.
- Raw h selected K=10: silhouette 0.4305, initialization ARI 0.7205,
  timestep NMI 0.4449; persistence exceeds its clock-conditioned null by -0.0041.
  Some clusters isolate the same early timestep across all 72 test episodes.
  Their apparent phases are largely compatible with episode progression.
- Residual h selected K=2: silhouette 0.4028, bootstrap-refit ARI 0.9602,
  timestep NMI 0.0795, excess persistence +0.2408. However **initialization ARI is
  -0.0232** across the three starts, despite the attractive bootstrap score.
  The bootstrap refits use the same initialization seed; they do not resolve
  competing local minima (inertias 1,018,173 versus 1,198,825/1,193,506).
  The automatic clustering report's positive residual-structure sentence is too
  strong without this qualification. Historical results/report are left unchanged.
  Do not turn these partitions into option labels or privileged “attack phases.”

None of these diagnostics compares auxiliary-on/off training seeds. They support
preserving useful existing structure while measuring temporal predictive deficits,
not adding all objectives or enforcing artificial latent diversity.

## Multi-step reward prediction: what exists

The main reward head predicts one destination-aligned reward from each posterior
state. Prior imagination repeatedly predicts rewards along action-conditioned
rollouts and the critic learns lambda returns. This provides multi-step **inference
and critic targets**, not a direct multi-step real-replay reward-prediction loss on
the starting world-model state. Prediction-head losses are evaluated on posterior
states; there is no reward overshooting/unrolled-prior reward supervision here.

The optional `replay_value_scale` (default 0) adds a real-replay lambda-return value
loss to the **critic**. Replay posterior features and target construction are
no-grad; it cannot directly make h/z/encoder retain reward-predictive information.
See [replay_value.py](../../dreamerrl/training/core/replay_value.py:9).
The offline ridge return probe likewise never updates Dreamer.

A future direct multi-horizon head or short action-conditioned prior-rollout reward
loss would therefore be a new auxiliary objective. It must distinguish
`sum_{i=1..k} gamma^(i-1) r[t+i]` from incoming `r[t]`, use only available replay
futures, cut resets, distinguish terminal from truncation, and avoid supplying
future actions to a nominally state-only prediction input. Recorded action
sequences are legitimate inputs for an explicitly action-conditioned model;
future observations/rewards must remain supervision only. Measure a time-only
baseline and conditional improvement, since cumulative return can be clock driven.
No such objective was implemented during this audit.

## Verification receipts and rerun commands

Generated receipts: `runs/cage2_auxiliary_audit_20261124/`.

| Check | Result |
|---|---|
| Root uv auxiliary head suite | 6 passed (0.60s) |
| Root uv production contracts, `--noconftest` | 21 passed (0.79s) |
| Explicit legacy uv full combined suite | Same 27 tests passed (0.78s); not 27 additional unique tests |
| Root replay-value delayed-credit/boundary checks | 6 passed, 5 deselected (0.21s) |
| Static checks | Ruff passed; repository Pyright: 0 errors, 0 warnings |
| Additional root synthetic audit | All five heads: finite loss, nonzero h/z/encoder/GRU/posterior/head gradients; zero isolated prior/other-head gradients |
| Terminal-only and `state` alias | Existing limitations reproduced, documented above |
| Same-config model serialization | Passed; incompatible head schema rejected |

The ordinary combined root command initially failed collection because the
invariants conftest unconditionally imports MiniGrid. Root CAGE2 deliberately
excludes MiniGrid; no dependency was added or pin changed. Root contract tests
are self-contained and pass with `--noconftest`. Legacy test run independently
confirms normal fixture collection. Using `-c /dev/null` bypasses project marker
registration, causing harmless unknown-marker warnings; legacy also reports its
existing pkg_resources deprecation. These runs do not establish full GPU/AMP
auxiliary-enabled trainer or real CAGE2 learning effectiveness.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --locked pytest \
  -c /dev/null -p no:cacheprovider -q dreamerrl/tests/aux
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --locked pytest \
  -c /dev/null -p no:cacheprovider --noconftest -q \
  dreamerrl/tests/invariants/test_learning_contracts.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 UV_PROJECT_ENVIRONMENT=.venv-legacy \
  uv run --locked --no-default-groups --group legacy pytest \
  -c /dev/null -p no:cacheprovider -q dreamerrl/tests/aux \
  dreamerrl/tests/invariants/test_learning_contracts.py
uv run --locked python environments/cage2/auxiliary_audit_checks.py \
  --output runs/cage2_auxiliary_audit_20261124_rerun/synthetic_checks.json
```

Additional replay-value checks:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --locked pytest \
  -c /dev/null -p no:cacheprovider -q \
  dreamerrl/tests/functional/test_sparse_reward_credit.py \
  -k 'delayed_recorded or terminal_cuts or reset_crossings or replay_critic_loss or no_valid_replay'
```

The audit-check command runs backward only, no optimizer steps. Its tracked-source
candidate is provided for review; no training algorithm or objective changed.
Source revision, lock hash and verification-source hash are saved with results.

## Recommendations and costs

1. **First: an offline temporal reward diagnostic**, not another auxiliary-on
   training run. Measure real-replay short-horizon reward prediction from frozen
   h/z against observation+time and action-history controls, including prior
   rollout reward calibration. Use episode splits and time-residual metrics.
   Estimate 0.5–1 day engineering, minutes of CPU/GPU inference using saved replay.
   This establishes whether the proposed new objective addresses an actual gap.
2. Before any existing-head ablation, review target semantics, the missing
   affordance replay contract, terminal-only masks, and loss-weight normalization.
   Add enabled-head optimizer/restart and target-frequency checks. Estimate 1–3
   days depending on label integration. Findings only; no fixes made here.
3. If a representation deficit is established, select **one** objective and run
   equal-interaction, independent-seed comparisons with development-only selection.
   Keep historical baseline distinguished from fresh controlled arms. Three seeds
   × two arms × approximately 2.5h at the pilot budget is roughly 15 Spark-hours
   before auxiliary overhead/evaluations; this is a planning estimate, not measured
   auxiliary throughput. Formal experiments require separate approval.
4. Defer hierarchical actors and cluster-derived skills until initialization and
   episode stability, observable semantics, action-conditioned prediction and
   behavior relevance have been verified. Do not optimize cluster aesthetics.

A head with hidden width H and output D adds `(1160+1)*H+(H+1)*D` parameters.
For the pilot's H=200 and A=145: scalar head 232,401; action-vector head 261,345;
all five 1,219,893 parameters. Adam FP32 parameter/gradient/moment storage is
approximately 16 bytes/parameter (about 19.5 MB for all five, excluding activations).
The dense first layer costs approximately `B*L*1160*H` multiply-accumulates per
head forward; current repeated `observe_step`/stacked passes add avoidable forward
work. Backward and environment overhead prevent translating this into a measured
wall-time percentage. Measure before promising efficiency; the default empty
objective list has no registered auxiliary-head parameter overhead.

## Scientific grounding

[DreamerV3](https://arxiv.org/abs/2301.04104) motivates representation learning
through observation/reward/continuation prediction and latent dynamics learning;
it does not establish benefits of these five project-specific targets.
[UNREAL](https://arxiv.org/abs/1611.05397) provides precedent for auxiliary reward
prediction improving shared RL representations, with a different agent and
experimental protocol; transfer to CAGE2 requires an ablation.
[DIAYN](https://arxiv.org/abs/1802.06070) uses a skill-conditioned policy and an
information-theoretic discovery objective, illustrating why incoming-action
reconstruction or post-hoc state clustering alone is not skill discovery.
These precedents support hypotheses, not claims of CAGE2 gains.
