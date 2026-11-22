# MiniHack baseline and auxiliary ablation

Start with a new run: continuation-head dimensions and reward/value bin buffers
changed, so old checkpoints are not compatible with these model corrections.
Use the `cage2_env` Python environment (Gymnasium >= 1.1 for vector autoreset).

```bash
python -m dreamerrl.scripts.run_minihack --cpu --seed 0 --updates 1000
python -m dreamerrl.scripts.run_minihack --cpu --seed 0 --updates 1000 --aux novelty
```

New runner invocations default to `--glyph-model spatial`: glyph IDs index a learned
embedding, followed by spatial convolutions. A spatial decoder predicts categorical
glyph distributions with cross entropy, replacing scalar symlog MSE. Replay still
stores flattened raw glyph IDs. `--glyph-model mlp` selects the previous vector
baseline for paired comparisons. Resume always uses the checkpoint's architecture;
use a fresh run to switch representations.

The optional finite-horizon clock bypasses the glyph embedding and is reconstructed
with a separate scalar MSE head. Vocabulary size comes from installed NLE's
`MAX_GLYPH`, rather than an image-style 0–255 range. Decoder vocabulary logits are
chunked and recomputed during backward to bound memory use.

TensorBoard records `wm/glyph_accuracy`, `wm/glyph_changed_cells`, and, when changed
cells exist, `wm/glyph_changed_loss` and `wm/glyph_changed_accuracy`. Changed cells
are compared within the supervised window; reset boundaries are excluded. These
metrics expose navigation errors that whole-grid reconstruction can hide. New spatial runs use modest informative-cell weighting described below.
For the legacy weighted objective, `--glyph-loss-balance weighted
--glyph-changed-weight 4` gives changed cells four times the reconstruction weight,
normalized by total weight. Supplying the legacy changed-weight option without a
balance option also selects weighted mode.
Neither representation nor weighting has established a corridor learning gain;
compare held-out success over multiple seeds at equal environment-step budgets.

The default task is `MiniHack-Room-5x5-v0`. Use `--env` to select another task.
The baseline disables auxiliary objectives; the ablation adds only novelty,
a prediction of next-observation change. Novelty is representation supervision,
not an intrinsic reward added to the actor's return.

Use [the offline/tmux workflow](minihack_offline.md) for persistent JSONL metrics,
a Rich dashboard, periodic paired evaluations, and fresh-seed final testing.
The runner writes a unique `runs/<environment>_<aux>_seed<seed>_<timestamp>/`
directory unless `--run-dir` is specified. TensorBoard events are under that run's
`tensorboard/train/` directory. Repeat training with seeds 1 and 2 before drawing
conclusions. Establish Room-5x5 learning first, then try a harder task; a passing
smoke test does not demonstrate learning.

The other auxiliary heads remain available for experiments, with narrower claims:

- `skill` predicts the recorded action; it does not discover options.
- `reachability` duplicates continuation supervision.
- `resource` duplicates reward supervision until real resource labels exist.
- `affordance` requires actual action masks. The runner does not enable it because
  the MiniHack wrapper does not currently supply these labels.
- Latent clustering is deferred; its helper requires a caller-owned persistent
  projection instead of constructing fresh random targets on every call.

## Contracts covered by regression tests

Replay is batch-major `(B, L, ...)`. Each transition stores the resulting
observation and the action/reward leading to it. The trainer also inserts each
initial observation with zero action/reward and `is_first=True`. Sampled windows
never cross episodes. Replay groups episodes by effective window length:
`min(seq_len, episode_length)`, with `seq_len` defaulting to 16. Each update picks
an episode uniformly to choose a bucket, then samples rows uniformly with
replacement inside that bucket. Thus a short episode cannot shorten a long
episode's training window, and short episodes remain eligible without padding.
All episodes long enough for `seq_len` share the same full-length bucket.

Bucket probability is proportional to episode count, preserving each episode's
marginal sampling probability rather than favoring rare short lengths. Rows in a
batch share a bucket and are therefore correlated in length; singleton short
buckets repeat that episode across rows. Sequence length can still vary between
updates, and JSONL `sequence_length` reports the actual supervised length. Bucket
indexes are cached until completed episodes are added/evicted and rebuilt after
checkpoint restoration. Episodes, RNG, and burn-in remain the persisted state.
Windows starting mid-episode include up to `--burn-in` preceding transitions
(default 16). The model reconstructs recurrent context with current weights under
`no_grad`, then detaches state before the training window. Context is excluded
from world-model losses and actor/critic imagination starts; `--seq-len` still
controls the supervised window. Short prefixes are left-padded with a validity
mask so padding never advances state. `--burn-in 0` restores zero-state windows.
A bounded prefix approximates full episode context; increase it for longer memory
requirements. Replay checkpoints preserve this setting; older checkpoints without
it retain disabled burn-in when resumed.

MiniHack uses same-step vector autoreset. `state` is the observation for the next
policy action, while `transition_state` preserves the observation for the action
just executed. `is_last` marks termination or truncation; `is_terminal` marks only
true termination. `is_first` describes a reset observation in `state`, so it may
coexist with `is_last` for the separately preserved transition.

Observe and imagine advance the deterministic state before computing their
latent distributions. Imagination samples one action per transition, scores it
at its source state, and predicts reward and continuation at the resulting state.
Actor and critic updates use detached replay posteriors, continuation-aware
lambda returns, and cumulative continuation weights. Terminal replay states have
zero weight. Return normalization tracks EMA percentiles in original reward units.

Reward and value heads apply symlog once, train against two-hot targets, and apply
symexp to expected bin coordinates at readout. Continuation uses Bernoulli BCE.
KL sums over latent factors before the free-nats floor, with separate 1.0 dynamics
and 0.1 representation weights and the respective stop-gradient directions.

Auxiliary binary targets retain their original labels. Scalar novelty/resource
regression uses symlog rather than batch standardization. Losses are not clamped;
the trainer's existing gradient clipping bounds optimizer gradients. Novelty
excludes targets across resets and at the end of a sampled sequence.

```bash
python -m pytest dreamerrl/tests -q
```

The suite includes numerical return references, KL gradients and scaling,
observe/reset timing, imagination action alignment, actual auxiliary-to-RSSM
gradients, sync/async autoreset sentinel tests, and two real MiniHack optimizer
cycles. It does not establish task success or CAGE readiness; use the training
comparisons above for that milestone.

## Optional paired learning experiment

The regular suite now includes a same-seed CPU training repeatability check with
all three networks updating. For a paired three-seed experiment with random and
untrained baselines and held-out greedy evaluation:

```bash
python -m pytest dreamerrl/tests/manual/test_worldmodel_aux.py --run-manual --steps=1000 --eval-episodes=10 -s
```

Here `--steps` means optimizer updates per arm and seed. The experiment reports
whole episode returns and saves a JSON report at the printed pytest temporary
path. Copy it to preserve results. Passing verifies execution and finite metrics;
learning and novelty benefit must be assessed from the reported returns. PopGym
is optional; its repaired sync/async transition and evaluation contracts run in
the regular suite. See the test README for its separate benchmark command.

## Checkpoint and resume

The MiniHack runner saves `<run_dir>/latest.pt` every 100 completed
updates and at normal completion. The offline runner also saves update-zero
`initial.pt` and validation-selected `best.pt` snapshots. Saves replace the prior file atomically, so an
interrupted or failed write leaves the last complete checkpoint available.

```bash
# Start a run; choose a stable destination and save frequency.
python -m dreamerrl.scripts.run_minihack --cpu --updates 1000 --checkpoint checkpoints/room.pt --checkpoint-every 50

# Continue an interrupted run to its original target update count.
python -m dreamerrl.scripts.run_minihack --cpu --resume checkpoints/room.pt

# Extend a completed run to 2000 total updates (not 2000 additional updates).
python -m dreamerrl.scripts.run_minihack --cpu --resume checkpoints/room.pt --updates 2000
```

Resume uses the saved environment, seed, model, auxiliary objectives, and training
configuration; environment, model and training flags apply to new runs. `--cpu` can select
CPU on resume. `--checkpoint` can select a new destination. With no new destination,
resume updates the source checkpoint. Extending `--updates` also extends the cosine
learning-rate schedule horizon; resuming to the original target preserves its
schedule and completed-update position.

Checkpoints include all three models, optimizer states, AMP scaler when active,
completed replay episodes, original evaluation baselines/monitor state, replay RNG,
Python/NumPy/PyTorch RNG states, update and
environment-step counters, replay sample counter, and recent episode returns.
Replay data can make checkpoint files large. The format includes Python objects;
load only trusted checkpoints produced by this project.

MiniHack's live simulator is not serialized. Resume resets the collector/RSSM and
discards unfinished episodes and their partial returns, retaining completed replay
and learned state. It therefore resumes training, but does not reproduce the exact
environment trajectory of an uninterrupted run. Periodic saves occur only after
all updates for that iteration complete; a crash or Ctrl-C resumes from the last
saved boundary. Changes since that checkpoint are lost.

For direct trainer use: `save_checkpoint(path)`, `load_checkpoint(path)`, or
`DreamerTrainer.from_checkpoint(path, cuda=False)`. The instance load requires a
matching environment/model and replay layout. Call saves at completed-update
boundaries. The `train(total_updates, checkpoint_path=..., checkpoint_every=...)`
API manages counters and those boundaries automatically.

## Fidelity profile

See [the bounded fidelity check](dreamerv3_fidelity.md) for the pinned official
reference, probability-gradient and EMA-critic corrections, and explicit local
architecture/optimization differences. Checkpoint v4 persists the GRU transition, factor-preserving
heads, and slow critic. Pooled-latent v1/v2 and MLP-RSSM v3 checkpoints are rejected
before any model state is loaded; their architecture/weight shapes are incompatible. Start a new run
for this architecture. Existing checkpoint files are not modified.

## Finite trial horizons and collection chunks

By default, a time-limit cutoff is a continuing-task truncation: `is_last=True`,
`is_terminal=False`, and continuation remains eligible for bootstrapping. When an
episode is the complete finite task (for example, a fixed-length trial), set:

```python
cfg.env.finite_horizon = True
cfg.env.max_episode_steps = 30
cfg.train.seq_len = 16
```

The MiniHack runner exposes this choice as `--finite-horizon`. It is opt-in and
is saved in checkpoints. This changes the objective, rather than merely changing
how long a collection call runs.

Finite mode appends one observation feature in [0,1] describing remaining time.
Initial/reset observations have 1; the final horizon transition has 0. Each row
has its own clock, so one environment ending does not reset the others. An early
natural ending may have positive remaining time but is still terminal. Adapters
must preserve `transition_state` separately from same-step reset observations.

The trainer marks episode ends terminal in finite mode. Actor/critic updates also
use the recorded remaining steps to cut continuation at known task ends, even if
the learned continuation head is inaccurate. The last action's reward is kept;
bootstrapping and loss weights beyond the task boundary are zero. Evaluation adds
the same time feature (`task_horizon=...` for the PopGym evaluator).

Collection chunks never mark task ends. They may be shorter than a replay window
or episode: the training loop collects until its first completed episode is ready,
then can sample context spanning collection calls. The first replay warmup is
bounded by the configured episode length; an adapter that fails to end episodes
raises a clear error instead of looping indefinitely.


## Sparse rewards and credit assignment

A longer imagination rollout only helps when its reward predictions or final
critic bootstrap contain a useful signal. The runner now exposes `--discount`
(default 0.99), `--lambda-return` (0.95), and `--random-exploration-steps` (1000
environment transitions), alongside `--imagination-horizon` (15). These settings
are saved in checkpoints. Explicit changes to the new controls require a fresh
run; resuming without overrides uses the saved settings.

Optional `--replay-value-scale 1` adds recorded-reward critic supervision to the
imagined critic objective. It reconstructs replay posteriors with burn-in, then
builds detached lambda-return targets in original reward units. For observation
`obs[t]`, the immediate target reward is `reward[t+1]`, since replay stores
resulting observations. Actual terminals stop bootstrapping; truncations stop the
trace but retain the final-observation value. Reset crossings and the final
bootstrap-only observation receive no replay loss. This loss updates the critic
without gradients into the world model or actor. Recorded trajectories come from
the behavior policy, so multi-step targets can introduce off-policy bias. The
extension is experimental and defaults to 0 (disabled).

An example experiment, not an established corridor configuration:

```bash
python -m dreamerrl.scripts.run_minihack_corridor --seed 0 \
  --discount 0.997 --imagination-horizon 30 --seq-len 32 --burn-in 32 \
  --lambda-return 0.95 --replay-value-scale 1 --random-exploration-steps 10000
```

Compare each change separately against the baseline over multiple seeds and equal
environment-step budgets. Increasing sequence and imagination lengths costs more
compute per update. At gamma 0.997 a reward 100 discount intervals away retains
about 0.74 of its value, versus 0.37 at gamma 0.99; continuation probabilities and
lambda mixing can further attenuate its contribution.

JSONL train events include `replay_positive_reward_steps` and
`replay_positive_reward_episodes`, plus a `reward_coverage` object with sampled
positive-reward fraction, overall reward prediction MAE, and conditional positive
reward prediction MAE/mean when positive examples are present. These count
**training rewards after action cost**, not task success; use evaluation for task
success. Replay counters reflect completed episodes currently retained, including
eviction and checkpoint restoration. A sampled window can miss rare positives
even when replay contains them.

The `credit_assignment` object reports weighted imagined reward, continuation,
return and value means, live start weight, and (when enabled and valid) replay
value loss, target count/mean and value-target MAE. Corresponding TensorBoard
scalars are under `wm/` and `ac/`. Value-target MAE is a bootstrap diagnostic,
not error against a measured full-episode return. If replay has no positive
outcomes, stronger critic supervision cannot supply an undiscovered reward.


## Measuring training cadence

Each collection cycle performs one world-model optimizer update, then one actor
and critic update unless they are frozen or an AMP update is skipped. No additional
updates are scheduled between collection cycles. Adjust
`--collect-steps`, `--batch-size`, and `--seq-len` to change replay usage.

Every JSONL training event includes `cadence` (this cycle) and `cadence_total`
(cumulative measured cycles). TensorBoard exposes the same scalars under
`cadence/` and `cadence_total/`. Important fields are:

| Field | Meaning |
| --- | --- |
| `env_transitions` | Actual collected transitions across all environments, including extra initial replay warmup. |
| `replay_tokens` | Batch size times actual supervised sequence length; repeated samples count again. |
| `replay_tokens_per_env_transition` | Replay tokens divided by newly collected transitions. This is the replay ratio used here. |
| `world_model_optimizer_steps`, `actor_optimizer_steps`, `critic_optimizer_steps` | Completed optimizer calls, excluding frozen/skipped updates. |
| `*_updates_per_env_transition` | Corresponding optimizer count divided by collected transitions. |
| `model_burn_in_tokens`, `actor_critic_burn_in_tokens` | Valid context positions used by each posterior reconstruction pass, excluded from the replay ratio. |
| `posterior_tokens`, `burn_in_padding_tokens` | All computed posterior positions across both passes, and the subset spent on masked burn-in padding. |
| `imagination_starts`, `imagined_transitions` | Generated imagination work, including terminal/zero-weight starts; transitions equal starts times imagination horizon. |
| `collection_seconds`, `replay_sample_seconds`, `world_model_seconds`, `actor_critic_seconds` | Wall time for the named phase. CUDA is synchronized at phase boundaries so timings include completed device work. |

The existing top-level `model_tokens_per_env_step` remains an alias for this
cycle's replay ratio. The dashboard labels it “Replay tokens/env step” and also
shows world-model updates and imagined transitions per environment transition.
Cumulative ratios divide summed tokens/updates by summed transitions; they do
not average per-cycle ratios, which would misweight warmup and short batches.

For the runner defaults, a full 16-by-16 replay batch processes 256 supervised
positions after collecting 8 environments times 16 steps (128 transitions): replay
ratio 2, with one world-model update per 128 transitions (0.0078125). With a
15-step imagination rollout and active actor/critic, it generates 3,840 imagined
transitions, or 30 per collected transition. Short length buckets and extra
warmup collection change these ratios, so use the measured values.

Totals persist in checkpoints. Older checkpoints lacking these measurements start
empty totals on resume; historical usage is not inferred from update counts.
Measurement covers `train()` cycles only. Evaluation, checkpoint writes, monitor
callbacks, and standalone calls to collection/update methods are excluded. Phase
synchronization adds some CUDA measurement overhead. Compare training at equal
collected environment-step budgets, alongside replay ratio and elapsed time.


## Acting state after model updates

New runs refresh the policy's recurrent state after each successful world-model
optimizer update. The trainer re-observes each environment's unfinished episode
from zero state using the updated weights. The retained history includes the
initial observation and the actions leading to subsequent observations, so the
next policy action uses the posterior for its current observation under the
updated model. Refresh is detached and does not train the model a second time.

`--acting-context -1` (default) uses the full live episode. This preserves access
to early episode observations but adds inference work that grows with episode
length. `--acting-context 64` limits refresh to the latest 64 observations,
including the current one; this starts from zero at the retained prefix and may
lose information from earlier observations. `--acting-context 0` disables refresh
and restores carrying the previously computed state across model updates.

Same-step autoreset rows have an empty unfinished episode. They are rebuilt from
the new reset observation with zero previous action, never the preceding terminal
observation. Other rows retain their own live histories. Finite-horizon clock
features are included in this history. Refresh sampling runs in an isolated Torch
RNG scope, leaving the training/action RNG stream and replay sampler unchanged.
A skipped model optimizer update does not refresh state.

JSONL events include an `acting_state_refresh` object; TensorBoard logs its fields
under `acting/`: valid `context_tokens`, `computed_tokens`,
`max_context_length`, `truncated_rows`, `hidden_rms_change`, and
`latent_changed_fraction`. The state-change metrics include posterior resampling
noise and, for capped histories, context truncation; they are not a pure measure
of parameter drift. Cadence reports `acting_refresh_tokens` and
`acting_refresh_computed_tokens` separately from replay posterior work. Refresh
time is included in `world_model_seconds`.

The selected context policy persists in checkpoints. Older checkpoints without
this field resume with refresh disabled; new runs default to full history. Live
histories are not checkpointed because the simulator resets on resume. A resumed
collector starts with fresh observations and builds new histories.


Acting refresh encodes real observations in chunks of 256, then filters only active
rows with the RSSM and posterior. It skips prior, decoder, reward, continuation,
auxiliary heads, and KL calculations. No padded observations are encoded or
filtered: `computed_tokens` now equals `context_tokens`. Histories remain full
length unless explicitly capped. Batched filtering preserves posterior dynamics
and sampling distributions; regrouping rows and omitting unused prior samples
can change fixed-seed latent draws compared with the previous refresh path.


## Frozen location probes

To check whether position is decodable from a checkpoint's representation:

```bash
python -m dreamerrl.scripts.probe_minihack_location \
  --checkpoint runs/corridor_r2_recent_seed0_20261116/latest.pt \
  --initial-checkpoint runs/corridor_r2_recent_seed0_20261116/initial.pt \
  --train-episodes 32 --test-episodes 16 --max-steps 256 --epochs 100 \
  --output runs/corridor_r2_recent_seed0_20261116/location_probe.json
```

This CPU-only diagnostic loads and freezes model weights without restoring replay,
optimizers, or the live collector. The training process can continue separately.
Position labels come from `blstats[:2]`; the world model still receives only its
usual glyph input and optional clock. Posterior modes make feature extraction
repeatable. The checkpoint update number is recorded because `latest.pt` may later
be replaced by the running trainer.

Use `--dataset-output path.pt` to retain CPU features and labels for re-fitting.
The diagnostic does not save a training-resumable checkpoint; retaining a
weights-only snapshot separately fixes the model version for repeated comparisons.

Probe fitting uses level seeds starting at 4,000,000 and testing uses seeds
starting at 5,000,000. Splits must be disjoint. Feature normalization and probe
optimization use only the fitting split. Nonconstant features use their training
standard deviation, preserving small-amplitude signals; constant features use
scale 1. Default random actions are independent
of checkpoint policy and help cover positions; `--policy sampled` or `greedy`
measures the corresponding on-policy distribution. Collection filters every step,
but stores at most four examples per cell per episode to reduce stationary-action
dominance. `--max-per-cell` and `--max-steps` control this tradeoff.

Linear and 64-unit MLP regressors predict `(x,y)` from `h`, flattened `z`, `(h,z)`,
the posterior probabilities, and the observation-only encoder output. The optional initial checkpoint observes
exactly the same trajectories and provides an untrained `(h,z)` control. Constant
training-mean position and shuffled-training-label probes provide further controls.
The report includes training/held-out Manhattan error in grid cells, exact-cell
accuracy, accuracy within one Manhattan cell, R-squared, and per-level macro mean
error. Predictions are not clipped to the grid before scoring.

Strong held-out decoding establishes position information, not semantic room
identity, long-term memory use, or the actor's ability to use that information.
Compare recurrent probes against the observation encoder and initial model;
position is often directly visible in the glyph grid. Proving memory specifically
requires a separate test where present observations are ambiguous and history
distinguishes the correct location. A small probe failing also does not establish
that all location information is absent.

Use `--sample-latents` to repeat the diagnostic on sampled posteriors, matching the
stochastic filtering used during collection. Posterior-probability probes help
identify information that exists in the distribution but is not exposed by its
hard mode. The actor consumes `(h,z)`, not those probabilities. The optional initial
model uses an isolated sampling stream so enabling its control does not change
the trained model's trajectory or latent draws.


## Posterior collapse diagnostics

Compare retained checkpoints on the same random-action traces without altering
training:

```bash
python -m dreamerrl.scripts.diagnose_posterior \
  --checkpoint runs/corridor_r2_recent_seed0_20261116/initial.pt \
  --checkpoint runs/corridor_r2_recent_seed0_20261116/location_probe_weights_update600.pt \
  --checkpoint runs/corridor_r2_recent_seed0_20261116/posterior_diagnostic_weights_update2100.pt \
  --episodes 8 --max-steps 128 \
  --output runs/corridor_r2_recent_seed0_20261116/posterior_diagnostics.json
```

The CPU-only report includes raw posterior/prior KL before the free-nats floor,
normalized entropy, mode/sample category usage, dominant-mode fraction, and
marginal entropy variation across examples. Constant modes alone do not prove
collapse: a uniform distribution also always chooses the same argmax. Likewise,
small posterior/prior KL alone does not prove failure if useful history is already
represented by the prior.

Controlled observation swaps hold deterministic history fixed; history swaps hold
the current embedding fixed. A second observation control swaps frames only
within the same level and requires different player positions. These are neural
sensitivity tests, not guaranteed valid environment transitions or estimates of
conditional mutual information.

Decoder ablations shuffle latent samples, remove them, substitute prior samples,
and shuffle deterministic states. Glyph reconstruction is measured on uniformly
sampled frames/cells plus the actual player cell, whose coordinates are used only
for scoring. Uniform-cell CE is an estimate of grid-average loss. Zero-latent
inputs can be out of distribution; shuffling is the more informative control.
The report includes both mode and sampled posterior trajectories. Terminal
observations are included, with actions and rewards aligned to their destination.

Add `--replay-checkpoint path/to/latest.pt` to measure raw KL and category usage
through the actual saved replay sampler and burn-in path. This reads the payload
once so weights and replay come from the same update; it does not resume training.
The report records that update, sampled sequence lengths, and positive-reward
coverage. Replay and held-out random trajectories can have different distributions,
so compare both before interpreting a KL floor as posterior collapse.


## Shared categorical fields and explicit reconstruction balance

`models/categorical_fields.py` defines categorical field schemas, shared linear
heads over entity tokens, and masked loss reduction. A field has its own class
vocabulary. `CategoricalFieldDecoder` sums independently reduced field losses,
with explicit `CategoricalLossConfig` options:

- `field_weight`: scales one field relative to other fields.
- `class_weights`: optional normalized class weights within that field.
- `observed`: per-field masks exclude missing labels. Missing labels can be -1 or
  NaN when masked; valid unknown categories are supervised normally.
- `balance="weighted"`: normalized CE with a `salient_weight` multiplier.
- `balance="groups"`: separately average salient and other entities, then combine
  them using `salient_fraction`. Extra background cannot dilute this fixed budget.

Group budgets apply across valid entities in the batch. If only one group is
present it receives the full budget, avoiding a loss-scale drop on unchanged
batches. An entirely unobserved field contributes differentiable zero loss. Class
weights modify the effective target distribution and should be selected explicitly,
not inferred from the held-out test set. Per-field metrics expose raw CE, balanced
CE, weighted loss, observation counts, and effective salient mass.

MiniHack uses the same `CategoricalFieldHead` for one glyph class per cell.
Vocabulary logits remain chunked/recomputed for memory efficiency. New spatial
runner jobs default to:

```text
--glyph-loss-balance weighted
--glyph-salient-weight 2
--glyph-foreground entities
--reconstruction-scale 1
```

Salient cells are the union of within-episode changed cells and configured
foreground glyphs. `entities` uses installed NLE's monster/object glyph taxonomy,
including the visible player; it does not read privileged player coordinates.
Thus a stationary player remains salient. Use `--glyph-foreground none` for a
changed-only experiment. The selected foreground glyph IDs are saved in the
configuration, fixing the taxonomy for checkpoint restoration. All ordinary terrain
cells retain weight 1; salient cells receive weight 2 by default. The weighted sum
is divided by total weight, maintaining supervision on persistent/unchanged cells.
Use `--glyph-loss-balance groups --glyph-salient-fraction 0.5` for the stronger
fixed-budget alternative; it is now an explicit experiment rather than the default.

`--glyph-loss-balance weighted --glyph-changed-weight 1 --glyph-foreground none`
recovers the previous uniform objective. `--glyph-salient-weight` aliases the
legacy changed-weight option. In groups mode this multiplier must stay 1; use the
salient-fraction setting instead. `--reconstruction-scale` controls total
observation reconstruction (including the optional clock) relative to reward,
continuation, and KL losses. Salience grouping applies only to glyph fields.
These objective settings require a fresh run when changed through the CLI.

Old checkpoint weights remain compatible: the glyph classifier keeps its original
`classifier.weight`/`classifier.bias` keys. Missing configuration fields restore
weighted mode, empty foreground, and reconstruction scale 1. Baseline journals
also migrate these defaults. The currently running experiment is not changed by
this implementation.

The CAGE2 schema in `env/cage2/categorical_schema.py` exposes independent Activity
(`None`, `Scan`, `Exploit`) and Compromised (`No`, `Unknown`, `User`, `Privileged`)
fields. `encode_host_fields` requires explicit unique host ordering and returns
per-field targets/masks. `Unknown` is a valid compromised category; a missing host
or field is masked instead. `salient_host_fields` identifies observed non-default
states, including Unknown, for an optional grouped objective.

```python
from dreamerrl.env.cage2.categorical_schema import (
    encode_host_fields, make_cage2_decoder, informative_host_loss,
)

targets, observed = encode_host_fields(host_observations, host_order)
decoder = make_cage2_decoder(feature_dim=64)
configs, salient = informative_host_loss(targets, observed)
# host_features has shape (hosts, 64) and retains host identity.
loss, metrics = decoder.loss(
    host_features, targets, observed=observed, salient=salient, configs=configs,
)
```

This is a reusable field decoder and observation schema, not a completed CAGE2
environment integration. The host encoder, RSSM-to-host features, action mapping,
and simulator adapter still need implementation. Targets describe the BlueTable
observation/belief, not hidden true compromise state.


The CAGE2 informative weighting preset uses Activity class weights
`None=1, Scan=2, Exploit=2`. These are modest explicit trial weights, not estimates
of empirical rarity; pass `activity_weight=1` to disable the extra emphasis.
Compromised targets always receive baseline weight 1, with a multiplier of 2 only
for genuine observed changes in inferred compromise. Persistent `No`, `Unknown`,
`User`, and `Privileged` labels remain supervised. `Unknown` is not missing.

For temporal targets shaped `(batch, time, hosts)`, call
`informative_host_loss(targets, observed, is_first=is_first, is_last=is_last)`.
A change requires both adjacent labels to be observed, the same stable host order,
and no intervening episode boundary. There is no bonus at a sequence's first
observation, across missing-label gaps, or across resets. Single-frame host batches
`(batch, hosts)` are not interpreted as time. `compromise_change_weight` configures
the change multiplier; both configurable multipliers must be finite and at least 1.
Each field's CE is normalized by the sum of its valid target weights, then the
independently weighted field losses are summed. The preset never drops valid
persistent labels or uses hidden simulator compromise state.


## Event-specific scores and previous-label baselines

New training-code invocations log MiniHack reconstruction scores under TensorBoard
`wm/glyph_*` and JSONL `reconstruction_events`. These are reconstruction scores on
sampled replay, not held-out task success. Counts and correct counts accompany
accuracy so aggregation can sum counts instead of averaging unequal batch rates.

- `glyph_player_accuracy`: exact glyph accuracy at the true player cell.
- `glyph_player_paired_accuracy` versus `glyph_player_persistence_accuracy`: the
  same player-cell score restricted to frames with a valid previous label.
- `glyph_changed_accuracy` versus `glyph_changed_persistence_accuracy`: exact
  glyph accuracy on within-episode changed cells. `glyph_changed_nonterminal_*`
  and `glyph_changed_terminal_*` separate ordinary motion/observation changes from
  terminal-screen effects. Persistence is necessarily
  wrong on an exact categorical change; report its support, not just this zero.
- `glyph_persistent_accuracy` versus `glyph_persistent_baseline_accuracy`: checks
  that unchanged labels remain accurate rather than being forgotten.
- `glyph_nearby_passability_paired_accuracy` versus
  `glyph_nearby_passability_persistence_accuracy`: observable terrain classes in
  the eight cells adjacent to the current true player position.

Nearby terrain is a conservative proxy: ordinary floors, corridors, open doors,
stairs/ladders, and selected fixtures are passable; walls, closed doors, bars,
trees, and raised bridges are blocked. Dark/unseen cells, actors, objects, water,
lava, and other ability-dependent terrain are unlabelled. Unknown predicted
terrain counts as wrong on a known target and is counted separately. This does
not model NetHack's full diagonal-movement, occupancy, or hazard rules.

Replay currently stores glyphs rather than player coordinates. Training metrics
infer position only when exactly one standard player-role glyph is visible;
ambiguous/absent frames are counted under `glyph_player_unresolved_frames` and
excluded from player/neighborhood scores. The scoring API accepts explicit
`player_ids` for other characters or diagnostic `player_positions` labels. The
offline tool uses exact `blstats[:2]` positions only for scoring, never as model
inputs or training targets. Even with supplied positions, a frame without the
player glyph at that position is excluded from player/neighborhood scoring; this
avoids crediting predictions of cleared terminal screens as player accuracy.

For held-out scoring, freeze a spatial checkpoint and run:

```bash
python -m dreamerrl.scripts.score_minihack_events \
  --checkpoint runs/corridor_r2_recent_seed0_20261116/latest.pt \
  --episodes 8 --max-steps 128 --max-frames 128 --sample-latents \
  --output runs/corridor_r2_recent_seed0_20261116/event_scores.json
```

Random-action traces use separate level seeds starting at 7,000,000. Uniformly
selected frames are decoded; previous true observations remain available even
when those frames were not selected for decoding. `posterior_reconstruction`
sees the current observation, while `one_step_prior` forecasts the next/resulting
observation from real past observations and the incoming action. Keep these
separate when comparing to previous-label persistence. This is teacher-forced
one-step prediction, not an open-loop imagination benchmark. Level resets always
break persistence comparisons.

For CAGE2, `eval/cage2_events.py:score_host_fields` accepts categorical predictions,
targets, and observed masks in stable host order. It reports each field separately:
confusion matrices, per-class support/recall, false-positive counts and rates,
changed-field accuracy, and persistent-field accuracy. Model and persistence
reports use identical masks in `paired_model` and `persistence`; `all_observed`
also includes first observations where persistence is unavailable. Missing gaps,
current resets, and previous episode ends exclude temporal comparisons. `Unknown`
is a valid Compromised class, not missing data.

```python
from dreamerrl.eval.cage2_events import score_host_fields
from dreamerrl.eval.field_events import json_metrics

predictions = decoder.predict(host_features)
report = score_host_fields(predictions, targets, observed,
                           is_first=is_first, is_last=is_last)
serializable_report = json_metrics(report)
```

Temporal field arrays have shape `(batch, time, hosts)`. Single-frame arrays still
have class reports but no persistence support. Recall is TP/(TP+FN); false-positive
rate is FP/(FP+TN), not FP divided by all observations. Undefined rates are omitted
with their zero support counts, rather than emitted as NaN or misleading zeros.
These helpers do not implement the pending CAGE2 environment adapter.

### Check navigation evidence through imagination

Run a frozen, action-conditioned comparison without changing training:

```bash
python -m dreamerrl.scripts.probe_minihack_imagination \
  --checkpoint runs/<run>/latest.pt --sample-latents \
  --horizons 1 3 5 10 --anchors 64 --episodes 8 \
  --output runs/<run>/imagination_probe.json
```

The diagnostic starts from the filtered posterior at each anchor and advances the
prior using recorded incoming actions at t+1 through t+H. It never encodes future
observations during the rollout. It compares decoded endpoints with posterior
reconstruction and a teacher-forced one-step prior at the same endpoints. Those
controls see the real observation or real intervening history, respectively.

All horizons use the same anchors, with no reset/terminal boundary crossing;
terminal endpoints are allowed and changed terminal tiles are scored separately.
The persistence baseline repeats the anchor label. Changed labels are endpoint
versus anchor differences, so reverted intermediate changes are not counted.
Long-horizon eligibility favors longer surviving episodes. These are random-action
prediction diagnostics, not an evaluation of the actor's policy or planning.

Reports include player glyphs, nearby known terrain, changed nonterminal tiles,
and conditional retention: how many cues decoded correctly by the posterior are
also decoded by the prior. The conditional rate is omitted when the posterior
gets none correct; that case cannot establish a separate dynamics failure.
Repeat across seeds/checkpoints before interpreting a horizon trend.

For CAGE2, `eval.cage2_events.compare_host_rollouts` accepts posterior field
predictions, per-horizon open-loop field predictions, observed masks and common
anchors. It reports Activity/Compromised per-class recall and false positives,
changed-field scores, anchor persistence, and posterior-correct change retention.
The shared `action_conditioned_rollout` engine can generate recurrent prior states;
connecting actual CAGE2 observations/actions to the world model still requires
its environment/encoder adapter. No CAGE2 simulator results are implied.

The frozen update-3500 legacy-objective checkpoint scored 0% player accuracy in
posterior and prior decoding at horizons 1/3/5/10. Posterior nearby-terrain accuracy
was about 9%; rollout accuracy was 7–10%. Thus reconstruction already misses most
navigation cues; this experiment does not isolate an additional long-horizon
failure. See `runs/corridor_r2_recent_seed0_20261116/imagination_probe_summary.md`
and its raw JSON report. Re-run after training with informative-cell weighting.

### Compare greedy and sampled policy behavior

```bash
python -m dreamerrl.scripts.compare_minihack_policies --device cuda \
  --checkpoint runs/<run>/policy_comparison_weights_update500.pt \
  --episodes 20 --seed-start 10000 \
  --output runs/<run>/policy_comparison_update500.json
```

Use a frozen weights-only snapshot so all modes evaluate the same checkpoint
while training continues. The CLI runs all four combinations of greedy/sampled
actions and mode/sampled posterior latents on identical environment seeds. Each
mode receives the same per-episode sampling seed; trajectories diverge, so this
is not a guarantee of identical random variates across paths. One sampling
replicate per level is descriptive; repeat seeds before drawing strong conclusions.

Per-episode metrics include return, termination/truncation, step count, fraction
of unchanged positions, longest stationary streak, unique tiles, two-step returns,
failed compass movement attempts, and action counts/frequencies. Aggregate reports
include movement and frozen-game-turn rates plus paired sampled-action return
differences. Positions/game turns come from `blstats` only for diagnostics and
never enter the policy. Single-environment stepping includes the final transition
and resets movement history for each episode. NetHack's default constant penalty
checks whether game time advanced; standing still during SEARCH/OPEN/KICK is a
different event. Neither termination nor movement alone proves task success.
