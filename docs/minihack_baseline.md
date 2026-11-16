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
metrics expose navigation errors that whole-grid reconstruction can hide. Optional
`--glyph-changed-weight 4` gives changed cells four times the reconstruction weight,
normalized by total cell weight; the default is 1 (uniform categorical loss).
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
and critic update unless they are frozen or an AMP update is skipped. No extra
updates are scheduled by the legacy `updates_per_step` field: only its default
value 1 is supported; other values now raise an error instead of being ignored.
Adjust `--collect-steps`, `--batch-size`, and `--seq-len` to change replay usage.

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
