# MiniHack baseline and auxiliary ablation

Start with a new run: continuation-head dimensions and reward/value bin buffers
changed, so old checkpoints are not compatible with these model corrections.
Use the `cage2_env` Python environment (Gymnasium >= 1.1 for vector autoreset).

```bash
python -m dreamerrl.scripts.run_minihack --cpu --seed 0 --updates 1000
python -m dreamerrl.scripts.run_minihack --cpu --seed 0 --updates 1000 --aux novelty
```

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
never cross episodes. Replay selects episodes before choosing a shared batch
length: the minimum of configured `seq_len` (default 16) and the shortest selected
episode. Short episodes are retained without padding. Sequence length can vary
between batches; frequent short episodes reduce context for that whole batch.
Windows starting mid-episode initialize recurrent state at
zero without a burn-in prefix.

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
