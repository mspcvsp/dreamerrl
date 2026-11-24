> Historical MiniHack research is OBE; CAGE2 is the primary environment.
> Example paths below refer to old artifacts proposed for user-approved cleanup.
> These examples may require regenerated data after cleanup; tracked environments and tests remain supported.

# Offline MiniHack learning runs on DGX Spark

The runner uses Rich for a local live dashboard, append-only JSONL for metrics,
Python logging for rotating diagnostic files, and TensorBoard files. No Loguru,
Textual, W&B account, network service, or browser is required. Evaluation uses
separate MiniHack environments and restores Python, NumPy and PyTorch RNG state.

## Prepare the environment before going offline

Use the explicitly selected legacy uv profile. NLE 1.3.0 requires Gymnasium
1.2.0; the default CAGE2 profile retains 1.4.0. Install without activation:

```bash
bash scripts/setup_uv.sh --legacy
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-legacy"
```

All commands below run from the repository root. See [uv profiles](uv_environment.md)
for setuptools compatibility and native Spark dependencies. Prepare dependencies
before disconnecting from the network.

Check the selected Python environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -c 'import torch, minihack, nle, gymnasium, rich; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

A two-update FP32 training/evaluation smoke run was verified in this workspace on
an aarch64 NVIDIA GB10, including checkpointing. The installed PyTorch reports a
capability-range warning for this GPU; the smoke run succeeded. Validate the
actual environment you will use rather than assuming that CUDA availability alone
proves every training kernel works. Runtime versions, device name/capability,
compiled architectures, git commit/dirty flag, and configuration are recorded.

## Run in tmux

From the project directory, open a session:

```bash
tmux new-session -s dreamer-room
```

Inside that session:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -u -m dreamerrl.scripts.run_minihack \
  --require-cuda --seed 0 --updates 10000 \
  --run-dir runs/room_seed0 --eval-every 100 --eval-episodes 20
```

Detach with **Ctrl-b, d** and reconnect with `tmux attach -t dreamer-room`.
SSH disconnection does not stop a detached tmux session. The dashboard uses Rich's
[Live display](https://rich.readthedocs.io/en/stable/live.html), with a plain-output
fallback when stdout is redirected. `--dashboard off` is available for batch jobs.
Neither the dashboard nor tmux is needed to retain the metric files.

**Ctrl-C or SIGTERM requests a stop after the current complete optimizer update.**
Evaluation can stop between environment steps. The runner then saves `latest.pt`,
flushes metrics, and exits with a `run_end` record marked `stopped`. SIGKILL or
power loss cannot run that cleanup: resume from the last periodic checkpoint.
The JSONL writer repairs an interrupted trailing record before appending on resume.

Resume:

```bash
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -u -m dreamerrl.scripts.run_minihack \
  --require-cuda --resume runs/room_seed0/latest.pt
```

The saved target and training configuration are restored. `--updates 20000` extends
the target to 20000 total updates. `--run-dir` can relocate the resumed output;
baselines also travel inside the checkpoint. A fresh run refuses to mix with an
existing run directory. Original random/untrained baselines are reused on resume.
Legacy v4 checkpoints without baseline metadata leave the untrained comparison
unavailable rather than mislabeling their trained weights as an initial policy.

## Files and dashboard

| File | Purpose |
| --- | --- |
| `config.json` | Initial configuration, including objective and auxiliary choices. |
| `metrics.jsonl` | UTC timestamped/session-tagged startup, episode, update, evaluation, checkpoint, stop and error records. |
| `run.log` | Human-readable diagnostics and tracebacks; rotates at 5 MiB with three backups. |
| `baselines.json` | Original random/untrained episode returns, fixed validation seeds, and matching configuration. |
| `evaluation.json` | Latest trained-policy validation result and paired comparisons. |
| `initial.pt` | Update-zero model snapshot for fresh final-test comparisons. |
| `best.pt` | Highest validation-mean checkpoint; selection is not proof of learning. |
| `latest.pt` | Periodic/latest safe training checkpoint, including replay and monitor metadata. |
| `tensorboard/train/` | Local training and evaluation TensorBoard events. |
| `failure.json` | Failure details, including initialization failures before the monitor starts. |

The dashboard shows progress, per-update environment throughput, smoothed training
ETA, replay size/context length, completed episodes, recent training return, losses,
KL terms, model tokens per environment transition, and allocated CUDA memory. Its
validation panel shows random, untrained and trained whole-episode returns plus
paired mean differences with approximate 95% intervals. ETA excludes future
evaluation/checkpoint time. Per-update throughput is not a complete GPU profiler.

Inspect files without attaching to tmux:

```bash
tail -n 5 runs/room_seed0/metrics.jsonl
tail -n 30 runs/room_seed0/run.log
cat runs/room_seed0/evaluation.json
```

TensorBoard can be started locally later; it is not contacted by training:

```bash
tensorboard --logdir runs/room_seed0/tensorboard
```

## Establish improvement, rather than just a successful run

1. Start with `--aux none` (the default). The initial policy and uniform random
   policy are evaluated on the same fixed validation levels before learning.
2. Compare whole-episode returns at equal environment-step budgets. Higher is
   better even when every score is negative. Loss decreases alone do not establish
   policy improvement.
3. Repeat independent training seeds 0, 1 and 2, with distinct run directories.
   Use the same task configuration and final environment-step budgets.
4. Evaluate the chosen trained checkpoint on **fresh test seeds**, paired with
   the saved initial policy and random actions:

```bash
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -m dreamerrl.scripts.evaluate_minihack --cpu \
  --checkpoint runs/room_seed0/best.pt \
  --initial-checkpoint runs/room_seed0/initial.pt \
  --episodes 100 --seed-start 1000000 \
  --output runs/room_seed0/test_evaluation.json
```

Omit `--cpu` to use the saved device. The evaluator rejects overlap with validation
seeds and rejects a purported initial checkpoint whose update count is nonzero.
It writes all per-episode returns and seeds, means, environment budget, training
seed, and paired differences to JSON. Inspect improvements against **both**
baselines across training seeds, not only the strongest run.

The intervals are paired normal approximations over evaluation episodes, not
corrections for repeated validation, model selection, or variation across training
seeds. With one episode no interval is reported. The runner deliberately does not
print a learning "PASS" based on one mean/interval. A smoke run, best checkpoint,
or positive validation difference is not evidence of reliable sample efficiency.

## Defaults and useful controls

New-run defaults: Room-5x5, auxiliaries off, 8 environments, 200-step episode cap,
16 collection steps/environment/update, batch 16, context 16, imagination 15,
500-update LR warmup, 10000 updates, FP32, checkpoint every 100 updates, evaluation
every 50 updates on 20 fixed seeds. The reduced collection chunk compared with the
old 250-step runner increases optimizer updates per collected transition; the
actual ratio is in the metrics. It is a starting configuration, not a tuned claim.

Training controls include `--num-envs`, `--parallel`, `--episode-steps`,
`--collect-steps`, `--batch-size`, `--seq-len`, `--imagination-horizon`,
`--warmup-updates`, `--amp` (opt-in CUDA FP16), and `--finite-horizon`.
`--require-cuda` prevents silent CPU fallback. `--cpu` is useful for wiring checks.
No torch compilation, multiprocessing dashboard, or new logging framework is added.

Training/environment/AMP flags apply to new runs; resume uses its saved profile.
Evaluation cadence can change via `--eval-every`; `0` means final evaluation only.
Validation seeds and episode count are restored automatically. Keep them unchanged
on resume so cached baselines remain paired; use the separate evaluator for a new
seed set. New-run `--eval-episodes` and `--eval-seed-start` select that validation set.

A short CPU wiring check:

```bash
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -m dreamerrl.scripts.run_minihack --cpu --updates 2 \
  --num-envs 2 --episode-steps 10 --collect-steps 5 --batch-size 2 --seq-len 4 \
  --imagination-horizon 3 --eval-episodes 2 --eval-every 1 \
  --run-dir runs/smoke --dashboard off
```

## Diagnose greedy versus sampled policies

Keep action selection and posterior latent selection independent when a high
training return disagrees with deterministic validation:

```bash
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -m dreamerrl.scripts.evaluate_minihack_modes --cpu \
  --checkpoint runs/room_seed0/latest.pt \
  --checkpoint runs/room_seed1/latest.pt \
  --checkpoint runs/room_seed2/latest.pt \
  --episodes 100 --seed-start 1000000 --sampling-seed 42 \
  --output runs/room_evaluation_modes.json
```

This evaluates greedy/mode, sampled/mode, greedy/sampled, and sampled/sampled
policies. Each episode uses its level seed plus the fixed sampling seed for Torch
sampling, independently of caller RNG position. All RNG states are restored.
Default training validation remains greedy/mode. The report retains all episode
returns, checkpoint budgets, settings, and paired differences versus random.
One fixed sampling seed is a diagnostic comparison, not an estimate of variation
across sampling seeds. These test levels have now been inspected: use new levels
for confirmation after making any changes based on the results.

## Corridor training and evaluation

Use the separate corridor entry point to test navigation through rooms connected
by generated corridors. It defaults to `MiniHack-Corridor-R2-v0` and a 1000-step
episode cap; other training defaults and monitoring match the Room runner.

```bash
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -u -m dreamerrl.scripts.run_minihack_corridor \
  --require-cuda --seed 0 --updates 10000 \
  --run-dir runs/corridor_r2_seed0 --eval-every 100 --eval-episodes 20
```

Select more rooms with `--env MiniHack-Corridor-R3-v0` or
`--env MiniHack-Corridor-R5-v0`. Use a distinct run directory for each task/seed.
These runs train a fresh policy with the corridor action space. As with Room,
compare trained returns against both random and untrained policies; completing
a run alone does not establish learning.

For a short CPU wiring check:

```bash
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -m dreamerrl.scripts.run_minihack_corridor --cpu --updates 2 \
  --num-envs 2 --episode-steps 10 --collect-steps 5 --batch-size 2 --seq-len 4 \
  --imagination-horizon 3 --eval-episodes 2 --eval-every 1 \
  --run-dir runs/corridor_smoke --dashboard off
```

Evaluate the selected checkpoint on fresh test levels using the same evaluator:

```bash
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -m dreamerrl.scripts.evaluate_minihack \
  --checkpoint runs/corridor_r2_seed0/best.pt \
  --initial-checkpoint runs/corridor_r2_seed0/initial.pt \
  --episodes 100 --seed-start 1000000 \
  --output runs/corridor_r2_seed0/test_evaluation.json
```

The checkpoint supplies the corridor environment and episode limit. Resume with
`UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -m dreamerrl.scripts.run_minihack_corridor --resume runs/corridor_r2_seed0/latest.pt`.

## NetHack evaluation seeding

Evaluation explicitly seeds NetHack's core, display, and level-generation RNGs
before every episode, with automatic reseeding disabled. Gymnasium's
`reset(seed=...)` alone does not seed these RNGs in the installed NLE version.
The action space and Torch sampling are seeded separately. Regression tests
compare real Room/Corridor glyph trajectories and rewards across repeated runs.

Reports and cached baselines generated before this fix do not guarantee matching
NetHack levels, even when their recorded episode seeds match. Re-evaluate saved
trained and initial checkpoints with the corrected evaluator for fresh paired
comparisons. Resuming an old run retains its cached baselines; those historical
baseline comparisons should not be treated as paired with corrected evaluations.
Training environment collection seeding is unchanged by this evaluation fix.

## Optional per-action training cost

`--action-cost` subtracts a fixed nonnegative cost from every collected action's
reward, including the final action. It defaults to zero. Replay stores these
shaped rewards, so world-model reward learning and imagined policy/value learning
use the cost. Initial reset observations incur no cost. Reported training episode
returns and all evaluation returns remain the original environment rewards.
This is reward shaping, not an auxiliary prediction loss.

For a separate corridor experiment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy python -m dreamerrl.scripts.run_minihack_corridor \
  --require-cuda --updates 10000 --eval-every 500 \
  --action-cost 0.001 --run-dir runs/corridor_r2_cost001_seed0
```

The value above is an experimental starting point, not a tuned recommendation.
The saved configuration retains the cost on resume. An explicit cost that differs
from the checkpoint is rejected: use a fresh run to change the objective without
mixing differently shaped rewards in replay. Older checkpoints default to zero.
A per-action cost encourages shorter episodes and may also encourage early
failure; assess goal success and episode lengths alongside original returns.
