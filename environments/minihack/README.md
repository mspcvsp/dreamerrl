# Director Room 5x5 experiments

The maintained implementation and diagnostic source live in this repository.
Use the exact source tag for a sealed run. The tags preserve the trainer,
configuration, and entry points used for each experiment. Use the committed
execution manifest from the integrating checkout when a launcher requires one.
Manifest checks resolve the manifest's own Git checkout and verify the tagged
execution checkout separately.

| Experiment | Source tag | Committed execution manifest or configuration |
|---|---|---|
| Original Director | `dreamerv3-director-room5x5-v1` | `director_room5x5_execution_manifest.json` |
| Director with flat AMP correction | `dreamerv3-director-room5x5-v1.1` | `director_room5x5_execution_manifest_v1_1.json` |
| Signed goal progress | `dreamerv3-director-progress-v1` | `director_progress_execution_manifest.json` |
| Offline absolute vs. displacement | `dreamerv3-director-goal-representation-v1` | `goal_representation.json` |
| Offline variance-aware goals | `dreamerv3-director-normalized-goals-v1` | `normalized_goals.json` and `goal_representation.json` |
| Standardized goal learning | `dreamerv3-director-normalized-learning-v1` | `director_normalized_learning_execution_manifest.json` |

Create an isolated worktree from the selected tag and provision the locked
legacy uv profile there. Training commands and protocol notes are in
[DIRECTOR_ROOM5X5.md](DIRECTOR_ROOM5X5.md),
[DIRECTOR_PROGRESS.md](DIRECTOR_PROGRESS.md), and
[NORMALIZED_LEARNING.md](NORMALIZED_LEARNING.md). The offline protocols are
in [GOAL_REPRESENTATION.md](GOAL_REPRESENTATION.md) and
[NORMALIZED_GOALS.md](NORMALIZED_GOALS.md). The tagged source tree is needed
where a launcher checks source provenance. An existing worktree can be reused
if it is clean and at that exact tag.

Pass output paths under the main checkout's ignored `runs/` to the current
training and offline workflows. Archived checkpoints, telemetry, and reports
are inputs or outputs only. The original v1/v1.1 launcher enforces a
checkout-local `runs/` path at its immutable source tag. For exact historical
reproduction, retain that layout for execution and copy the finished output
into the main `runs/` with count and SHA-256 verification. The completed
v1/v1.1 run directories are already available there.

The frozen goal audit uses the checked-in `director_goal_audit.py` entry
point and the exact v1.1 source checkout. The run's recorded launch command
identifies the checkpoint, manifest, and output paths. The
`.experiments/` directory is an optional location for Git worktrees, so no
source snapshot from generated output or untracked worktree files is required.
