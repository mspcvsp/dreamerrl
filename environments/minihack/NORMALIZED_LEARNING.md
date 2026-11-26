# Standardized absolute goal learning in Room 5x5

The study uses the update-2,216 world weights from the fingerprinted
Director v1.1 Room 5x5 checkpoint. Both arms use that world in evaluation
mode with gradients disabled. The RSSM, observation model, reward and
continuation heads receive zero optimizer steps. World losses in the log
are forward-only diagnostics. The source policy, critics, goal model,
replay, and optimizer states are excluded. Fresh trainable parameter
tensors are identical within each paired seed. The initial decoded
native goal values can differ under the treatment's inverse transform.

Training-only random trajectories on levels 412100–412139 supply
posterior samples from the frozen world. The calibration uses population
standard deviation with the same 0.01 floor as the offline study. Statistics
are fixed and hash-pinned. The standardized actor checkpoints save them
as buffers. The control's autoencoder uses native RSSM features throughout.
The treatment encodes and reconstructs in standardized coordinates, then
inverse-transforms each decoded goal to native coordinates before the
worker sees it. Both arms compute absolute cosine similarity between the
native latched goal and the post-action native RSSM feature. The manager
task-return objective, architecture and four-action goal interval stay
the same. Signed progress and normalized-space cosine are excluded.

The manifest fixes the two paired training seeds, interaction and optimizer
budgets (2,400 updates and 307,200 transitions per arm and seed), and
three-worker concurrency cap. The shared fallback is update 1,800. FP32 is used for formal training
so AMP overflow cannot change the number of optimizer updates across arms.
Both FP32 and AMP are exercised in verification. The thermal watchdog and
wall-time guard request checkpoint-and-stop. Primary comparisons require both
arms to reach the registered budget. A shared fallback checkpoint is reported
separately if an interruption prevents completion. Corridor R2 stays deferred.

The frozen report uses disjoint seeded levels. It reports native and
standardized goal cosine matrices with the same calibration in both arms.
Worker action TV compares all goal codes at the identical posterior.
Common-random-number sampled interventions keep the starting posterior
and random action draws paired. Goal specificity compares actual progress
toward a given code when that code is assigned with progress toward the
same code under alternative assignments. Return and natural goal occupancy
are measured separately for greedy and fully sampled evaluation. Paired
level bootstrap intervals are conditional on each training-seed pair.

## Reproduction

Run from the clean tagged checkout and its locked legacy profile.
Use the sealed manifest from its separate review checkout, the pinned
normalization archive under main runs, and a new main runs output path.

```bash
bash scripts/run_director_room5x5.sh check "$manifest"
tmux new-session -d -s dreamerrl-director-normalized-learning -c "$PWD" \
  bash scripts/run_director_room5x5.sh launch "$manifest" "$output"
tmux attach -t dreamerrl-director-normalized-learning
tail -f "$output"/job_*/console.log
kill -TERM "$(jq -r .supervisor_pid "$output/execution_plan.json")"
```

The formal launcher requires explicit approval in the reviewed manifest.
