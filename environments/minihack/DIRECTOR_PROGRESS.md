# Signed goal progress in Room 5x5

The reference is the Director v1.1 goal audit in
`runs/director_goal_audit_20261125T234817Z/report.md`. Swapping all eight
codes left greedy behavior unchanged. Goal 7 also started closer to its
decoded target than it finished. This study isolates the worker reward.

The control uses post-action cosine similarity. The opt-in worker reward
is post-action minus pre-action cosine similarity to the same latched
decoded goal. A goal boundary selects the next goal before its first action.
The previous action is scored against the previous goal. Existing
continuation, horizon and worker-target boundary handling still apply.
The manager, RSSM, autoencoder, critic ownership, and code selection keep
their original objectives. Goal-achievement metrics continue to use the
absolute similarity. Version 6 checkpoints store the reward choice in
the saved actor-critic configuration. Older checkpoints default to absolute.

## Protocol

Two fresh pairs use training seeds 19 and 23. Both arms in each pair
initialize the same network tensors and target 1,600 updates, or 204,800
transitions. The supervisor starts at most three workers and queues the
fourth. It checks thermal telemetry and requests checkpoint shutdown at
the 60-minute safety cutoff. Completion and training-monitor evaluations
are reported per worker. Frozen primary comparisons require both arms
of a seed pair at the exact target budget. A separately labeled
update-1,200 fallback is available when both fallback checkpoints exist.
Fallback results do not count toward acceptance.

The final frozen evaluator uses 40 fixed levels disjoint from the
training-monitor seeds. It measures return, natural manager occupancy,
and forced-goal sensitivity. Worker-only sampled interventions hold
the posterior mode fixed and share action uniforms across all codes.
For natural sampled evaluation, actions and posterior latents are sampled.
Goal specificity compares progress toward a target under its assigned
code with progress toward the same target under the other codes at the
same start. Bootstrap intervals resample levels within each training
seed pair. The report presents seed pairs separately.

Point criteria for progress are return at least +0.90, positive assigned-goal
progress and specificity, at least 10% distinct endpoints, median worker action
TV at least 0.05, and at least two effectively used manager goals. The result
is exploratory, with independent uncertainty across only two training seeds.
Corridor R2 remains deferred.

## Execute after formal launch approval

From the clean tagged source checkout, with the reviewed manifest in its
separate Git checkout and an absolute output path under the main `runs/`:

```bash
bash scripts/run_director_room5x5.sh check "$manifest"
tmux new-session -d -s dreamerrl-director-progress -c "$PWD" \
  bash scripts/run_director_room5x5.sh launch "$manifest" "$output"
tmux attach -t dreamerrl-director-progress
tail -f "$output"/job_*/console.log
kill -TERM "$(jq -r .supervisor_pid "$output/execution_plan.json")"
```

The last command requests checkpoint-and-stop. The run automatically writes
the supervisor summary, per-arm logs and checkpoints, learning curves, and
frozen diagnostic report. The launcher verifies that the approved manifest
matches the tagged source and locked legacy environment.
