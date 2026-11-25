# Director-inspired DreamerV3 on MiniHack Room 5x5

The flat and existing hierarchical policies keep their original objectives and
checkpoint schemas. Director is opt-in with `--actor-mode director`. The
RSSM and all world-model losses are shared across the three arms.

A categorical autoencoder reconstructs detached posterior features from replay.
The target concatenates the RSSM deterministic state and flattened stochastic
factors. A straight-through one-hot code with a soft marginal entropy bonus
provides encoder gradients and encourages code utilization. Reconstruction and
goal decoding run in FP32. This first design uses eight categorical codes,
a 16-unit bottleneck network hidden layer, and the existing four-action interval.
The manager chooses one code at each boundary. The worker sees the decoded
feature vector and remaining phase. The vector stays fixed during an interval,
including across learner updates. Termination and truncation reset the collector's
goal state.

The autoencoder optimizer owns its encoder and decoder exclusively. The actor
optimizer owns the manager and worker policies. The critic optimizer owns both
value heads. Manager targets use predicted task reward and semi-Markov discount
products. Worker targets use cosine similarity between the latched vector and
the predicted next RSSM feature. Worker returns stop at goal boundaries. The
manager tail uses the task critic, including at a partial final interval.
World-model gradients are restricted to the existing world-model update.
Checkpoint v6 serializes the additional optimizer. Existing versions 4 and 5
retain their layouts.

This is a simplified adaptation of Hafner et al. It uses one categorical code
instead of the paper's factored discrete VAE and uses a marginal code-usage
bonus without a manager exploration reward. It retains DreamerV3's existing
REINFORCE losses, worker phase input, slow critics and return normalization.
Similarity uses the full RSSM feature and can be dominated by predictable
features. Episode goal completion counts a goal if any actual next posterior
in its interval reaches cosine similarity 0.9. The report includes ended and
time-limited intervals, with censoring recorded in episode logs. Goal
similarity and code occupancy alone provide limited evidence about semantic
meaning and controllability.

[Hafner et al., Deep Hierarchical Planning from Pixels](https://proceedings.neurips.cc/paper_files/paper/2022/file/a766f56d2da42cae20b5652970ec04ef-Paper-Conference.pdf)

## First comparison

Three fresh arms use the same fixed seed value in independent processes and
checkpoints. Each targets 2,400 updates with 128 new transitions per update.
Greedy evaluation uses the same 20 seeded levels per checkpoint every 100
updates. The 60-minute supervisor cutoff and thermal stop can censor the
schedule. The consolidated report uses the last evaluation budget all three
arms reached. Per-episode variation describes one fitted policy in each arm.

The existing MiniHack monitor records evaluations and per-update metrics. The
existing offline replay posterior diagnostic measures latent use on each final
checkpoint. Goal reconstruction loss, encoder usage, manager code occupancy,
cosine reward, policy and value losses, gradient norms and optimization cadence
are in metrics JSONL. Per-arm console logs, checkpoints, and configs stay in
their own subdirectories. Thermal telemetry records shared GPU and host load.

## Reproduce after approval

Provision the clean tagged source checkout using the repository's locked legacy
profile and pinned native wheel. Keep the reviewed execution manifest in a
separate Git checkout. To check it and launch from the tagged source checkout:

```bash
bash scripts/run_director_room5x5.sh check "$manifest"
tmux new-session -d -s dreamerrl-director-room5x5 -c "$PWD" \
  bash scripts/run_director_room5x5.sh launch "$manifest" "$output"
```

Use an absolute manifest path and an absolute, fresh output path under the
tagged checkout's `runs/`. The launcher requires the manifest's recorded
approval. Launch preflight checks available memory, temperature and disk space
and sets concurrency up to three. For live monitoring and graceful stop:

```bash
tmux attach -t dreamerrl-director-room5x5
tail -f "$output"/job_*/console.log
kill -TERM "$(jq -r .supervisor_pid "$output/execution_plan.json")"
```

The supervisor requests checkpoints at update boundaries on stop, then writes
its summary. The launcher generates `final_report.md`, `final_report.json` and
`learning_curves.png`, including available partial results.
