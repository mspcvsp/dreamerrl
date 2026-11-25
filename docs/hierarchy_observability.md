# Hierarchical training observability and offline goal probes

## Logging audit

| Evidence | Where to find it | Interpretation |
| --- | --- | --- |
| Manager and worker actor and critic losses | `metrics.jsonl` `credit_assignment` and TensorBoard `ac/*` | Critic losses include the corresponding slow-value term when enabled. |
| Gradient norms and optimizer steps | `world_model_grad_norm`, `credit_assignment` actor/critic norms, `cadence` and `cadence_total` | Norms are pre-clip and unscaled under AMP. Actor and critic each have one optimizer shared by their two levels. Nonfinite hierarchy scalars are null in JSON and skipped in TensorBoard. |
| Goal occupancy and entropy | `goal_<id>_decisions`, `goal_entropy`, `goal_active_count` | Counts and empirical entropy use live imagined manager decisions. Sum counts across updates before computing run-level occupancy or entropy. |
| Embedding norms and temporal abstraction | `goal_embedding_norm_{mean,max}`, `goal_persistence`, `goal_boundary_repeat_rate`, observed/configured interval and full interval counts | Persistence uses live adjacent imagined steps. Boundary repeats distinguish goal reselection from scheduled persistence. Use pair counts to weight pooled rates. Observed intervals include censored tails. |
| World-model losses | `metrics.jsonl`, TensorBoard `wm/*` | Existing reward, reconstruction, continuation, KL and auxiliary metrics. |
| Evaluation returns | Scheduled `development_*.json`, `development_curve.json`, and hierarchy TensorBoard `evaluation/development_return` | Evaluation uses the launcher's existing schedule. The offline probe also reports native POPGym returns. |
| Throughput, elapsed time and run identity | Update timing and samples/sec in `metrics.jsonl`, resolved config and supervisor summary | Hierarchical records carry training and environment seeds, run name and TensorBoard root. `train_call_elapsed_seconds` resets at each call to `train`. Supervisor throughput uses shared wall time across workers. |
| Checkpoint and stop outcome | Run `complete.json`, thermal/shutdown acknowledgment and supervisor summary | Normal completion records the final saved path and reason. Cooperative interruptions record the request and checkpoint outcome. A crash without an acknowledgment remains failed or unconfirmed. |

The added training reductions run once per update on existing imagination
tensors. They exclude masked starts and dead positions and do not sample any
new actions. There are no new objectives or optimizer changes. Flat-mode
metrics and the training update are preserved. Run-level diagnostics for real
goal occupancy and embedding redundancy belong to the offline probe.

## Offline methodology

Use a trusted hierarchical POPGym checkpoint. The collector creates a fresh,
single-row environment, freezes the policy and posterior, and restores mode
flags and Python, NumPy and Torch RNG state. It pairs each manager decision
with a future posterior in the same episode, including a valid final
transition observation. The default lag is one configured manager interval.
The report measures native evaluation returns on the same independent rollout
episodes and keeps all decision counts, including decisions too close to an
episode end to provide a prediction target.

A deterministic 60/20/20 episode split separates fit, validation and test.
The probe fits up to four latent-change principal axes using fit episodes,
fits ridge models on that split, tunes regularization on validation
episodes and scores only held-out test episodes. Controls use latent only,
observation only, time only, and all three jointly. Goal-only and joint
controls plus one-hot goal models complete the comparison. The report
includes held-out RMSE, MAE, R², episode-bootstrap RMSE intervals, and paired
incremental intervals. Added-goal minus control MAE favors negative values.
The intervals describe variation across held-out episodes conditional on the
fitted probe. They do not quantify uncertainty across training seeds.

Occupancy, entropy and effective goal counts flag potential collapse. A goal
with at least 90% of decisions triggers the collapse heuristic. Near-identical
vectors or cosine similarity of at least 0.95 flag potentially redundant
embeddings. These are descriptive thresholds. Predictive gain can reflect
information already present in the state, policy selection or model
misspecification. Establishing causal controllability would require a separate
intervention study with goal assignments at matched states.

## Reproduce

From the repository root, after provisioning the locked UV profile:

```bash
output=runs/hierarchy_probe_example
uv run --locked --no-sync python environments/cage2/hierarchy_diagnostics.py \
  --checkpoint /path/to/trusted/popgym/latest.pt --output "$output" --episodes 60
```

Run the bundled regressions, checkpoint probes and CPU logging-overhead
microbenchmark with `environments/cage2/run_hierarchy_diagnostics.py`. It
accepts an existing fresh output directory under `runs/` and repeatable
`--checkpoint` arguments. The run directory holds the exact detached launch
command and generated `final_report.json`.

The collector currently covers POPGym's discrete observation and action
interface. The CPU overhead estimate measures the reducer and incremental
JSON serialization. GPU synchronization and TensorBoard disk I/O require an
end-to-end production-profile measurement. These checks establish
observability and diagnostic correctness on small workloads. They do not
establish production learning stability or validate a new formal run.
