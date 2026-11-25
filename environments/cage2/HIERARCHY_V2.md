# Hierarchical CAGE2 v2 execution and readiness

The tracked protocol extends the host-aware configuration with eight goals,
16-dimensional goal embeddings, and a four-action interval. It uses fresh
models, replay and optimizers for seeds 0, 1 and 2. The final budget is 16,899
updates and 507,450 interactions per seed. Development evaluations follow the
historical schedule. The historical −28.64 score is a one-seed reference.

After review and separate launch authorization, commit the implementation and
create an annotated `dreamerrl-hierarchy-v2` tag. From that clean source
checkout, use `prepare_host_manifest.py --hierarchy --tag dreamerrl-hierarchy-v2`
with an output path in a separate Git checkout. Record the explicit approval
with `--launch-approval` and commit the manifest in that separate checkout.
The tagged source, dependency inventory, sealed base and hierarchy
configuration hashes, and approved manifest are enforced by the existing gate.

Build the launch plan with
`bash scripts/run_hierarchy_v2.sh --manifest "$manifest" --output-root "$run" --check-only`.
This has no training side effects. After reviewing the plan, remove
`--check-only` only for an authorized formal launch in tmux. Give the launch
a new run directory under `runs/` and keep its outer console and exit file
outside the supervisor's output root, which must start absent. The wrapper
uses locked uv, the existing supervisor, one thermal watchdog and three worker
slots. The supervisor requests stop after 24,900 seconds and allows 280
seconds for checkpoint acknowledgments and two more seconds for termination.
Final report I/O follows process shutdown.

Use the supervisor PID in `execution_plan.json` with SIGTERM for a cooperative
stop. Each worker writes its resolved configuration, initialization record,
metrics, checkpoints and evaluations in its own directory. The launcher
reports per-run and aggregate throughput, aggregate thermal and memory samples,
checkpoint SHA-256, and which seeds completed the final evaluation and budget.
Only a completed, acknowledged run with a final checkpoint contributes a
matched-budget return. Fingerprints identify the saved bytes. Reloading a
production hierarchy checkpoint on the target hardware remains a startup and
operational verification item. The offline representation collector covers
POP Gym. Full-size three-worker CAGE2 throughput is still unmeasured.

For synthetic and POP Gym readiness checks, run
`bash scripts/verify_hierarchy_v2.sh "$verification_root"` in a fresh existing
directory under `runs/`. The verification command writes a pytest log, exit
status and final report. It uses no formal training or real thermal trip.
