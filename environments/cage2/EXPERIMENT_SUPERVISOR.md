# CAGE2 experiment supervisor

Use `scripts/run_host_aware.sh supervise` to queue independent host-aware
experiment arms. The existing launcher syntax still runs a single invocation.
Each supervised job uses the same `host_aware_experiment.py` entry point and
must pass its existing approved-manifest and tagged-source checks.

From the clean, approved source checkout, set `REPLICATION_MANIFEST` to the
absolute path of the reviewed committed execution manifest. Each `--job`
takes a manifest, configuration arm and seed. Use one entry per independent
run and a fresh output root:

```bash
tmux new-session -d -s cage2-supervised -c "$PWD" \
  bash scripts/run_host_aware.sh supervise \
  --output-root runs/cage2_supervised_example \
  --concurrency 2 \
  --job "$REPLICATION_MANIFEST" cage2_host 1 \
  --job "$REPLICATION_MANIFEST" cage2_host 2 \
  --warn-c 75 --stop-c 80 --sustain-seconds 30
```

The launcher provisions the locked uv profile once. The supervisor starts one
watchdog and verifies a valid GPU temperature below the stop threshold before
starting any trainer. The example temperatures are configurable. Jobs start in
the listed order as slots become available. Each gets its own `job_NNN` with
a console log and separate results, checkpoint, replay and TensorBoard state.
A failed training job is recorded and independent queued jobs can still run.

A sustained thermal stop, watchdog failure, or supervisor interrupt closes the
queue. Active trainers receive a checkpoint-and-exit request through their run
directories. After the `--checkpoint-timeout` window (300 seconds by default),
the supervisor terminates unresponsive process groups, then forcibly stops
groups that remain. Signal the live supervisor PID in `execution_plan.json`
with SIGTERM for a graceful manual stop, or use Ctrl-C in the attached tmux
pane. Hard-killing the supervisor bypasses its cleanup. The watchdog stops
after the supervised workers exit.

`execution_plan.json` records the queue and commands. `thermal.jsonl` records
watchdog telemetry. `execution_summary.json` records each job's terminal
status, exit code, runtime, observed progress and checkpoint outcome.
Aggregate throughput divides summed updates and environment steps by elapsed
wall time from the first launch to the last worker exit. Failed jobs contribute
their last recorded progress. The result includes startup, evaluation,
checkpointing and shutdown time inside that interval. The supervisor returns
0 for complete success, 75 for a thermal stop with successful checkpointing,
1 for a failure, or 128 plus the signal number for a successful signal-driven
shutdown.
