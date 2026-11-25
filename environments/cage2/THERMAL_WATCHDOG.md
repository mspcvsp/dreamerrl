# DGX Spark thermal watchdog

The watchdog supports concurrent single-arm runs of
`host_aware_experiment.py`. Launch the approved workers using the
[concurrent replication procedure](CONCURRENT_REPLICATION.md) from a source
revision containing the thermal stop integration. Once both fresh arm
directories exist, start the watchdog in its own detached tmux session:

```bash
tmux new-session -d -s cage2-thermal -c "$PWD" \
  uv run --locked --no-sync python -u environments/cage2/thermal_watchdog.py \
  --run runs/replication_seed1/results/cage2_host_seed1 \
  --run runs/replication_seed2/results/cage2_host_seed2 \
  --warn-c 75 --stop-c 80 --sustain-seconds 30 \
  --log runs/replication_thermal.jsonl
```

Run this from the repository root after the locked uv profile is provisioned.
Use a fresh log path. The temperatures above are example operating thresholds.
GPU index 0 is the default. Use `--gpu` to select a different index or UUID
when appropriate. List each selected worker's arm directory with its own
`--run` argument.

The watchdog polls every 10 seconds. Its JSONL samples record GPU temperature,
utilization, free GPU memory and hardware/software thermal slowdown indicators
where available, plus Linux `MemAvailable` for Spark's unified memory.
Missing fields are null. Each threshold has its own elapsed-time window.
With the defaults, a violation requires four consecutive qualifying samples
spanning 30 seconds. A lower or missing reading resets the corresponding
window. Warnings are logged once per sustained crossing.

At a sustained stop threshold, the watchdog publishes a request inside each
selected unfinished arm directory. The trainer cooperates at a collection or
update boundary, or after an in-flight development evaluation. It atomically
saves `latest.pt`, writes `thermal_stop_status.json` with the reason and
checkpoint outcome, and exits with code 75 after a successful thermal
checkpoint. The interrupted arm has no `complete.json`. The experiment
launcher ends before subsequent arms or held-out evaluation.

The watchdog exits when all selected arms finish normally or after recording
the outcomes of a thermal stop. Inspect the `stop_requested` and
`final_checkpoint` events in the log along with each worker's exit code.
A successful checkpoint is confirmed by the matching request ID. A worker
that finishes its normal budget during shutdown is logged separately.
A failed request, failed checkpoint or missing acknowledgment yields watchdog
exit code 1. Use `--checkpoint-timeout` to change the default 300-second
acknowledgment window. A timeout leaves the request in place so the trainer
can still honor it when it reaches a safe boundary.
