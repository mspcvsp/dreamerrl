# Two-process replication on DGX Spark

**Recommended default: two independent training processes per Spark**, with one
training seed per process and isolated replay, optimizers and outputs. Use one
process when minimizing the latency of a single result matters. Do not extrapolate
to three processes or vectorized CAGE2; those have not been measured.

This recommendation is an execution policy, not approval for a training experiment.
No formal runs are launched by this document or the finalization tests.

## Measured reference and methodology

Reference: `runs/cage2_concurrency_20261124/results.json`.
Tracked source: `environments/cage2/concurrency_benchmark.py`.
The tracked `concurrency_benchmark_20261124.json` records SHA256 fingerprints of
all 30 saved benchmark files and the exact measured benchmark source.

On 2026-11-24 (15:01 EST), the benchmark ran for 283.81 seconds. It measured a
single worker, then two simultaneous workers, nominally 135 seconds per phase.
Initialization and 16 initial real 30-step CAGE2 episodes occurred before each
phase. Exclude the first five completed updates per worker; both workers share a
measurement window starting after the later fifth update and ending at the common
phase deadline. Count only updates completed inside that window.

Workers use the existing `load_config('cage2_host', seed, output)` and
`initialize_paired`, fresh weights/Adam/replay, the production collector, replay
sampler and world-model/actor–critic updates. Seeds: single=0, concurrent=0/1.
No historical checkpoint is used. Architecture and configured hyperparameters
match the host-aware configuration; **the benchmark keeps the existing 100000-update
calibration scheduler target throughout**, rather than reproducing the formal
pilot's switch to the 16899-update horizon after update 26. This is a throughput
measurement, not a learning-curve reproduction or comparison of returns.
All processes use two Torch/OpenMP/MKL/OpenBLAS threads. The existing formal shell
launcher uses OpenBLAS=1; it is preserved unchanged, so sustained launch timings
need not exactly match this benchmark.

Production cadence timers synchronize CUDA at phase boundaries. Linux CPU,
MemAvailable, swap allocation and swap-page I/O counters are sampled about once
per second alongside `nvidia-smi` utilization/temperature. GPU memory usage is
unavailable through nvidia-smi on this unified-memory device; Linux available
memory is reported instead. CPU percentages distinguish whole-machine busy time
from process CPU time (100% = one logical CPU).

| Metric | One process | Two processes |
|---|---:|---:|
| Scored updates | 282 | 398 (199 each) |
| Scored wall seconds | 132.40 | 131.48 |
| Aggregate updates/sec | 2.12994 | 3.02716 |
| Collection seconds/update/worker | 0.2383 | 0.3251 |
| World-model seconds/update/worker | 0.1166 | 0.1591 |
| Actor–critic seconds/update/worker | 0.1102 | 0.1699 |
| Mean GPU utilization | 16.84% | 64.55% |
| GPU mean / peak temperature | 48.61 / 51°C | 54.57 / 57°C |
| Mean host CPU busy | 5.21% | 7.53% |
| Mean aggregate worker CPU, one core=100% | 97.23% | 145.97% |
| Minimum available unified memory | 112.29 GiB | 110.28 GiB |
| Maximum swap used / swap-in / swap-out | 0 / 0 / 0 | 0 / 0 / 0 |

Aggregate throughput rose **42.12%**. Each concurrent worker is slower, but two
16899-update jobs extrapolate to **3h06m together**, versus **4h24m sequentially**
(29.64% less elapsed time). This excludes evaluation, startup and checkpoint I/O.

Limitations: one short ordered trial, no repeat/order randomization or confidence
interval; different thermal state and warmed caches between phases; only one
architecture, batch size and two-thread CPU setting; small fresh replay rather
than full long-run memory pressure. Evaluation overlap and simultaneous checkpoint
writes were not measured. Temperature 57°C and available memory 110.3 GiB are
observations, not sustained guarantees. The 300-second budget is cooperative;
external blocked telemetry/filesystem calls are not protected by an OS watchdog.
No persistent benchmark model checkpoints or policy-quality claims are produced.

## Reproduce the benchmark (only when separately requested)

From the repository root, select a fresh output directory:

```bash
uv sync --locked
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
uv run --locked python -u environments/cage2/concurrency_benchmark.py \
  --run-dir runs/cage2_concurrency_reproduction
```

The source refuses an existing output directory. Results, per-worker configs,
JSONL updates, completion counters, telemetry, source diff and rerun command are
under that directory. The original supervisor log is preserved as `console.log`.
The benchmark source fingerprint matches its original manifest; finalization
has not edited or recomputed original artifacts in place.

## Prepare a formal replication pair

1. Review a fixed-interaction configuration, independent seeds and explicit
   execution/evaluation schedule. Retain the same development seeds across runs;
   keep held-out seeds out of selection. Do not select runs by wall-clock finish
   or treat correlated evaluation episodes as independent training seeds.
2. Commit reviewed experiment source/configuration, use a **new** annotated
   experiment tag and prepare a reviewed, committed execution manifest with the
   existing `prepare_host_manifest.py`. Obtain explicit launch approval. The
   manifest must state the intended independent pair/subset and training budget.
   Never reuse or update original baseline/pilot tags or archived manifests.
3. Run from the clean, exact tagged checkout with clean pinned submodules and its
   locked root uv profile. The manifest can live in a separate committed checkout
   as supported by `verify_gate`; it cannot live only under runs/. Use the existing
   uv/native-wheel setup. Both processes can share this read-only source checkout
   and installed environment; training state remains process-local. Sync once
   before starting either process.
4. Use `scripts/run_host_aware.sh` with paired `--arm` and `--seed` selectors.
   Without selectors the original full sequential schedule is unchanged. A
   selected worker executes the unchanged `run_arm` function, including its fixed
   budget, checkpoints and development evaluations. It writes
   `execution_selection.json`; it does not automatically inspect held-out seeds
   or generate a full multi-arm comparison. Perform the preregistered final
   held-out/reporting step only after the complete approved comparison exists.
   The old `launch_host_pilot.py` has a fixed seed/session and is intentionally
   not repurposed for independent replications.

Before launch, this read-only gate check validates selection and provenance:

```bash
uv run --locked python environments/cage2/host_aware_experiment.py \
  --manifest "$REPLICATION_MANIFEST" --output-root runs/unused_gate_check \
  --arm cage2_host --seed 1 --check-only
```

The current historical manifests do **not** authorize new current-source jobs;
expected fingerprint failures must not be bypassed.

## Detached execution procedure (for a subsequently approved pair)

Set `REPLICATION_MANIFEST` to the absolute path of the reviewed committed manifest.
The example launches independent host-aware seeds 1 and 2, already in the existing
configuration. Use fresh directory/session names for each new approved study.
For a controlled flat/host comparison, pair arms by training seed and retain all
configured budgets/evaluation settings; do not substitute the historical baseline.

```bash
EXPERIMENT_ROOT=$(git rev-parse --show-toplevel)
REPLICATION_MANIFEST=/absolute/path/to/reviewed_execution_manifest.json
uv sync --locked

for seed in 1 2; do
  session="cage2-rep-seed${seed}"
  job_root="$EXPERIMENT_ROOT/runs/replication_seed${seed}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session already exists: $session"
    exit 1
  fi
  mkdir "$job_root" || exit 1
  tmux new-session -d -s "$session" bash -c '
    cd "$1" || exit 1
    bash scripts/run_host_aware.sh \
      --manifest "$2" --output-root "$3/results" \
      --arm cage2_host --seed "$4" > "$3/console.log" 2>&1
    job_exit=$?
    printf "%s\n" "$job_exit" > "$3/exit_code"
    exit "$job_exit"
  ' _ "$EXPERIMENT_ROOT" "$REPLICATION_MANIFEST" "$job_root" "$seed"
done
```

This is a tmux shell recipe around the existing tracked launcher, not a new job
framework. The console directory is distinct from the launcher's fresh results
root. No shared writable replay, optimizer, checkpoint or TensorBoard directory.
SSH/VS Code disconnection does not stop tmux. Each worker's exit code survives
session termination; nonzero status requires investigation, never an automatic
retry. A detached session alone is not evidence of optimizer progress.

## Monitor sustained work

The [DGX Spark thermal watchdog](THERMAL_WATCHDOG.md) can supervise both
selected workers from its own tmux session and request a final checkpoint.

After detaching, confirm both logs advance, world/actor/critic counters advance,
and the first `latest.pt` is readable before treating startup as successful.
Read-only model/replay inspection must not append to training replay. Use:

```bash
tmux list-sessions
tmux attach -t cage2-rep-seed1
# Detach with Ctrl-b, then d.
tail -f runs/replication_seed1/console.log
cat runs/replication_seed1/exit_code
cat runs/replication_seed2/exit_code
```

An absent exit_code while a process is active means unfinished, not success.
Compare each results subtree's `complete.json` with expected update/env-step
budgets and development artifacts. `run_arm` automatically writes a final
checkpoint/complete.json, while the shell records success/failure. A vanished
tmux session with no exit file is abnormal and must not trigger duplicate training.

Production `metrics.jsonl` already contains `update`, `env_steps`, optimizer-step
cadence, collection/world/actor–critic seconds and per-update timings. Inspect both:

```bash
tail -n 1 runs/replication_seed1/results/cage2_host_seed1/metrics.jsonl
tail -n 1 runs/replication_seed2/results/cage2_host_seed2/metrics.jsonl
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv -l 5
vmstat 5
grep -E 'MemAvailable|SwapTotal|SwapFree' /proc/meminfo
```

Ignore vmstat's first since-boot average. Later `si/so` columns show ongoing swap
I/O; SwapFree changes show allocation, not necessarily current thrashing. Save
monitor output under the corresponding generated study directory if needed.
Record resource trends over full runs, including evaluation/checkpoint overlap.

For real aggregate wall throughput including pauses, sample both update counters
over the **same** wall interval. The following read-only example waits 60 seconds:

```bash
uv run --locked python - <<'PY'
import json
import time
from pathlib import Path
paths = [Path(f"runs/replication_seed{s}/results/cage2_host_seed{s}/metrics.jsonl") for s in (1, 2)]
def updates(path):
    # Skip a possibly incomplete final write.
    for line in reversed(path.read_text().splitlines()):
        try:
            return int(json.loads(line)["update"])
        except (ValueError, KeyError):
            continue
    raise RuntimeError(f"No completed updates in {path}")
start = [updates(p) for p in paths]
t0 = time.monotonic()
time.sleep(60)
end = [updates(p) for p in paths]
seconds = time.monotonic() - t0
print({"updates_per_worker": [b-a for a,b in zip(start,end)],
       "aggregate_updates_per_second": sum(b-a for a,b in zip(start,end))/seconds})
PY
```

For long logs, sample recent tail rows rather than repeatedly reading the entire
file. Do not sum CUDA phase reciprocals as if they were disjoint device work:
phase durations overlap across processes. If sustained aggregate speedup disappears
or swap becomes persistent, prefer sequential execution pending diagnosis.
No automatic budget/configuration changes are warranted from this short benchmark.

## Finalization verification

Finalization ran no training or benchmark workloads. **17 lightweight tests
passed**: original execution order, valid/invalid single-arm selectors, mocked
single-arm CLI routing with no held-out access, exact timing recomputation on
temporary copies, all 30 artifact fingerprints, paired initialization/provenance
gates and existing pilot reporting/supervision checks. Ruff and repository Pyright
passed. Shell examples passed `bash -n` without execution, and launcher `--help`
confirmed the supported arguments. The learner, sampler, models, configuration,
dependency pins and measured benchmark source remain unchanged by finalization.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
uv run --locked python -m pytest --rootdir=. --confcutdir=. \
  --import-mode=importlib --noconftest -c /dev/null \
  -o pythonpath=/home/mspcvsp/Documents/dreamerrl/environments/cage2 \
  -p no:cacheprovider -q \
  environments/cage2/test_concurrency_workflow.py \
  environments/cage2/test_host_experiment_support.py \
  environments/cage2/test_host_pilot.py
```

Artifact-dependent tests skip on machines without the original Spark run;
selection/gate tests remain runnable. Formal independent-seed launch, sustained
resource behavior, evaluation overlap and restart recovery have not been exercised
by this documentation-only finalization. The new selectors do not add resume or
supervision features; preserve stopped jobs and diagnose failures with existing
checkpoint tools before planning a separately authorized restart.
