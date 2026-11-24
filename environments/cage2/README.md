# Dreamer and CAGE2 on the DGX Spark

This isolated uv project uses Python 3.10.19 and PyTorch 2.9.1+cu130 on Linux
aarch64. The existing Conda environment remains available for MiniHack.
`conda_reference.json` records its package versions, Python, Torch build and GPU.
Torch was installed by pip there, but its original index was not recorded.

## Setup

Working directory on this Spark:
`/home/mspcvsp/Documents/dreamerrl/environments/cage2`.

```bash
cd /home/mspcvsp/Documents/dreamerrl/environments/cage2
bash --noprofile --norc setup.sh
```

The setup initializes the official `cage-challenge/cage-challenge-2` submodule at
`26ce1c1253fa9e2e73f25e6a7f2da32860c11257`, prepares the hash-checked native wheel,
and runs `uv sync --locked`. The Git submodule commit and `uv.lock` together pin
the source and dependencies. `pyproject.toml` reuses the root Dreamer metadata as
an editable dependency. No MiniHack, SB3, AWS emulation, or development extras
are added. CybORG's own declared requirements include pytest and docutils.

The uv executable is pinned to 0.13.9 (`~/.local/bin/uv` on this Spark), and Python
comes from uv's managed installation, not Conda. A new clone needs Git submodule
initialization, which `setup.sh` performs without forcing or discarding edits.

## Locked verification

```bash
bash --noprofile --norc verify_clean.sh --output verification_locked.json
```

This deactivates Conda in its child shell, removes Conda/Python/library-path
contamination, runs `uv sync --locked`, then `uv run --locked python smoke.py`.
It imports Dreamer and CybORG, runs an actual Dreamer actor forward/backward pass
on `cuda:0`, checks finite nonzero gradients, and executes 30 BlueMonitorAgent
steps against the supplied B_lineAgent in bundled Scenario2. Seed 0 gives a Blue
return of -171.7. This is an environment check, not a trained-policy score.

From an already clean shell, the direct commands are:

```bash
uv sync --locked
uv run --locked python smoke.py --output verification_locked.json
uv pip check --python .venv/bin/python
```

## Explicit packaging repairs

The upstream CybORG wheel omits `version.txt` and scenario YAML files. Installing
the pinned submodule's bundled `CybORG` package editable preserves those files;
no manual file copying or changes to the official source are required.

NVIDIA's cuSPARSELt 0.8.0 ARM wheel has an aarch64 filename but an invalid
`manylinux2014_sbsa` internal tag. This makes uv reinstall it on every run and
causes a platform error in dependency checking. `prepare_native_wheel.py` verifies
the upstream SHA256, fixes the platform metadata, adds the visible local version
`0.8.0+spark1`, and regenerates RECORD. The native library is byte-identical.
The deterministic rebuilt wheel's SHA256 is pinned in both the helper and
`uv.lock`; `==0.8.0+spark1` satisfies Torch's public `==0.8.0` requirement. The large
wheel is generated under ignored `wheels/`, not committed. Its provenance JSON
records both wheel hashes and the unchanged native-library hash. Generate it
before locked sync using `setup.sh`; do not edit site-packages.

## Native prerequisites outside the lock

- Linux aarch64 with glibc >=2.28; tested on spark-3f04 with glibc 2.39.
- NVIDIA GB10 and its host driver, tested with 580.95.05. `libcuda.so.1` and normal
  system C/C++ runtime libraries come from the host, not uv.
- Git, Bash, and the pinned uv executable; network access for initial downloads.

CUDA 13 runtime libraries are locked wheels. The installed CUDA toolkit 13.0.3
is not needed for these binary-wheel smoke tests; neither nvcc nor Conda library
paths are used. Stock Torch 2.9.1 reports a capability warning for GB10 (12.1 versus
its advertised maximum 12.0); the verification checks actual CUDA execution and
gradients and does not permit CPU fallback. It does not certify every Torch kernel.

## One real replay learner cycle

```bash
bash --noprofile --norc run_cycle_clean.sh \
  --run-dir /home/mspcvsp/Documents/dreamerrl/runs/cage2_blue_cycle_seed0_20261123
```

Use a new run directory for every rerun. The existing smoke entry point delegates
`--learner-cycle` to `learner_cycle.py`. The job uses the production collector,
replay sampler, vector encoder, world-model learner, and imagination actor/critic
learner. Sixteen collected 30-step episodes prepare replay; the normal trainer's
collection adds a seventeenth episode before one update. Batch size is 16,
sequence length 16, and burn-in 8. The data policy is seeded uniform discrete
warmup. Actor/critic freezing and LR warmup are disabled for this one-update
integration check; AMP/fused optimizers are off. No reward shaping is applied.

The installed ChallengeWrapper returns 52 int64 values and 145 discrete actions;
the adapter converts observations to float32 without changing their values. Its
complete runtime action mapping is saved in `interface.json`. The wrapper's cap
sets `done=True` at 30 while `info['done']` retains the native false flag. The
adapter therefore records `is_last=True`, `is_terminal=False`, preserves the final
transition observation, and resets the next observation separately. The cap is
a continuing-task truncation (`finite_horizon=False`), so continuation bootstraps
across it. If native done metadata is absent, done at the cap is assumed truncation
and earlier done is assumed terminal. No finite-task clock is appended.

The cycle saves a normal v4 checkpoint with replay and all optimizer states,
reloads it in a fresh `uv run --locked` process, then runs 30 greedy Blue steps
with posterior modes and seed 10000. The return is only a pipeline check. It also
records native observation/action/reward alignment and a nonzero reward's path
through replay, sampler, two-hot target, and the production reward loss. Recorded
replay/optimizer state is restored; simulator state starts a new episode on reload,
consistent with the existing checkpoint contract. The shell sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUDA matrix operations.

Targeted adapter checks (independent of the MiniHack pytest configuration):

```bash
uv run --locked pytest -c /dev/null -q test_adapter.py
```
