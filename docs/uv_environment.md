# uv environment management

The root `pyproject.toml` and `uv.lock` are the authoritative maintained development
configuration. Use uv 0.13.9 and managed Python 3.10.19. No activation is required.
Do not use unpinned pip installs. Conda is not a prerequisite and has not been
uninstalled or modified.

## Maintained CAGE2 profile (default)

From the checkout root:

```bash
bash --noprofile --norc scripts/setup_uv.sh
uv sync --locked
uv run --locked python scripts/verify_uv.py --gpu
uv run --locked ruff check .
uv run --locked pyright
uv run --locked pytest -c /dev/null -q environments/cage2/test_adapter.py environments/cage2/test_reproduction_support.py environments/cage2/test_uv_management.py
```

This uses root `.venv`, Gymnasium **1.4.0**, and setuptools **84.2.1**, matching
the historical CAGE2 runtime versions. All shared runtime package versions on
ARM64 match the original baseline. Additional development tools are explicitly
locked. On Spark, Torch remains **2.9.1+cu130**, CUDA runtime **13.0**, NumPy
**1.26.4**, and cuSPARSELt **0.8.0+spark1**. Linux x86_64 CI selects the same public
Torch version's CPU build from the official CPU index; it is not a CUDA benchmark.

Maintained entry points use root uv, including the scripts in `environments/cage2/`:

```bash
uv run --locked python environments/cage2/learning_curve.py --help
uv run --locked python environments/cage2/learner_cycle.py --help
uv run --locked python environments/cage2/report_learning.py --help
```

These help commands do not train. The tracked `run_learning_clean.sh` and
`run_cycle_clean.sh` now synchronize/run the root environment, never the historical
source snapshots under `runs/`. Executing either launcher without `--help` can
start an experiment; no experiment was launched during this migration.

## Explicit legacy MiniHack/NLE profile

NLE **1.3.0** requires Gymnasium **exactly 1.2.0**; it cannot share the default
CAGE2 profile. MiniHack **1.0.2** imports `pkg_resources`, which recent setuptools
removed. Therefore **setuptools==80.9.0 is a runtime pin exclusively in legacy**.
The default profile retains 84.2.1. The baseline's original build-isolation
constraint of 80.9.0 is historical provenance, distinct from installed runtime
versions, and remains unchanged in its original project.

The root lock declares `dev` and `legacy` mutually exclusive. Install legacy into
its own generated directory, keeping the default `.venv` untouched:

```bash
bash --noprofile --norc scripts/setup_uv.sh --legacy
UV_PROJECT_ENVIRONMENT=.venv-legacy uv sync --locked --no-default-groups --group legacy
UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy pytest -q
```

Both profiles share Torch/CUDA/NumPy pins and the same source checkout, but have
separate site-packages. Do not combine `dev` and `legacy`. The extra legacy
packages, Gymnasium 1.2.0 and setuptools 80.9.0 are compatibility differences,
not historical CAGE2 replacements. All remaining shared runtime versions align.
Editor analysis uses the two uv site-packages directories for retained legacy
imports; it never imports packages from a Conda environment. Select root `.venv`
with **Python: Select Interpreter** if the editor remembers an old selection.

## Historical baseline: strict locked reproduction in a detached worktree

The original `environments/cage2/uv.lock` is immutable. Adding modern root groups
and package sources changes uv's dependency metadata freshness checks, even
without changing original package dependencies. Consequently, **do not regenerate
the historical lock, and do not bypass freshness with --frozen**. Install the
original source and lock together in a separate worktree:

```bash
bash --noprofile --norc environments/cage2/setup.sh
bash --noprofile --norc environments/cage2/verify_clean.sh --output /tmp/cage2-baseline-verification.json
```

The tracked `scripts/baseline_uv.py` creates a detached checkout under
`.baseline/cage2_baseline_v1` at commit
`1adc2a0c68f2ce9b361c36a9ac30dddc67f23012`. It verifies the unchanged annotated tag,
clean checkout, pinned submodule, and original lock SHA256:
`45b6f78568e96dec19be2eb9a54579f07a92c653393ae91e0fcdefadf4e056b4`.
It reuses the existing repair helper, copies the generated hash-checked wheel,
and runs **uv sync --locked**. Verification uses **uv run --locked** in that same
historical project. No source is executed from `runs/`, and no historical result
is overwritten. `--output` should point to a new file outside the historical runs.

For manual inspection after setup:

```bash
uv sync --locked --project .baseline/cage2_baseline_v1/environments/cage2
uv run --locked --project .baseline/cage2_baseline_v1/environments/cage2 python -c 'import sys, torch; print(sys.executable, torch.__version__)'
git -C .baseline/cage2_baseline_v1 status --short
```

Historical worktree source, its virtual environment and generated wheels are
separate from active development, gitignored, and disposable caches. Do not edit
that checkout, retarget its tag, or treat it as the maintained development tree.
The full checkpoint provenance, original schedule and user-confirmed OneDrive
backup are in [the reproduction guide](../environments/cage2/reproducibility/README.md).

## Native dependencies and CUDA

`setup_uv.sh` initializes official CybORG revision
`26ce1c1253fa9e2e73f25e6a7f2da32860c11257` and calls the existing
`environments/cage2/prepare_native_wheel.py`. There is one repair implementation.
It changes cuSPARSELt wheel metadata, not native bytes; repaired wheel SHA256 is
`3974cc125c2ad53997dd8384e60ff048f79b36422aa3fdde517afc3aa64d48c0`.
Generate the ignored wheel before locked sync on a new checkout.

Tested host: Linux aarch64 Spark GB10, driver 580.95.05, glibc 2.39. Host
`libcuda.so.1`, normal C/C++ runtime libraries, Git and Bash remain prerequisites.
CUDA runtime libraries are locked wheels; system toolkit 13.0.3 is not required
for these binary-wheel checks. No Conda library path is used. Stock Torch emits
a GB10 capability warning; real GPU forward/backward execution is verified.

CI uses setup-uv 0.13.9 and strict locked commands. Docker uses pinned Python/uv
image versions, the root lock, and the same repair helper. Initialize submodules
before building. The Docker default is the CAGE2 development profile; testing
legacy is a separate explicit profile, not an implicit image dependency.

## Conda audit and optional future removal

No maintained workflow sources activation hooks, requires a Conda executable, or
uses hardcoded Conda interpreters/site-packages. Historical references remain in
`conda_reference.json`, the baseline manifest, immutable Git objects and preserved
run artifacts. These are evidence, not active environment configurations. Generic
shell cleanup may unset inherited `CONDA_*` variables without requiring Conda.
No obsolete environment.yml/conda-lock file exists to remove.

Before any system-wide removal, separately verify other projects and shell
startup files, preserve any unrelated environments, and back up required artifacts.
After explicit approval, remove Conda initialization from shell startup and
uninstall only the installation the user identifies. Then open a fresh shell and
repeat these locked checks. No such system-wide changes are authorized or made
here. Host libraries are not exhaustively locked, and GPU results are not promised
bitwise identical across drivers/hardware. CI/Docker execution must be reported
separately from checks actually run on the Spark.

## Completed verification on 2026-11-24

Strict locked installation and dependency compatibility passed independently for
root CAGE2, isolated legacy, and the clean tagged baseline checkout. Root CAGE2
support tests: 14 passed. Legacy full suite: 289 passed, 1 skipped. Root and tagged
baseline both executed finite GB10 gradients; the historical Blue smoke return
was -171.7. All 63 historical artifact fingerprints still match.

Docker image `dreamerrl:uv-review` built successfully with strict locked sync.
Its lightweight CPU and `--gpus all` GB10 backward checks both passed before
further Docker validation was deferred. No additional container work is queued.
Base images use version tags rather than digests, apt libraries are not locked,
and extended container workloads were not tested. GitHub Actions and x86 runtime
were not executed locally; the x86 locked installation plan passed a dry run.
See [the machine-readable receipt](maintenance/uv_migration_validation.json).
