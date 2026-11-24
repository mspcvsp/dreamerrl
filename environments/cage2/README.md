# CAGE2 environment entry points

The **root uv project** is the maintained development environment. See
[uv environment management](../../docs/uv_environment.md) for locked setup,
profiles, tests, native prerequisites, and the detached historical checkout.

```bash
# From the repository root, no activation required:
bash --noprofile --norc scripts/setup_uv.sh
uv run --locked python scripts/verify_uv.py --gpu
uv run --locked python environments/cage2/learning_curve.py --help
```

The original local pyproject.toml and uv.lock are retained unchanged as the
historical baseline dependency contract. Do not synchronize that frozen lock
against modern root metadata. Instead, `setup.sh` provisions the immutable tag
in `.baseline/cage2_baseline_v1`, then performs strict `uv sync --locked` there;
`verify_clean.sh` runs its original GPU/30-step baseline smoke test with strict
`uv run --locked`. This neither trains nor overwrites historical artifacts.

```bash
bash --noprofile --norc environments/cage2/setup.sh
bash --noprofile --norc environments/cage2/verify_clean.sh --output /tmp/cage2-baseline-check.json
```

Maintained `run_cycle_clean.sh` and `run_learning_clean.sh` use the root project.
The tracked `launch_learning.py` writes only generated logs/metadata under runs/;
its `--dry-run` option prints commands without starting training. Successful
learning invokes the tracked reporter. No executable snapshots under runs/ are
used. `--schedule original` replays the recorded two-phase schedule; it requires
the exact original update-1 checkpoint, fresh output paths and training approval.

[Original reproduction record](reproducibility/README.md) contains checkpoint
fingerprints, schedule, dependency provenance and the OneDrive backup record.
`conda_reference.json` is retained solely as historical evidence; no maintained
workflow depends on that installation. The baseline tag remains immutable.
