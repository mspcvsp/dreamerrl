# Research and experiment artifacts

TTCP CAGE Challenge 2 Blue is the primary research environment. Historical
MiniHack experiments are OBE; retain version-controlled environments and
regression tests unless the user explicitly requests their removal.

## Source of truth

- Authoritative experiment entry points, launchers, evaluation/reporting code,
  configurations and reproducibility manifests belong in version-controlled
  project paths, such as `environments/cage2/` and `dreamerrl/`.
- Never place authoritative executable source or configuration only under
  `runs/`. Do not import, execute, source or load executable code from `runs/`.
  Historical source snapshots there are evidence, not supported entry points.
- Before launching an experiment, ensure its source/configuration/manifest are
  tracked and reviewable. Do not commit without user authorization.
- Keep generated checkpoints, replay, metrics, logs, TensorBoard data and run
  snapshots gitignored. Keep pinned dependency declarations and lockfiles tracked.
- Record the source revision/diff, resolved configuration, seeds, execution and
  evaluation schedule, dependency provenance and source-checkpoint fingerprints.
  For exact reproduction, use the recorded schedule rather than a new timing budget.

## Preservation and cleanup

- Preserve the immutable `cage2_baseline_v1` tag and historical CAGE2 artifacts,
  especially the original update-1 checkpoint and its independent backup record.
- Propose exact obsolete paths and disk usage, audit maintained references and
  active processes, then await explicit user approval before deleting artifacts.
- Never infer a verified backup from a filename or user statement: distinguish
  user-confirmed location from independently hash-verified bytes.
- Do not alter historical results to make reproduction checks pass. Document
  limitations and verify preserved fingerprints after changes.

## Environment management

- Use uv, tracked pyproject.toml declarations and uv.lock files exclusively for
  maintained dependency management. Use `uv sync --locked` and `uv run --locked`;
  do not use unpinned pip installs or activation scripts.
- The root `.venv` is the maintained CAGE2 development profile. Keep Torch/CUDA
  versions aligned with the historical baseline. Never upgrade pins just to
  resolve a profile conflict without reporting that conflict first.
- NLE 1.3.0 requires Gymnasium 1.2.0; baseline/CAGE2 uses 1.4.0. The root `dev`
  and `legacy` groups are mutually exclusive. Select legacy explicitly with
  `UV_PROJECT_ENVIRONMENT=.venv-legacy uv sync --locked --no-default-groups --group legacy`.
- Preserve the original environments/cage2/uv.lock unchanged. Verify historical
  reproduction with the detached immutable-tag checkout under `.baseline/`,
  provisioned by `scripts/baseline_uv.py`; do not relax --locked or regenerate
  the historical lock against modern root metadata.
- Reuse environments/cage2/prepare_native_wheel.py for cuSPARSELt. Keep generated
  native wheels and all uv environments/worktrees ignored.
- Do not require or source Conda in active workflows. Historical environment
  records may mention it as evidence. Do not uninstall it system-wide without
  explicit user approval.
