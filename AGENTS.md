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
