# Proposed obsolete MiniHack artifact cleanup

**Approved and completed on 2026-11-24.** The exact entries below were deleted after inventory and process checks. See [execution record](minihack_cleanup_execution.json).

Paths are relative to `/home/mspcvsp/Documents/dreamerrl`. Each directory includes its entire contents recursively. [Exact nested file list](minihack_cleanup_files.csv) and [machine-readable audit](minihack_cleanup_proposal.json).

| Proposed path | Files | Allocated MiB |
|---|---:|---:|
| `runs/__pycache__` | 1 | 0.012 |
| `runs/action_reward_frozen_update1160_20261122` | 16 | 112.047 |
| `runs/corridor_r2_cost001_seed0` | 7 | 1043.492 |
| `runs/corridor_r2_demo_check_20261122` | 57 | 3166.477 |
| `runs/corridor_r2_fresh_demos_seed0_20261122` | 60 | 5540.383 |
| `runs/corridor_r2_local3x3_handoff_20261123` | 14 | 31.820 |
| `runs/corridor_r2_local3x3_seed0_20261122` | 59 | 5264.539 |
| `runs/corridor_r2_recent_seed0_20261116` | 37 | 2031.809 |
| `runs/corridor_r2_seed0` | 11 | 1743.137 |
| `runs/corridor_r2_weighted_seed0_20261117` | 18 | 1509.129 |
| `runs/repeat_room_seeds.py` | 1 | 0.008 |
| `runs/replication_console.log` | 1 | 0.059 |
| `runs/room_evaluation_modes.json` | 1 | 0.035 |
| `runs/room_replication_results.json` | 1 | 0.027 |
| `runs/room_replication_status.json` | 1 | 0.004 |
| `runs/room_seed0` | 12 | 1652.250 |
| `runs/room_seed0_pilot_status.json` | 1 | 0.004 |
| `runs/room_seed1` | 10 | 1443.320 |
| `runs/room_seed2` | 10 | 1900.531 |
| `runs/stair_direction_frozen_update1160_20261122` | 14 | 38.762 |
| `runs/stair_direction_mlp32_update1160_20261122` | 17 | 1.344 |

Total: **21 entries, 349 files, 26,716,864,512 allocated bytes (24.882 GiB)**.

Estimates include directory blocks and count hard-linked inodes only once. The two hard-link paths are both within the proposed cleanup. Filesystem snapshots/shared storage may change actual recovery. No symlinks are followed.

## Dependency audit

No tracked Python or shell code depends on the candidate run paths or experiment names. Generic `__pycache__` cleanup/exclusion rules are not dependencies on the room replication cache. No active MiniHack trainer or diagnostic process was found. Historical examples in `docs/minihack_offline.md` and `docs/minihack_baseline.md` reference these artifacts; after approved deletion they will require regenerated data or a separate archive. The JSON audit records exact documentation lines.

## Preserved

All four `runs/cage2_*` directories are excluded, including the update-1 checkpoint, metrics, evaluations, provenance, logs and historical source snapshots. The mixed `runs/runs.txt` inventory is preserved because it also records CAGE2. It may contain stale MiniHack paths after an approved cleanup.

Tracked MiniHack environments, tools and regression tests remain. The existing Git stash is untouched. The run cache contains only obsolete room replication bytecode.

## Approval boundary

The user approved this exact table. Inventory, file sizes, allocation, modification times, hard links and active processes were rechecked before deletion. All 63 CAGE2 fingerprints passed afterward. The original proposal JSON and file inventory are retained as approval evidence; the execution record documents completion.
