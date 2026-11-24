# Initial host-aware CAGE2 pilot

Authorized by the user on 2026-11-24: one host-aware run, training seed 0,
507,450 fixed environment interactions (16,899 updates), then stop for manual
review. No flat control, additional seed, held-out evaluation, or follow-on
experiment is launched automatically.

`host_pilot_v1_config.json` records the protocol before training. It uses the
existing host encoder and production Dreamer learner unchanged: 480 real
transitions prefill fresh replay, collect 30 per update, batch 16 x 16, burn-in 8,
imagination horizon 15, FP32, and the inherited 100,000-interaction random-action
warmup. Original LR phase horizons are preserved. Initialization uses the
previously tested fresh reference procedure; no historical weights or optimizer
state are loaded.

Evaluate greedy actions with posterior modes on the same 20 development seeds
at updates 1, 4244, 8462, 12681 and 16899. These correspond to 510, 127800,
254340, 380910 and 507450 interactions. Historical comparisons use the preserved
raw results, with fingerprints in the execution manifest. The final reference
is -35.39. This is exploratory: initialization and the historical collector's
checkpoint-resume seed trajectory differ, there is only one training seed, and
episode-bootstrap intervals do not estimate training-run uncertainty. No ranking
or held-out generalization claim is supported.

Target duration is approximately 2–3 hours. The callback requests a graceful stop
at 3h57m, reserving three minutes for final checkpoint/evaluation/reporting. The
tracked job supervisor enforces a four-hour child process-group limit. If the
wall cap prevents the fixed interaction target, the report marks that outcome as
not matched to the historical final budget. A forced hard kill can prevent final
reporting; intermediate checkpoints and metrics remain available.

After tests, commit the pilot wrapper/configuration, create a new annotated tag,
and prepare/commit a manifest-only follow-up recording that source commit, tag
object, complete tracked-file fingerprints, configs, installed dependencies,
pinned submodule, checkpoint provenance and the user's explicit launch approval.
Run from a clean detached checkout of the source tag with a locked uv environment.
The externally committed execution manifest avoids a self-referential commit hash.
Historical tags and artifacts remain unchanged.

The tracked `launch_host_pilot.py` launches only the `cage2-host-aware-v1` tmux
session. `scripts/run_host_pilot.sh` runs the supervisor and single pilot. It
records launch command/time/PID/exit status and console.log in the new run directory.
No executable source is created under runs/.

Successful completion automatically produces resolved_config.json, manifest.json,
initialization.json, parameters.json, metrics.jsonl (including model, reward,
reconstruction, actor and critic losses and cadence), progress.json, intermediate
checkpoints/evaluation JSON, latest.pt, learning_curve.json/png, scorecard.csv,
final_summary.json, final_report.txt, complete.json, launch.json, exit_code and
reproduce_command.txt. The pilot never selects or launches a subsequent experiment.
