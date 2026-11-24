# Original CAGE2 baseline reproduction record

This is a retrospective record of `cage2_blue_learning_seed0_20261123`, not a new
experiment. It supplements the immutable `cage2_baseline_v1` tag. Neither that tag
nor any historical run file is changed. Paths below are relative to the checkout
unless absolute. Original checkout: `/home/mspcvsp/Documents/dreamerrl`.

## Source and starting checkpoint

The baseline tag resolves to `1adc2a0c68f2ce9b361c36a9ac30dddc67f23012`.
The original `revisions.json` records `89607e9100ba157e193aeb275f5522b9d2f62971`:
the experiment scripts were untracked at launch and committed unchanged later.
The empty historical `code.diff` omitted those untracked files. Their saved
snapshots and recorded hashes match the tagged scripts byte-for-byte. There is
no learner, architecture, reward or dependency drift between those commits.

Required starting checkpoint:

- Location: `runs/cage2_blue_cycle_seed0_20261123/latest.pt`.
- Original absolute path:
  `/home/mspcvsp/Documents/dreamerrl/runs/cage2_blue_cycle_seed0_20261123/latest.pt`.
- SHA256: `94f378899c0422e0af2db2e07cbb13eab9b27a1448cfbe2deb955617c0e5d2ff`.
- Provenance: the successful seed-0 collect/train/reload cycle at commit `89607e9`,
  against bundled Scenario2 and supplied B-line; its `provenance.json`,
  `checkpoint_expectation.json`, `reward_trace.json`, and `verification.json`
  remain in the original one-update run directory.
- Format: project v4 checkpoint, update 1, 510 transitions, 17 replay episodes
  containing 527 observation rows. Model, actor and critic optimizer counters
  are all 1. Preserve world/actor/critic/slow-critic weights, all optimizer states,
  replay, configuration, counters, and Python/NumPy/Torch/CUDA RNG states.
  `original_baseline.json` lists all required checkpoint keys and provenance.

Check the hash before loading this trusted pickle checkpoint. It is not part of
Git. A fresh run of the one-update producer is not a substitute for this exact
checkpoint when reproducing the historical continuation. Back it up separately;
no durable external archive or download location is currently recorded.

## Configuration and exact execution schedule

`resolved_config.original.json` preserves the original resolved configuration.
`original_baseline.json` contains machine-readable source and dependency hashes,
checkpoint requirements, schedule, seeds, environment variables, reporting
contract, and a complete SHA256 inventory of the 63 historical run/job files.
The resolved configuration is inherited from the starting checkpoint; current
project defaults are not a replacement. Relocating output paths is harmless.

The interface is 52 int64 observation values converted to float32, 145 discrete
Blue actions, and 30 steps against `B_lineAgent` in Scenario2. The time cap is a
truncation with continuation bootstrap, not a native terminal state. The runtime
action mapping is preserved in the original `action_mapping.json`.

The original sequence was:

1. Restore update 1, replay, optimizer state and RNG state. The resume contract
   starts a fresh simulator episode at seed 510; original replay contains seeds
   0–16. Subsequent training episode seeds increment by one.
2. Before learning, evaluate actor/update-1, sleeping Blue and seeded random Blue
   on seeds 1000000000–1000000019. Use one unassisted 30-step episode per policy
   per seed. Actor actions are greedy with posterior modes. Policy seed is
   simulator seed + 100000; random actions use an independent NumPy generator.
   Isolate evaluations from training RNGs, replay and model state.
3. Complete updates 2–26 using `trainer.train(100000, ...)`, stopping at 26.
   The 25-update calibration measured 0.41584069735952656 seconds/update.
   Save `update_26.pt`; calibration latest-checkpoint interval is 25 updates.
4. Complete updates 27–16899 using `trainer.train(16899, ...)`. Save latest every
   100 updates and immutable evaluation checkpoints at 4244, 8462, 12681, 16899.
   Their cumulative environment steps are 127800, 254340, 380910, 507450.
   Evaluate the same 20 seeds at each checkpoint.
5. Use the predeclared final checkpoint for the main conclusion. Bootstrap the
   20 matched seed differences with 20000 resamples, NumPy generator seed
   20261123, percentile 95% intervals. Intermediate selection is exploratory.

The normal trainer assigns a shared cosine LR to all three optimizers. Its
zero-based scheduler index is completed update minus one, base LR is 0.0003,
warmup is zero and floor is 10% of base. Matching both scheduler horizons above
matters; training to a different total changes the LR trajectory.

**The tagged `run_learning_clean.sh` is a timing-adaptive experiment launcher,
not an exact fixed-schedule replay launcher.** It recalculates the final target
and has no fixed-budget option. The historical `rerun_command.txt` reproduces
that procedure, not necessarily the original schedule. An exact continuation
must consume the saved schedule above rather than remeasure timing. This
change adds documentation and a manifest, not a new training launcher.
The nominal budget was 7200 seconds; actual recorded runtime was
7814.591335489997 seconds. Do not force a two-hour wall-time cutoff when matching
the original update budget.

## External dependency preparation

Preparation commands below are documented for a future reproduction; they were
not executed as part of this documentation change:

```bash
cd /home/mspcvsp/Documents/dreamerrl/environments/cage2
bash --noprofile --norc setup.sh
```

This initializes the official Git submodule at
`26ce1c1253fa9e2e73f25e6a7f2da32860c11257`, uses its bundled CybORG 2.1 package
editable (retaining scenario/version files), prepares the repaired native wheel,
and runs locked sync. Use uv 0.13.9, managed Python 3.10.19, Torch 2.9.1+cu130 and
the unchanged `uv.lock`:
`45b6f78568e96dec19be2eb9a54579f07a92c653393ae91e0fcdefadf4e056b4`.

The generated, ignored wheel is
`nvidia_cusparselt_cu13-0.8.0+spark1-py3-none-manylinux2014_aarch64.whl`.
Its SHA256 is
`3974cc125c2ad53997dd8384e60ff048f79b36422aa3fdde517afc3aa64d48c0`.
`prepare_native_wheel.py` verifies upstream SHA256
`400c6ed1cf6780fc6efedd64ec9f1345871767e6a1a0a552a1ea0578117ea77c`,
repairs the erroneous `sbsa` metadata tag to `aarch64`, adds the local version
and rebuilds RECORD deterministically. Native library bytes are unchanged:
`libcusparseLt.so.0` SHA256 is
`20de6b36cbbf5753b26826eec753a2005608e14cbc3bfdd4681138098be86941`.
Full upstream URL and provenance are embedded in the manifest.

Tested host: spark-3f04, Linux aarch64, NVIDIA GB10, NVIDIA driver 580.95.05,
glibc 2.39 (documented wheel minimum 2.28). Host `libcuda.so.1` and system C/C++
runtime libraries are outside uv.lock. Git, Bash and initial download access
are needed. CUDA runtime libraries are locked wheels; installed toolkit 13.0.3
was not required. Run without Conda or injected Python/library paths; keep the
existing Conda environment available for MiniHack. Exact launcher variables
are recorded in the manifest. Stock Torch warns about GB10 capability 12.1;
the original GPU checks and training passed, but cross-platform bitwise identity
is not promised.

## Separate reporting and expected artifacts

Training writes raw baseline/evaluation episodes, `scorecard.json`,
`complete.json`, metrics, training seed logs, checkpoints, TensorBoard data and
provenance files. Native reward components and decoded actions are in the raw
episode JSON files. The full inventory is in the manifest.

Reporting is separate: `report_learning.py` reads `scorecard.json` and
`complete.json`, and writes **`scorecard.csv` and `final_report.txt`**. The original
tmux completion worker ran it with system Python after a successful training
exit; it has only standard-library dependencies. It does not run evaluations.

For a future NEW run directory, from a prepared Conda-deactivated environment:

```bash
cd environments/cage2
uv run --locked python report_learning.py --run-dir /absolute/path/to/NEW-run-directory
```

To regenerate only a historical report, first copy the two input JSON files to
a new scratch directory and target that directory. **Do not run the reporter
against the historical directory:** it overwrites its two output files.
Historical console logs are in `runs/cage2_blue_learning_seed0_20261123_job/`.

## Read-only verification and limitations

From the repository root:

```bash
python3 -B environments/cage2/reproducibility/verify.py
```

This standard-library checker only reads Git objects, files and wheel members.
It verifies the tag, all historical hashes, exact configuration, script hashes,
lockfile, CybORG gitlink, native wheel/library, checkpoint fingerprint and
schedule against the saved results. It does not load pickle, install packages,
run reporting, simulate an episode or train. The separate documentation-time
checkpoint inspection also verified required keys and optimizer/replay counts.
`verification.json` records the completed verification.

Remaining limitations: local artifacts still need an independently preserved
archive; the original tag cannot contain later documentation; exact scheduling
requires the saved two-phase schedule rather than the adaptive CLI; live
simulator state is not checkpointed, so restarting midway can change collection;
host libraries are not exhaustively locked; GPU/driver changes and future
package availability may prevent bitwise reproduction. This one training seed
and 20 development seeds do not measure variation across independent training
runs. No training or historical report regeneration was performed here.
