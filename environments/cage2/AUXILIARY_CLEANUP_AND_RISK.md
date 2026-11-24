# Auxiliary cleanup and offline observable-risk study

## Maintained architecture

The five generic built-in objectives, head factory, latent-cluster skill helper,
extra reward-head container, auxiliary-only fixtures/manual learning experiment,
and old MiniHack auxiliary CLI option were removed. Existing Git tags retain the
historical implementation; the original baseline/pilot tags and artifacts are
unchanged. The pre-cleanup audit is retained as historical assessment, not active
API documentation. The obsolete audit-only executable was removed.

Standard reward and continuation heads remain. A single `RewardModel.main`
container preserves historical `reward_heads.main.*` state keys and the existing
`reward_head` alias without auxiliary heads. Initialization remains at the same
point; no discarded head initialization or per-observation auxiliary forwards
remain. With auxiliaries disabled, exact before/after checks cover RNG, weights,
state keys, scalar loss and every computed parameter gradient.

`AuxObjectiveConfig(name, fn, weight=1)` is the only extension contract:
`fn(h, z, batch)` receives batch-major `(B,L,H)` and `(B,L,K,C)` features and
returns one finite normalized scalar. Contributions are summed as
`sum(weight_i * loss_i)`, never averaged by the number of objectives. Names are
unique and weights finite/nonnegative; zero skips the callback. Callbacks own
label detachment, masking, episode boundaries and normalization. They are stateless
and must not close over unregistered trainable parameters; `nn.Module` callbacks
are rejected. Future learned environment heads need explicit registration and
checkpoint tests rather than an implicit generic factory. No objective is enabled
or invented by this interface.

The callback interface and WorldModel add no environment-specific imports or
labels. The existing observation factory retains encoder selection. Pre-existing
glyph reconstruction/metrics branches in the generic training-step module remain;
this cleanup does not claim to remove all historical environment coupling.

Original auxiliary-disabled checkpoints remain supported, including their old
pickled configuration attributes. Historical configurations/snapshots in tagged
checkouts remain authoritative for historical reruns. The obsolete global
`aux_reward_scale` is removed from maintained declarations/configuration; legacy
pickled extra attributes are unused. Old auxiliary-enabled checkpoints requiring
removed Python targets must use their historical source tag.

## Risk target validity

Use **observable defensive evidence burden**, a proxy rather than true risk:

```
R_t = (# visible host records with Activity=Exploit OR
       Compromised in {User, Privileged}) / 13
Y_t(H) = mean(R_{t+1}, ..., R_{t+H}), H in {1,5}
```

Primary horizon is 5; horizon 1 is descriptive. The supplied native BlueTable
wrapper serializes 13 hosts, four bits each. Activity pairs are None=00, Scan=10,
Exploit=11; Compromised pairs are No=00, Unknown=10, User=01, Privileged=11.
The target decoder validates native binary fields and rejects invalid Activity=01.
It uses only the saved actor-visible vectors. No simulator state, attacker
internals, native reward or candidate/future action enters the probe features.
Future visible vectors supply labels only, never features.

Unknown is reported separately and is **not evidence of safety**. No/None are
wrapper beliefs, not proof of no threat. Scan-only threats, hidden compromise,
service criticality and actual damage are not captured. Repeated/remembered labels
can make the target persist after evidence becomes stale; the target is not an
instantaneous ground-truth host count. All hosts are weighted equally. Restore,
Monitor and Analyse can change evidence independently of true risk. Minimizing
this signal could reward suppressing visibility rather than improving defense.
Consequently it is neither a proposed reward function nor a validated training
loss, and it is not a calibrated probability of compromise.

The initial aggregate label audit found nonzero evidence burden in 9512/10800
observations (mean 0.12090 of hosts). This confirms support, not predictive
performance. Both User and Exploit occur; Privileged is absent in this dataset.
The specification follows visible category meanings, not held-out score selection.

## Frozen offline protocol

Reuse the complete host-pilot latent dataset: 360 30-step greedy evaluation
episodes, original 216/72/72 fitting/validation/test split. Check original source
revision/tag, checkpoint SHA256, pinned clean CybORG source, uv.lock, frozen-weight
receipt and input hashes recorded by the prior clustering analysis. No Dreamer
model is loaded by this study, so model gradients/updates are impossible.

Only origins with all H future **pre-action** observations available are scored;
no terminal frame, episode crossing, padding or censored tail enters targets.
For H=5 this gives 5400/1800/1800 rows; for H=1, 6264/2088/2088. Episodes remain the
uncertainty unit. These test episodes have already been inspected in prior
analyses: the study is exploratory development evidence, not fresh confirmation.

Compare fitting-mean and current-risk persistence baselines with ridge probes on
scalar timestep, timestep one-hot lookup, raw observation, h, z, h+z,
observation+time, and a strong observable control including observation, time,
current risk and Unknown fraction. Additional h/z/h+z plus this control assess
incremental linear information. The nonlinear current-risk decoding in the control
prevents an unfair comparison with a raw-bit linear observation probe.

Normalize using fitting rows only; select ridge alpha on validation MSE over the
tracked grid. Alpha selection uses raw linear predictions; all scored risk
predictions are clipped to [0,1] with no validation/test recalibration. Report
R²/MAE/RMSE, fit/validation counts, test episode-bootstrap 95% intervals (2000
resamples), paired incremental R²/MAE intervals and regression calibration
(bias, slope, intercept and fixed-width predicted-risk bins). Calibration means
agreement of forecast burden with observed mean burden, not event probability.
Intervals describe episode variation for one agent, not independent training seeds.

A gain indicates decodable predictive information under this frozen policy; it
can reflect nonlinear recoding or clock/history, not a causal defense skill.
Forecasting behavior-policy observations is not counterfactual action-conditioned
risk. Even a strong result does not establish that adding its loss improves RL.
The automatic recommendation is to defer a new auxiliary until target validity,
action-conditioned predictive deficits and causal behavioral benefit are tested.

## Execution and outputs

```bash
uv run --locked python environments/cage2/observable_risk_job.py \
  --run-dir runs/cage2_observable_risk_host_pilot_20261124
```

Detached tmux: `cage2-observable-risk`; hard maximum 2700 seconds. The tracked
launcher clears Conda variables and uses strict locked uv. It records launch
command/time/PID/exit status. The worker saves resolved configuration, full source
diff and hashes; new source is staged for review, not committed without approval.
Do not launch duplicate jobs. Reruns require a new output directory.

Generated outputs include `comparison.csv/json`, `paired_information.json`,
`calibration.json`, validation paths, fitted coefficients/normalization,
`heldout_predictions.npz`, `risk_predictions_calibration.png`, `final_report.txt`,
`manifest.json`, `complete.json`, `console.log`, `launch.json`, `exit_code` and the
exact `reproduce_command.txt`. Existing checkpoint and dataset hashes are verified
again before successful completion.

Regression receipts: `runs/cage2_auxiliary_cleanup_20261124/`. The before-cleanup
snapshot is generated evidence, not executable code. Exact equivalence rerun:

```bash
uv run --locked python environments/cage2/auxiliary_cleanup_checks.py \
  --snapshot runs/cage2_auxiliary_cleanup_20261124/disabled_before.pt
```

Bound pytest discovery to the repository and use importlib mode: unbounded
null-config collection encountered an unrelated blocked OneDrive mount while
searching parent directories. The file trace and original logs were retained;
those invocations were terminated and not treated as passes.

## Completed verification gate

- Root CAGE2 profile: **42 passed, 1 skipped** in 4.50 seconds. Includes flat,
  glyph and host observation paths, CPU/GB10 real-replay integration, core learning
  contracts, weighted callback tests and risk target/alignment tests. The skipped
  test is the MiniHack-specific configuration journal; it passes under legacy.
- Locked legacy profile: **38 passed** in 5.58 seconds, including restart and
  next optimizer update, glyph models, recurrent burn-in, repeatability and all
  callback tests. These are bounded regression updates, not a training experiment.
- Ruff passed on changed Python files; repository Pyright: zero errors/warnings.
- Historical update-1 and final host-pilot checkpoints strictly reload on CPU with
  exact world-model weights and replay. Both SHA256 fingerprints match prior
  records; original CAGE2 lock and both original tag objects also match.
- Disabled-mode synthetic baseline before/after is bitwise equal for weights,
  registered keys, post-initialization RNG, loss and all parameter gradients.
- Existing Torch GB10 capability/deprecation warnings remain; the GPU integration
  test passes without any dependency or driver changes.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
uv run --locked python -m pytest --rootdir=. --confcutdir=. \
  --import-mode=importlib --noconftest -c /dev/null \
  -o pythonpath=/home/mspcvsp/Documents/dreamerrl/environments/cage2 \
  -p no:cacheprovider -q \
  dreamerrl/tests/functional/test_auxiliary_interface.py \
  dreamerrl/tests/functional/test_observation_model_factory.py \
  dreamerrl/tests/invariants/test_learning_contracts.py \
  environments/cage2/test_host_encoder.py environments/cage2/test_observable_risk.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
UV_PROJECT_ENVIRONMENT=.venv-legacy \
uv run --locked --no-default-groups --group legacy python -m pytest \
  --rootdir=. --confcutdir=. --import-mode=importlib -c /dev/null \
  -p no:cacheprovider -q \
  dreamerrl/tests/functional/test_checkpoint_resume.py \
  dreamerrl/tests/functional/test_glyph_model.py \
  dreamerrl/tests/functional/test_replay_burn_in.py \
  dreamerrl/tests/functional/test_training_repeatability.py \
  dreamerrl/tests/functional/test_auxiliary_interface.py

uv run --locked python environments/cage2/auxiliary_cleanup_checks.py \
  --checkpoint runs/cage2_blue_cycle_seed0_20261123/latest.pt \
  --checkpoint runs/cage2_host_aware_pilot_seed0_20261124/latest.pt \
  --output runs/cage2_auxiliary_cleanup_20261124_rerun/checkpoint_compatibility.json
```
