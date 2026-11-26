# Frozen host-pilot RSSM analysis

The tracked `latent_analysis_job.py` launches the single detached tmux session
`cage2-latent-analysis`, using `scripts/run_latent_analysis.sh` and strict locked uv.
The worker enforces a 2700-second (45-minute) process-group limit. It records
launch command, time, PID, progress and exit status, and does not queue retries or
Dreamer training. Use a new output directory on every rerun.

```bash
uv run --locked python environments/cage2/latent_analysis_job.py \
  --run-dir /home/mspcvsp/Documents/dreamerrl/runs/cage2_latent_analysis_host_pilot_20261124
```

`latent_analysis_config.json` fixes seeds and the protocol before collection.
The checkpoint SHA256 must be
`8a2483a6add81e1db709f35079f086c50778b900c24164b99fa29b62240cc1bd`,
from tag `cage2-host-aware-pilot-v1` / commit `ac2ec1b936b29ccf37d11ab4e2656f6636211fb4`.
Source audit compares all Dreamer source and the lock against the pilot manifest,
and verifies the tag object, pinned clean CybORG source and final summary (-28.64).
The analysis records its own source revision/diff and file hashes. It never
executes historical scripts under runs/.

## Protocol

Collect 360 fresh simulator seeds 3000000000–3000000359, disjoint from training,
prior development and previously reserved held-out seeds. Use the frozen greedy
actor and posterior modes, Scenario2, supplied B-line and 30 steps. Policy RNG
seed is simulator seed +100000. All model parameters are frozen; collection is
under torch inference mode. Verify model/buffer hashes, replay size and optimizer
update counters, and rehash the checkpoint after analysis.

At each decision t=0..29, encode the real observation o_t using the actual previous
action and recurrent history, then record h_t, one-hot z_t and posterior/prior
probabilities. Predict G_t=sum(rewards[t:30]), including the reward for action a_t.
Features precede a_t: no candidate action, current/future reward, successor or
terminal frame enters any probe. Raw observations, native rewards, timesteps and
actions are retained as dataset evidence. No simulator truth or hidden compromise
state is read. The combined feature is concatenation of h and flattened z.

Split whole episodes using seed 20261124: 216 fit, 72 validation and 72 test,
respectively 6480/2160/2160 transitions. Never split rows randomly across episodes.
Ridge uses fitting-only feature means/scales and an unpenalized intercept. Select
alpha from the fixed grid using validation MSE, then score test once per family.
Objective is mean squared error + alpha*||beta||². Save coefficients and paths.

Primary table: raw observation, h, z, combined h/z, scalar-time and a strong
train-only per-timestep-mean linear lookup baseline. Additional time-one-hot
controls test whether latent decoding adds more than episode progression. The
predeclared strongest incremental comparison is h/z+observation+time versus
observation+time. A secondary h+observation+time control isolates the recurrent
state comparison. Report pooled transition R²/MAE/RMSE, sample/episode counts,
time-residualized R² and paired **episode** bootstrap intervals (2000 resamples).
These intervals describe evaluation episodes under one fitted agent, not training
seed uncertainty. Clock-control comparisons are exploratory secondary diagnostics.

## Diagnostics and outputs

`frozen_features.npz` contains observed features and targets; `episode_splits.json`
records IDs. `linear_probe_comparison.csv/json`, `validation_paths.json`,
`linear_probe_coefficients.npz` and `incremental_information.json` carry the
quantitative evidence. `predicted_vs_actual.png` visualizes test predictions.

Diagnostics report hard category frequencies/coverage, conditional posterior
entropy, soft marginal entropy and variation, posterior-prior KL, h variance,
covariance effective rank and participation ratio, and time-residualized h rank.
Low mode usage alone is not proof of posterior collapse. Policy restriction,
limited observed trajectories, categorical mixtures and nonlinear information
can all limit linear probing. Weak accuracy means difficult linear decoding, not
proof of absent information.

PCA is fitting-only on standardized h and combined h/z. Twelve representative test
trajectories are chosen by fixed episode IDs, not rewards. Explained variance and
IDs are saved in pca_metadata.json; plots are descriptive, not cluster-based proof.
No UMAP, sklearn or new dependency is introduced. Interpret findings as
correlation/policy-conditional prediction, not causal evidence for defense.

The automatic final_report.txt prioritizes the probe table and limitations.
All generated artifacts, logs, manifest, launch metadata, completion status and
exact rerun command live under the new runs/ directory. Historical checkpoint,
results and tag remain unchanged. Unit tests verify leakage boundaries,
standardization, ridge correctness, label/terminal alignment and full reporting.
