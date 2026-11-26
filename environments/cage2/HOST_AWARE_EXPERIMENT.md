# Host-aware encoder v1: implementation and proposed controlled experiment

Formal training is **not authorized or started**. The implementation and uv
migration were committed by the user at `5364c6f`; follow-up validation notes and
manifest metadata await review before sealing an experiment tag.
`host_aware_v1_manifest.json` is intentionally unsealed and fails the launch gate.
Do not alter the immutable `cage2_baseline_v1` tag.

## Architecture and visible information

The original 52-value vector represents 13 Blue-visible hosts, four bits each:
Activity and the defender's inferred Compromised state. `host_schema.py` reads
only BlueTable's visible Blue-observation layer, using exactly its serialization
order. It never reads the simulator truth table, Red state, or true compromise.
It retains host names and observed subnet group membership, not randomized IPs.
Schema is validated on each episode reset and serialized with model config.

`WorldModelConfig.obs_encoder` defaults to `flat`; `cage2_host` opts in. The
original flat MLP implementation and parameter layout are unchanged. Both paths
output 200 features, so RSSM, posterior, prior, actor, critic, decoder, reward
model, losses, replay sampler and training algorithm retain their interfaces.
The existing `encoder_hidden=256` field did not control the original embedding;
actual RSSM/encoder width is 200 and stays 200.

| Operation | Shape |
|---|---|
| Visible vector → host fields | `[...,52] → [...,13,4]` |
| Host identity embeddings | `13 × 8` |
| Visible subnet embeddings | `3 × 4` |
| Shared host MLP, SiLU | `16 → 64 → 64` per host |
| Masked mean/max | two global 64-feature vectors |
| Global context | three subnet means + coverage fractions: `195 → 64`, SiLU |
| Final projection | pooled mean/max + context: `192 → 200`, SiLU |

Complete host records (fields, identity, subnet and mask) may be reordered without
changing output. Moving evidence between identities is deliberately different.
Padding/missing data is masked before nonlinearities, including NaN sentinels.
All-masked observations produce a finite learned constant. A healthy all-zero
record and Compromised=Unknown are valid evidence, not missingness. Actual
Scenario2 uses all 13 hosts; masking supports safe batching, not hidden-state
inference. Fixed identity embeddings do not establish unseen-topology transfer.

| Parameters | Flat | Host-aware |
|---|---:|---:|
| Encoder | 50,800 | 56,508 |
| Whole world model | 1,896,814 | 1,902,522 |
| Actor | 400,529 | 400,529 |
| Critic | 373,801 | 373,801 |
| Total trainable | 2,671,144 | 2,676,852 |

This is +5,708 parameters: 11.24% in the encoder, 0.214% of the whole trainable
agent. It is not an exactly parameter-matched comparison. The slow critic is an
unchanged non-trainable copy and is excluded from the trainable total.

## Initialization and controlled configuration

`host_aware_v1_config.json` is authoritative. Six sequential runs compare flat
and host-aware encoders with training seeds 0, 1, 2. Both arms use fresh weights,
optimizer state and replay. They never load historical baseline weights.

A fresh flat reference model per seed initializes **all non-encoder modules
identically** in both arms; the host encoder keeps its own fresh weights. No
historical checkpoint is consulted. Flat initialization uses that fresh flat
reference encoder. Training/collection Torch RNG is reset to seed+123456 after
architecture-dependent construction, then real initial observations are filtered
again. Matching common-module hashes are recorded. All losses and optimizers
remain the existing production path.

The historical -35.39 at 507,450 interactions is a reference only. Its checkpoint
resume also changed the simulator seed sequence, whereas the controlled arms are
fresh uninterrupted runs. It is not an interchangeable controlled comparator.
Original checkpoints, run results and OneDrive archive remain untouched.

Each run collects 16 initial 30-step episodes (480 interactions), then completes
16,899 optimizer updates, adding 30 interactions each, for exactly 507,450.
The production batch is 16 × 16 with burn-in 8 and imagination horizon 15;
FP32, no fused optimizer, LR warmup 0, no actor/critic freeze, and the inherited
100,000-interaction random-action warmup stay fixed. The original LR phases are
preserved: through update 26 use scheduler target 100,000; then target 16,899.
No wall-clock measurements determine the training budget.

Development checkpoints are at 510, 30,000, 60,000, 100,020, 150,000, 250,020,
380,010 and 507,450 interactions. Development seeds are the original 20 seeds
1000000000–1000000019. Final checkpoints are evaluated on 200 new held-out seeds
2000000000–2000000199 only after all six runs complete. Greedy actions and
posterior modes are used everywhere. Evaluation preserves model/replay/RNG state.
No held-out result chooses architecture or checkpoint; no best-intermediate
selection enters the main comparison.

Reports include each training seed, final held-out returns, paired episode
bootstrap intervals conditional on those fitted policies, the three training-seed
paired differences and their t interval (df=2). With only three training seeds,
training uncertainty remains preliminary. Development learning curves/AUC,
wall-clock times and parameter counts are saved separately. A source/config/
dependency gate is rechecked before evaluated checkpoints and every 100 updates.
Native restart checkpoints are saved; the framework does not serialize live
simulator state, so a mid-run restart can change the collector trajectory.

Estimated cost remains roughly 14–18 GB10-hours for all six sequential runs plus
held-out evaluation. One-batch integration timings are startup/cache-sensitive
and are not a reliable steady-state speed comparison or budget selector.

## Lightweight reproduction checks (no formal run)

From the root uv environment:

```bash
uv sync --locked
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --locked pytest -c /dev/null -q \
  environments/cage2/test_adapter.py environments/cage2/test_reproduction_support.py \
  environments/cage2/test_uv_management.py environments/cage2/test_host_encoder.py \
  environments/cage2/test_host_experiment_support.py

# Two bounded integration batches, one per encoder; use a new output file.
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --locked python \
  environments/cage2/check_host_integration.py --output /tmp/NEW-host-batch-check.json

UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy pytest -q
uv run --locked ruff check .
uv run --locked pyright
python3 -B environments/cage2/reproducibility/verify.py
```

The integration script uses 480 real transitions per arm and one production-sized
world-model + imagination actor/critic update. It does not conduct a learning
experiment or retain trained checkpoints. Unit tests also check serialization,
flat/host mismatch rejection, fresh paired initialization and original update-1
loading. Data-dependent historical tests skip only when local historical blobs
are unavailable, as in CI; they run on this Spark.

## Required pre-training gate, after user review

1. Review all outstanding uv/cleanup and encoder changes. Commit relevant source,
   configuration, tests and policies. Ensure a clean working tree and clean pinned
   CybORG submodule. Implementation commit `5364c6f` exists; finish reviewing the follow-up records
   before creating the experiment source tag.
2. Create a **new annotated** tag such as `cage2-host-aware-encoder-v1` at that
   implementation commit. Never reuse/force-update an existing tag. This tag does
   not exist yet; subsequent code changes require another commit and new tag.
3. From that clean tagged source, run `prepare_host_manifest.py --tag TAG --output
   /path/to/NEW-execution-manifest.json`. It records resolved commit and tag object,
   every tracked-file fingerprint, configuration and dependency hashes, runtime
   package versions, CybORG revision, native wheel, seeds, schedule and fresh
   checkpoint provenance. Its launch approval field remains empty.
4. Review and commit this execution manifest in the manifest/review checkout.
   Because a Git commit cannot contain its own hash, the sealed execution manifest
   is recorded in a separate subsequent manifest-only commit. Run the experiment
   from a **separate clean detached checkout of the source tag**, not that later
   manifest commit. The runner verifies the external manifest matches its own
   committed Git blob and the source checkout matches the recorded tag exactly.
5. Install that source checkout with `scripts/setup_uv.sh`; run `--check-only`
   with the reviewed external manifest. Source, configuration, exact installed
   package set, uv.lock, native wheel, imported source paths and clean pinned
   submodule must match. The unsealed template is deliberately rejected.
6. Request explicit formal-launch approval. Record that approval in the reviewed
   execution manifest and commit the manifest-only update. Formal launch is blocked
   without it. Do not begin training merely because tests pass.

After approval, the tracked `scripts/run_host_aware.sh` can be launched in a named
detached tmux session, passing `--manifest /path/to/COMMITTED-manifest.json` and
`--output-root /path/to/TAGGED-checkout/runs/NEW-host-ablation`. It sequentially
executes the six arms and evaluates held-out seeds only at the end. It never
executes or imports code from runs/. Exact commands/tag/manifests will be finalized
at the gate, not guessed while changes are uncommitted.

## Ranking scope

This ablation covers only simulated Blue defense against B-line over 30 steps.
The official final leaderboard sums means across three opponents and three
horizons, with 1000 validation episodes per combination. A 30-step B-line score
alone cannot support a top-20/top-10 ranking claim. Matching those published
conditions is a separate future evaluation; no extra opponents, hierarchy,
reward shaping or auxiliary objectives are introduced here.

Source: https://github.com/cage-challenge/cage-challenge-2 (official final results
and evaluation protocol, checked 2026-11-24).

## Implementation verification on 2026-11-24

27 maintained CAGE2/host tests passed; the retained legacy suite passed 289 tests
with 1 manual skip. Ruff passed and Pyright checked 227 files with zero errors
and warnings. Both encoders completed one real, production-sized GB10 learner
batch with finite losses, optimizer counters 1/1/1 and nonzero encoder gradients.
Fresh non-encoder initialization hashes matched. Original update-1 loading and
all 63 historical artifact fingerprints passed. The unsealed launch gate refused
execution before creating an output directory. Detailed generated receipts are
in `runs/cage2_host_encoder_checks_20261124/`; no executable source is stored there.
The user committed implementation `5364c6f` during verification; this agent made
no commits. No formal run or new experiment tag was created. Policy improvement
is unmeasured pending the reviewed/tagged, explicitly approved experiment.
