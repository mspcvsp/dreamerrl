# DreamerRL

TTCP CAGE Challenge 2 Blue is the primary research environment.
Use the [root uv development environment](docs/uv_environment.md):
`bash scripts/setup_uv.sh`, then `uv run --locked ...`. Start with the
[CAGE2 setup](environments/cage2/README.md) and
[baseline reproduction record](environments/cage2/reproducibility/README.md).
Historical MiniHack runs are OBE; their tracked environments and regression
tests remain. Artifact cleanup requires explicit approval; see the
[exact cleanup proposal](docs/maintenance/minihack_cleanup_proposal.md).

## TL;DR — RuntimeEnvInfo

`RuntimeEnvInfo` is the *only* authoritative source of environment metadata.
The trainer extracts metadata once from the real Gym environment and stores it here.
Everything else — policy, buffer, diagnostics, tests — must read from `env_info`.

**Why:**
- Prevents stale or duplicated dimensions
- Eliminates encoder/LSTM/action‑space mismatches
- Simplifies tests
- Future‑proofs multi‑env setups

**TrainerState only stores training‑time scalars.**
It exposes compatibility aliases (`flat_obs_dim`, `action_dim`) that forward to `env_info`.

**Tests may override `env_info` directly** without constructing a real environment.
