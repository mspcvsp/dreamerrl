# Contributor README — How to Modify the Policy Safely

This document is for anyone (including future you) who wants to modify the LSTM PPO policy without breaking:

- PPO correctness
- TBPTT
- diagnostics
- rollout/training consistency

The policy has **two execution paths**:

- `forward_step` — rollout-time, single-step, fast
- `_forward_core` → `forward` → `evaluate_actions_sequence` — training-time, full-sequence

Understanding this split is critical before making changes.

---

## Golden Rules

1. **Do not assume rollout and training paths are identical.**
   They are intentionally different and must remain so.

2. **Never change `_forward_core` or `forward_step` without re-running all LSTM tests.**
   These functions define the core recurrence and state flow.

3. **Preserve shapes and layouts:**
   - Rollout buffer: `(T, B, …)`
   - Internal LSTM: `(B, T, H)`
   - `evaluate_actions_sequence` outputs: `(T, B, …)`

4. **Gradients:**
   - `logits`, `values`, `logprobs`, `entropy` **must require gradients**
   - `new_hxs`, `new_cxs`, gate diagnostics **must NOT require gradients**

5. **Masks/dones must align with `(T, B)` time-major layout.**

---

## Where You Can Safely Add Things

### 1. Auxiliary Heads (e.g., new prediction tasks)

Add them in `_forward_core`:

- Use `out` (B, T, H) as input
- Return predictions in batch-first, then transpose to time-major in `evaluate_actions_sequence`
- Do **not** use them in `forward_step` (rollout must stay lightweight)

Remember to:

- detach where appropriate for diagnostics
- keep gradients where used in losses

---

### 2. New Diagnostics (e.g., gate stats, drift metrics)

Attach them to:

- `LSTMCoreOutput.gates` in `_forward_core`
- or compute them in the trainer using time-major tensors

Rules:

- Diagnostics should be **detached** (no gradients)
- Shapes must be `(T, B, H)` or `(T, B)` to align with rollout data

---

### 3. Changing the LSTM Cell / Hidden Size

If you change:

- `GateLSTMCell`
- hidden size
- encoder output size

You must ensure:

- `encoder` output dim == `lstm_cell.input_size`
- `_forward_core` asserts still pass
- all tests in the LSTM suite pass:
  - contract test
  - TBPTT equivalence
  - mask alignment

---

## Where You Must Be Extremely Careful

### 1. `forward_step`

This is the **rollout-time** path.

- It must remain **fast** and **single-step**
- It must not depend on sequence-level constructs
- It must not compute auxiliary losses
- It must not change shapes used by the rollout buffer

If you change how `forward_step` computes logits/values, you are changing **what the agent does in the environment**.

Always re-run:

- rollout-related tests
- any behavior/regression tests you have

---

### 2. `_forward_core` and `forward`

This is the **training-time** path.

- It defines the full LSTM unroll
- It computes auxiliary predictions
- It computes AR/TAR
- It feeds PPO and diagnostics

If you change:

- the unroll loop
- how `out` is constructed
- how heads are applied
- how AR/TAR are computed

You must re-run:

- TBPTT equivalence test
- contract test for `evaluate_actions_sequence`
- any diagnostics tests

---

### 3. `evaluate_actions_sequence`

This function is the **bridge** between:

- rollout buffer `(T, B, …)`
- internal batch-first core `(B, T, …)`
- PPO losses and diagnostics

It is responsible for:

- transposing obs to batch-first
- calling `forward`
- transposing outputs back to time-major
- computing logprobs and entropy
- detaching hidden states
- transposing gates

If you change:

- shapes
- transposes
- how actions are interpreted
- how logprobs/entropy are computed

You must re-run **all** LSTM tests.

---

## What You Should NOT Try to Do

- Do **not** try to make `forward_step` and `_forward_core` bit-identical.
  That would:
  - remove LayerNorm benefits
  - remove auxiliary losses
  - break diagnostics
  - slow down rollout

- Do **not** mix rollout-only concerns (e.g., env-specific hacks) into `_forward_core`.

- Do **not** detach logits/values/logprobs/entropy inside `evaluate_actions_sequence`.

---

## Tests as Guardrails

The following tests exist to protect these invariants:

- **Contract test**
  Ensures shapes, gradients, and time-major layout are PPO-compatible.

- **TBPTT equivalence test**
  Ensures full-sequence vs chunked evaluation produce the same outputs.

- **Mask-aware test**
  Ensures masks/dones align with `(T, B)` and zero out correctly.

If any of these fail after a change, you have broken a core invariant.

---

## TL;DR for Contributors

- Two paths (rollout vs training) are **intentional** and **standard**.
- Rollout must stay **fast and simple**.
- Training must stay **rich and sequence-aware**.
- Always think in terms of:
  - shapes
  - time-major vs batch-first
  - gradient flow
  - PRE-STEP vs POST-STEP semantics
- Never merge or “simplify away” the two-path design without redesigning the entire training stack.
