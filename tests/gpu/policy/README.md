# Test Suite Overview
_What each test enforces and why it matters_

This directory contains tests that enforce the **real invariants** required for stable, correct recurrent PPO training.

These tests do **not** enforce bit‑equality between rollout and training paths, because my architecture intentionally uses two different execution paths.

Instead, they enforce:

- correct shapes
- correct time‑major layout
- correct gradient flow
- correct PRE‑STEP → POST‑STEP semantics
- correct TBPTT behavior
- correct mask alignment

---

# 1. Rollout‑Consistency Contract Test

**File:** `test_evaluate_actions_sequence_contract_rollout_like.py`

### What it checks

- `evaluate_actions_sequence` returns tensors in **time‑major** format `(T, B, ...)`
- `logits`, `values`, `logprobs`, `entropy` **require gradients**
- `new_hxs`, `new_cxs`, and gate diagnostics **do not require gradients**
- actions are interpreted correctly as `(T, B)` or `(T, B, 1)`
- no silent shape corruption

### Why it matters

PPO uses:

- logprobs → policy gradient
- values → value loss
- entropy → entropy bonus

Hidden states **must not** carry gradients across rollout boundaries.

This test ensures the training path is **PPO‑compatible**.

---

# 2. TBPTT Equivalence Test

**File:** `test_evaluate_actions_sequence_tbptt_equivalence.py`

### What it checks

- Full‑sequence evaluation
  vs
- Chunked TBPTT evaluation

produce **identical logits and values**.

### Why it matters

TBPTT correctness ensures:

- stable recurrent training
- correct PRE‑STEP → POST‑STEP state propagation
- no drift across chunk boundaries
- consistent gradients regardless of chunk size

If this test fails, TBPTT is broken.

---

# 3. Mask‑Aware Test

**File:** `test_evaluate_actions_sequence_mask_alignment.py`

### What it checks

- masks/dones align with time‑major layout `(T, B)`
- masked logprobs/values zero out correctly
- shapes remain consistent after masking

### Why it matters

PPO uses masks to:

- cut gradients across episode boundaries
- zero out invalid timesteps
- compute advantages correctly
- avoid mixing episodes in minibatches

If masks don’t align with `(T, B)`, training silently corrupts.

---

# Summary Table

| Test | Enforces | Why It Matters |
|------|----------|----------------|
| Rollout‑Consistency Contract | Shapes, gradients, time‑major layout | PPO correctness |
| TBPTT Equivalence | Full vs chunked evaluation match | Stable recurrent training |
| Mask‑Aware | Masks align with `(T,B)` and zero out correctly | Correct advantage/value computation |

---

# Final Notes

These tests encode the **real** invariants required by my architecture.
They do **not** enforce bit‑equality between rollout and training paths, because that is not a valid invariant for this model.

If any of these tests fail, PPO training becomes unstable or incorrect.
