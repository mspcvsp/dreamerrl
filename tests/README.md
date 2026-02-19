# Testing Architecture Overview

This document explains the structure, purpose, and invariants of the entire test suite.
It exists so that future contributors (and future Sean) can understand:

- what each test layer protects
- why the architecture uses two execution paths
- how rollout, training, and TBPTT interact
- where to add new tests safely

This README lives at the **top level of the `tests/` directory**.

---

# 1. Philosophy of the Test Suite

Recurrent PPO is fragile.
Small mistakes in:

- hidden‑state flow
- mask propagation
- TBPTT chunking
- advantage normalization
- GAE
- rollout vs training path semantics

…can silently corrupt training.

The test suite is designed to enforce **architectural invariants**, not just correctness of individual functions.

The goal is:

> **If a change breaks a core invariant, a test must fail immediately.**

---

# 2. Directory Structure

```
tests/
│
├── cpu/                     # CPU-only tests (fast, deterministic)
│   ├── policy/              # Policy-level invariants
│   ├── trainer/             # Trainer-level logic (CPU)
│   └── infra/               # Shape, layout, PRE-STEP semantics
│
├── gpu/                     # GPU tests (device correctness, TBPTT)
│   ├── policy/
│   ├── trainer/
│   └── infra/
│
└── README.md                # This file
```

CPU tests run quickly and catch most logic bugs.
GPU tests ensure device placement, TBPTT, and mask propagation behave identically.

---

# 3. Policy Test Layer

Policy tests validate the **training-time path**, not the rollout path.

The policy has two execution paths:

- **Rollout path** → `forward_step` (fast, single-step)
- **Training path** → `_forward_core` → `forward` → `evaluate_actions_sequence` (full-sequence)

These paths are intentionally different.

Policy tests enforce:

### ✔ Rollout‑Consistency Contract
- Shapes are correct `(T, B, …)`
- Gradients flow only through logits/values/logprobs/entropy
- Hidden states and diagnostics are detached
- Time-major layout is preserved

### ✔ TBPTT Equivalence
Full-sequence evaluation must match chunked evaluation:

```
evaluate_actions_sequence(full_seq)
==
concat(evaluate_actions_sequence(chunks))
```

This ensures TBPTT does not change the computation.

### ✔ Mask Alignment
The policy is mask‑agnostic, but its outputs must be compatible with trainer masks.

---

# 4. Trainer Test Layer

Trainer tests validate:

- GAE
- advantage normalization
- mask propagation
- hidden-state resets
- TBPTT boundary behavior
- rollout PRE‑STEP semantics
- minibatch slicing invariants

These tests ensure the trainer produces correct PPO inputs.

### ✔ GAE Correctness
Matches the mathematical recurrence exactly.

### ✔ Advantage Normalization
Only valid timesteps contribute to mean/std.

### ✔ Mask Monotonicity
Masks must be non-increasing along time.

### ✔ Hidden-State Reset
If `done[t] == True`, then the PRE‑STEP state at `t+1` must be zero.

### ✔ TBPTT Mask Propagation
Chunked evaluation must match full-sequence evaluation under masks.

---

# 5. Infra Test Layer

Infra tests enforce **shape, layout, and PRE‑STEP invariants**:

- `(T, B, …)` vs `(B, T, …)` correctness
- PRE‑STEP state stored in rollout buffer
- POST‑STEP state fed into next env step
- deterministic state flow
- correct transposes in `evaluate_actions_sequence`

These tests catch the most subtle bugs.

---

# 6. What Tests Do *Not* Enforce

The suite intentionally does **not** enforce:

- bit‑equality between rollout and training paths
- equality of logits/values between `forward_step` and `_forward_core`
- equality of hidden states between rollout and training

These are **not valid invariants** for this architecture.

---

# 7. Adding New Tests

When adding a new test, ask:

1. **What invariant does this protect?**
2. **Is this a policy invariant or a trainer invariant?**
3. **Does this belong in CPU, GPU, or both?**
4. **Does this test PRE‑STEP or POST‑STEP semantics?**
5. **Does this test shape/layout correctness?**

If the test protects a core invariant, it belongs here.

---

# 8. Summary Table

| Layer | Purpose | Examples |
|-------|---------|----------|
| **Policy** | Training-path correctness | TBPTT, contract tests, mask alignment |
| **Trainer** | PPO math correctness | GAE, advantage norm, mask monotonicity |
| **Infra** | Shape & state-flow invariants | PRE‑STEP, time-major, device placement |

---

# 9. Final Notes

This test suite is designed to make the entire RL pipeline:

- deterministic
- debuggable
- TBPTT‑safe
- mask‑correct
- rollout‑consistent
- PPO‑stable

If any of these invariants break, a test should fail immediately.
