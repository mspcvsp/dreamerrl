# 🧠 Why TBPTT Matters for LSTM‑PPO
Recurrent PPO only works when **state‑flow** and **gradient‑flow** are handled with precision.
TBPTT (Truncated Backpropagation Through Time) is the mechanism that makes that possible.
This cheat sheet captures the *why*, not just the *what*

---

## 1. What TBPTT Actually Is

**TBPTT = Truncated Backpropagation Through Time**

It means:

- unroll the LSTM over the full sequence
- backpropagate only through short windows (e.g., 16–64 steps)
- **detach hidden state** between windows

This gives you long‑horizon memory without long‑horizon gradients.

---

## 2. Why TBPTT Is Essential in PPO

### A. PPO cannot handle full‑sequence BPTT

Full BPTT through long episodes would:

- explode GPU memory
- create massive autograd graphs
- destabilize PPO’s clipped objective
- produce extreme gradient variance

TBPTT keeps PPO’s update stable and bounded.

---

### B. PPO needs long‑term memory, not long‑term gradients

Your LSTM must remember:

- partial observability
- delayed rewards
- hidden environment state
- agent‑internal context

But PPO does **not** need gradients from hundreds of steps ago.

TBPTT gives you:

- **long‑term hidden state flow**
- **short‑term gradient flow**

This is the correct separation of concerns.

---

### C. TBPTT preserves PPO’s optimizer assumptions

PPO assumes each update uses:

- fixed advantages
- fixed logprobs
- fixed batch statistics

Full BPTT would leak gradients across episode boundaries and violate these assumptions.

TBPTT prevents that leakage.

---

## 3. What TBPTT Guarantees (via Invariants)

These are the invariants your test suite enforces.

### Invariant 1 — Hidden state must be *pre‑step* (`h_t`, `c_t`)

Ensures:

- deterministic rollout replay
- correct TBPTT slicing
- correct next‑state alignment

If this breaks, everything breaks.

---

### Invariant 2 — Chunked forward must match full forward

TBPTT validation test checks:

- full‑sequence forward
vs
- chunked forward with hidden‑state carryover

If these diverge, your LSTM is no longer a function.

---

### Invariant 3 — Hidden state must detach between TBPTT windows

Prevents:

- exploding gradients
- vanishing gradients
- unbounded graph growth
- PPO instability

This is the heart of TBPTT.

---

### Invariant 4 — Masks must zero hidden state on termination

If masks are wrong:

- terminated episodes leak memory
- truncated episodes leak memory
- LSTM learns impossible correlations

Mask tests protect this.

---

### Invariant 5 — Diagnostics must match TBPTT semantics

Drift, saturation, and entropy metrics must:

- operate on the same slices
- respect masks
- average over hidden units
- be deterministic

Otherwise, LSTM PPO monitoring is broken.

---

## 4. Why TBPTT Makes LSTM‑PPO Trainable

TBPTT is the only method that satisfies all constraints:

| Requirement | Full BPTT | No BPTT | TBPTT |
|------------|-----------|---------|--------|
| Long‑term memory | ✅ | ❌ | ✅ |
| Stable gradients | ❌ | — | ✅ |
| Bounded compute | ❌ | — | ✅ |
| PPO‑compatible | ❌ | ❌ | ✅ |
| Deterministic replay | ❌ | ❌ | ✅ |

---

## 5. One‑Sentence Summary

**TBPTT lets the LSTM remember the whole episode while only backpropagating through short, stable windows — preserving PPO’s assumptions, keeping gradients sane, and keeping the computational graph bounded.**
