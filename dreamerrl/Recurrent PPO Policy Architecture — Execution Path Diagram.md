# Recurrent PPO Policy Architecture — Execution Path Diagram

This diagram shows the **two execution paths** used by the LSTM‑based PPO policy:

- **Rollout path** (fast, single‑step, no auxiliary losses)
- **Training path** (full‑sequence, auxiliary losses, diagnostics, TBPTT‑compatible)

This split is **intentional** and **standard** in modern recurrent RL systems.

---

## High‑Level Diagram

```
                   ┌──────────────────────────────┐
                   │        Environment            │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                        (obs_t, h_t, c_t)
                                  │
                                  ▼
                    ┌────────────────────────┐
                    │      forward_step       │
                    │  (rollout-time path)    │
                    └────────────────────────┘
                                  │
                                  ▼
                     logits_t, value_t, action_t
                                  │
                                  ▼
                        (h_{t+1}, c_{t+1})
                                  │
                                  ▼
                   Stored into rollout buffer (T,B,…)
```

Rollout path is **single-step**, **fast**, and **minimal**:
- No LayerNorm across time
- No auxiliary heads
- No AR/TAR losses
- No sequence unroll
- No diagnostics
- Used only for acting in the environment

---

## Training-Time Path

```
Rollout Buffer (T,B,obs) + PRE-STEP states (h0,c0)
                    │
                    ▼
        ┌────────────────────────────────────┐
        │     evaluate_actions_sequence       │
        │ (training-time full-sequence path)  │
        └────────────────────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────┐
        │               forward               │
        │   (wraps _forward_core + heads)     │
        └────────────────────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────┐
        │            _forward_core            │
        │   (full LSTM unroll, batch-first)   │
        │   - LayerNorm                       │
        │   - Auxiliary heads (obs/reward)    │
        │   - AR/TAR regularization           │
        │   - Gate diagnostics                │
        └────────────────────────────────────┘
                    │
                    ▼
      logits(T,B,A), values(T,B), logprobs(T,B), entropy(T,B)
      new_hxs(T,B,H), new_cxs(T,B,H), gates(T,B,H), aux preds
```

Training path is **full-sequence**, **rich**, and **supervised**:
- Applies LayerNorm
- Computes auxiliary predictions
- Computes AR/TAR losses
- Records gate activations
- Uses batch‑first LSTM unroll
- Returns time‑major tensors for PPO
- Supports TBPTT chunking

---

## Why Two Paths?

```
Rollout: must be FAST → single-step, minimal compute
Training: must be RICH → full sequence, aux losses, diagnostics
```

Trying to force them to be identical would:
- break auxiliary losses
- break diagnostics
- break TBPTT
- slow down rollout dramatically
- remove LayerNorm benefits
- reduce stability and sample efficiency

This two‑path design is used in:
- IMPALA
- R2D2
- Dreamer / DreamerV2
- RLlib recurrent PPO
- DeepMind Acme recurrent agents
- CleanRL recurrent PPO

---

## Key Invariants (What Must Match)

Even though the *implementations* differ, PPO requires these **contracts** to hold:

```
1. PRE-STEP state semantics:
   evaluate_actions_sequence(h0,c0) must use the same PRE-STEP states
   that rollout used to produce actions.

2. Logprob/value correctness:
   logprobs(actions) and values must be computed from the training path.

3. Time-major layout:
   All outputs must be (T,B,...) for PPO minibatching.

4. Gradient correctness:
   - logits/values/logprobs/entropy require gradients
   - new_hxs/new_cxs must NOT require gradients

5. TBPTT equivalence:
   Full-sequence evaluation == chunked evaluation.

6. Mask alignment:
   Masks/dones must align with (T,B) layout.
```

These are exactly what the test suite enforces.

---

## What Must *Not* Match

```
forward_step ≠ _forward_core
forward_sequence ≠ evaluate_actions_sequence
```

These paths differ by design:
- LayerNorm
- auxiliary heads
- AR/TAR losses
- sequence-level encoding
- diagnostics
- batch-first vs time-major handling

Bit‑equality is **not** a valid invariant.

---

## Summary

This architecture uses two execution paths because:

- Rollout must be **fast and minimal**
- Training must be **rich and fully supervised**
- PPO only requires **contract-level consistency**, not bit-equality
- This design is **standard** in modern recurrent RL

The test suite enforces the **real invariants** that matter for correctness and stability.
