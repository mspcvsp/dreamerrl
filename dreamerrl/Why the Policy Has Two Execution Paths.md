# Why This Policy Has Two Execution Paths
_Recurrent PPO architecture notes_

This policy intentionally uses **two different execution paths**:

- **Rollout path** → `forward_step`
- **Training path** → `_forward_core` → `forward` → `evaluate_actions_sequence`

This is not a bug or an inconsistency.
It is a **standard, widely‑used design** in recurrent RL systems, including:

- IMPALA
- R2D2
- Dreamer / DreamerV2
- DeepMind Acme recurrent agents
- RLlib recurrent PPO
- CleanRL recurrent PPO

The two paths exist because rollout and training have **different requirements**.

---

## 1. Rollout Path: Fast, Lightweight, Single‑Step

Rollout happens every environment step, often millions of times.
It must be:

- fast
- single‑step
- minimal overhead
- no auxiliary heads
- no sequence‑level operations
- no AR/TAR losses
- no LayerNorm across time

This is exactly what `forward_step` does.

It computes:

- logits
- values
- next hidden state

for **one timestep only**, with minimal computation.

---

## 2. Training Path: Full‑Sequence, Rich, Supervised

Training happens on **entire sequences**, not single steps.
It must:

- unroll the LSTM over the full sequence
- compute auxiliary predictions (next‑obs, next‑reward)
- compute AR/TAR regularization
- expose gate activations for diagnostics
- operate in batch‑first format for efficiency
- return time‑major tensors for PPO
- preserve gradients through logits/values

This is exactly what `_forward_core` + `forward` + `evaluate_actions_sequence` do.

---

## 3. Why the Two Paths Cannot Be Bit‑Identical

Even if we wanted them to match exactly, they **cannot**, because:

- `_forward_core` applies **LayerNorm**
- `_forward_core` computes **auxiliary heads**
- `_forward_core` computes **AR/TAR losses**
- `_forward_core` encodes the entire sequence at once
- `forward_step` encodes one timestep at a time
- `forward_step` does not apply LN or auxiliary heads

These differences **change the hidden state distribution**, so logits/values will differ.

This is expected and correct.

---

## 4. What PPO Actually Requires

PPO does **not** require:

- rollout logits == training logits
- rollout hidden states == training hidden states

PPO **does** require:

- rollout actions are evaluated using the *training* path
- evaluate_actions_sequence uses the correct PRE‑STEP states
- evaluate_actions_sequence returns correct logprobs/values
- gradients flow only where they should
- TBPTT chunking does not change the computation
- masks/dones align with time‑major layout

These are the invariants enforced by the test suite.

---

## Summary

Two execution paths are:

- **Normal**
- **Intentional**
- **Standard in recurrent PPO**
- **Necessary for auxiliary losses, diagnostics, and TBPTT**

Do **not** try to force them to be identical — that would break the architecture.
