# 📘 Recurrent PPO — GPU Validation Suite

This directory contains the GPU‑side validation suite for the recurrent PPO implementation.  
These tests enforce the mathematical and architectural invariants required for:

- deterministic LSTM state‑flow  
- correct TBPTT behavior  
- reproducible rollouts  
- stable per‑unit diagnostics (drift, saturation, entropy)  
- correct hidden‑state alignment  
- correct masking semantics  
- shape invariants for all recurrent tensors  

If any test in this suite fails, the recurrent pipeline is no longer guaranteed to be correct.

---

## 📁 File Overview

### **Recurrent Core Tests**
Validate the core recurrent PPO invariants:

- **TBPTT determinism** — chunked evaluation must match full‑sequence evaluation  
- **Rollout replay determinism** — identical inputs must produce identical outputs  
- **Hidden‑state alignment** — buffer must store the *pre‑step* LSTM state `(h_t, c_t)`  
- **Mask correctness** — terminated/truncated episodes must not leak hidden state  

---

### **Diagnostics Tests**
Validate per‑unit LSTM diagnostics:

- gate means (i, f, g, o)  
- per‑unit drift  
- gate saturation (sigmoid/tanh)  
- gate entropy  
- replay determinism for diagnostics  
- mask‑aware drift computation  
- no NaNs, no shape mismatches  

---

### **Initialization & Shape Tests**
Micro‑tests that catch subtle regressions:

- LSTM state shape invariants  
- deterministic state initialization  
- correct device placement  

---

## 🧠 Why These Tests Matter

Recurrent PPO is extremely sensitive to:

- hidden‑state alignment  
- deterministic transitions  
- correct TBPTT slicing  
- correct masking  
- stable per‑unit metrics  

A single shape mismatch or incorrect state carry‑over can silently corrupt training.  
This suite ensures that every rollout, every update, and every diagnostic is mathematically correct.

---

## 🧪 Running the Suite

```bash
pytest -q tests/gpu/
pytest tests/gpu/test_recurrent_core.py -q
pytest tests/gpu/test_recurrent_core.py::test_rollout_replay_determinism -q
```

---

# 🧩 GPU Infra Test API Guide

This guide explains why GPU tests use a **batch‑major API** while the core PPO system uses a **time‑major API**, and why the GPU fixtures (`fake_buffer_loader`, `fake_batch`, `fake_rollout`) act as compatibility shims.

Both APIs are valid — each serves a different layer of the system.

---

## 1. Two Valid API Shapes

### Time‑major API `(T, B, …)` — “Training‑time truth”

**Used by:**

- `RecurrentRolloutBuffer`
- `RecurrentBatch`
- PPO training
- TBPTT chunking
- LSTM diagnostics
- CPU tests

**Why this layout:**

- preserves temporal structure  
- aligns with TBPTT invariants  
- `hxs[t]`, `cxs[t]` are *pre‑step* hidden states  
- `next_obs[t] = obs[t+1]`  
- chunking slices cleanly along time  

This is the canonical layout for recurrent PPO.

---

### Batch‑major API `(B, T, …)` — “Test‑time convenience”

**Used by:**

- GPU tests  
- policy‑level convenience methods (`forward_sequence`, `forward_tbptt`)  
- GPU fixtures in `gpu/infra/conftest.py`  

**Why this layout exists:**

Before the refactor, the policy exposed a batch‑major API and the GPU tests were written against that interface.  
Instead of rewriting the entire GPU suite, the fixtures provide a shim that:

- converts **`(T, B, …)` → `(B, T, …)`**  
- synthesizes `h0`, `c0` from `hxs[0]`, `cxs[0]`  
- synthesizes `done` from `terminated | truncated`  

Both layouts are correct in their respective contexts.

---

## 2. Why the GPU Fixtures Act as Shims

GPU tests expect:

```python
replay.obs   # (B, T, …)
replay.h0    # (B, H)
replay.c0    # (B, H)
rollout.done # (B, T)

