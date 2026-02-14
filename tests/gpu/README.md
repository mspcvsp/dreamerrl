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

