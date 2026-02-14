📘 Recurrent PPO — GPU Validation Suite
This directory contains the GPU‑side validation suite for the recurrent PPO implementation.
These tests enforce the mathematical and architectural invariants required for:

deterministic LSTM state‑flow

correct TBPTT behavior

reproducible rollouts

stable per‑unit diagnostics (drift, saturation, entropy)

correct hidden‑state alignment

correct masking semantics

shape invariants for all recurrent tensors

If any test in this suite fails, the recurrent pipeline is no longer guaranteed to be correct.
