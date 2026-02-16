# Recurrent PPO State-Flow & TBPTT Invariants

This document captures the core invariants that make the recurrent PPO pipeline deterministic, debuggable, and mathematically correct. These invariants are validated by dedicated CPU/GPU tests and must hold across:

- rollout
- buffer storage
- trainer minibatching
- TBPTT chunking
- evaluate_actions_sequence
- CPU/GPU execution paths

---

## 1. PRE-STEP vs POST-STEP LSTM State-Flow

Each timestep t has two distinct LSTM states:

    PRE-STEP:  (h_t, c_t)        # state before consuming obs[t]
    POST-STEP: (h_t+1, c_t+1)    # state after consuming obs[t]

### Rollout-time behavior

During rollout:

- PRE-STEP states (h_t, c_t) are stored in the rollout buffer.
- POST-STEP states (h_t+1, c_t+1) become the next timestep’s PRE-STEP.

### Training-time behavior

evaluate_actions_sequence must reproduce the exact same transitions:

    full.pre_hxs[t] == h_t
    full.new_hxs[t] == h_t+1

### Why this invariant matters

If PRE/POST alignment breaks:

- TBPTT chunking begins from the wrong hidden state
- PPO logprobs and values become misaligned
- recurrent diagnostics (drift, saturation, entropy) become meaningless
- rollout vs replay diverges
- CPU/GPU determinism is lost

### Tests that enforce this invariant

- test_policy_pre_post_state_flow_alignment
- test_pre_post_state_flow_alignment_gpu
- test_tbptt_state_flow_equivalence
- test_rollout_vs_replay_equivalence_gpu

---

## 2. TBPTT Gradient-Flow Invariant

TBPTT must truncate gradients at chunk boundaries.

For chunk size K:

- Full unroll: gradients flow through all timesteps
- TBPTT unroll: gradients should be significantly smaller for timesteps earlier than T - K

### Correct invariant

We compare mean gradient magnitudes:

    early_grad_mean  <<  late_grad_mean
    ratio = early_grad_mean / late_grad_mean

### Why we use means (not sums)

CUDA LSTM kernels produce small but consistent residual gradients due to:

- fused gate kernels
- FP32 accumulation noise
- tensor-core rounding
- kernel reordering

Summing across many timesteps inflates early gradients.
Means give a stable, architecture-correct signal.

### Expected behavior

- Correct TBPTT: ratio between 0.2 and 0.6
- Broken TBPTT: ratio near 1.0
- Perfect truncation (rare): ratio near 0.0 to 0.1

### Tests that enforce this invariant

- test_tbptt_gradient_flow_correctness
- test_tbptt_gradient_flow_correctness_gpu
- test_tbptt_replay_equivalence_gpu

---

## 3. Why These Invariants Matter

These invariants ensure:

- deterministic state-flow
- correct PPO logprob/value alignment
- correct TBPTT chunk boundaries
- stable recurrent diagnostics
- reproducible CPU/GPU behavior
- correct replay vs rollout equivalence

Breaking any of these invariants leads to:

- nondeterministic training
- misaligned gradients
- incorrect diagnostics
- unstable PPO updates

These invariants form the backbone of a research-grade recurrent PPO implementation.
