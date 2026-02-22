# Dreamer Invariants (CPU + GPU)

This document defines the core invariants for Dreamer‑Lite and full Dreamer.
These invariants replace the LSTM‑PPO TBPTT invariants and form the basis of the Dreamer test suite.

---

## 1. World Model Invariants

### 1.1 RSSM State Shape
- `init_state(B)` returns:
  - `state.h.shape == (B, deter_size)`
  - `state.z.shape == (B, stoch_size)`
- All tensors must be on the same device.

### 1.2 Observe Step Contract
`observe_step(prev_state, obs_t)` must return:
- `state.h: (B, deter_size)`
- `state.z: (B, stoch_size)`
- `recon: (B, obs_dim)`
- `reward_pred: (B, 1)`
- `kl: scalar or (B,)`

### 1.3 Observe Step Determinism
Given the same `(prev_state, obs_t)`:
- CPU and GPU outputs must match within tolerance.
- Repeated calls must be deterministic.

### 1.4 Imagine Step Contract
`imagine_step(state)`:
- Does not depend on real observations.
- Preserves batch size and device.
- CPU/GPU equivalent.

---

## 2. Replay Buffer Invariants

### 2.1 Episode Construction
- Episodes end on `is_last`.
- Episodes shorter than `min_episode_len` are dropped.
- Final observations are correctly stitched when autoreset occurs.

### 2.2 Sampling
`sample(B, L)` returns:
- `state: (B, L, obs_dim)`
- `action: (B, L)`
- `reward: (B, L)`
- `is_first, is_last, is_terminal: (B, L)`
- All sequences are contiguous within an episode.

---

## 3. Actor / Critic Invariants

### 3.1 Actor
- `actor(h, z)` → logits `(B, action_dim)`
- `act(state)` returns:
  - `actions: (B,)`
  - `logprobs: (B,)`
- Actions must be in `[0, action_dim - 1]`.
- CPU/GPU equivalent.

### 3.2 Critic
- `critic(h, z)` → `(B, 1)`
- CPU/GPU equivalent.

---

## 4. Imagination Rollout Invariants

### 4.1 Horizon
- `imagine_step` produces a valid latent sequence of length `H`.
- No NaNs or infs.
- Shapes preserved.

### 4.2 λ‑Return
- `lambda_return(rewards, values)`:
  - preserves shape `(T, B)`
  - is finite
  - CPU/GPU equivalent

---

## 5. World Model Training Step

### 5.1 Loss Components
A single training step must produce:
- finite reconstruction loss
- finite reward loss
- finite KL
- gradients for encoder, RSSM, decoder, reward head

### 5.2 CPU/GPU Equivalence
- Loss values match within tolerance.
- Gradients match within tolerance.

---

## 6. Trainer Invariants

### 6.1 Exploration Schedule
- For `global_step < random_exploration_steps`, actions are uniform random.
- After that, actions come from `actor.act`.

### 6.2 World Model State Update
- `world_state = observe_step(world_state, obs)` must be consistent across devices.

---

# Summary

These invariants define the minimal, complete Dreamer test suite:
- World model correctness
- Replay buffer correctness
- Actor/critic correctness
- Imagination rollout correctness
- CPU/GPU numerical equivalence where it matters

They intentionally exclude all PPO‑specific invariants:
- TBPTT
- LSTM drift
- gate saturation
- clip range
- GAE
- mask propagation
- rollout buffer stitching
