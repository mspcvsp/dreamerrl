# Dreamer Core Modules

This directory contains the *algorithmic heart* of the Dreamer implementation.
All math, rollout logic, and update rules live here.
Both `DreamerTrainer` (production) and `_TestDreamerTrainer` (unit tests) import these functions.

The goal is to keep trainers as thin orchestration shells while ensuring:

- a **single source of truth** for Dreamer math,
- **no duplicated logic** between trainer and test harness,
- **time‑major correctness** across all components,
- **clean invariants testing** for each subsystem.

---

## Modules

### `lambda_return.py`
Implements the time‑major λ‑return used for actor–critic training.

Inputs:
- `reward`: `(T, B)`
- `value`: `(T+1, B)`
- `discount`: scalar γ
- `lam`: scalar λ

Output:
- `(T, B)` λ‑returns

This function is used by:
- `actor_critic_update`
- unit tests (`test_lambda_return.py`)

---

### `imagination.py`
Implements latent imagination rollouts:

- repeatedly calls `world_model.imagine_step`
- collects deterministic state `h`, stochastic state `z`
- optionally samples actions from the actor
- optionally predicts values from the critic

Used by:
- `_TestDreamerTrainer` (for imagination rollout tests)

---

### `world_model_update.py`
Computes the world model loss for a single sequence:

- RSSM transition
- posterior/prior KL
- reconstruction loss
- reward prediction loss

Used by:
- `DreamerTrainer.update_world_model`
- `_TestDreamerTrainer.world_model_training_step`

---

### `actor_critic_update.py`
Runs the full Dreamer actor–critic update:

- imagination rollout
- reward/value stacking
- value bootstrapping
- λ‑return
- actor loss
- critic loss

Used only by:
- `DreamerTrainer.update_actor_critic`

---

## Design Principles

### 1. Time‑major everywhere
All rollouts and returns use `(T, B, ...)` ordering.
This matches Dreamer v1/v2 and avoids transposes.

### 2. Pure functions
Core functions contain **no side effects**:
- no optimizers
- no logging
- no replay buffer access
- no environment interaction

This makes them:
- testable,
- deterministic,
- reusable.

### 3. Trainers orchestrate, core computes
`DreamerTrainer` and `_TestDreamerTrainer` only:
- sample batches,
- call core functions,
- apply optimizers.

All math lives here.

---

## Testing Philosophy

Each core function has a corresponding invariants test:

- `test_lambda_return_invariants.py`
- `test_imagination_invariants.py`
- `test_world_model_update_invariants.py`
- `test_actor_critic_update_invariants.py`

These tests ensure:
- shape correctness,
- device correctness,
- time‑major correctness,
- finite outputs,
- deterministic behavior under fixed seeds.

This guarantees that the Dreamer pipeline is stable and correct before integrating with environments or training loops.
