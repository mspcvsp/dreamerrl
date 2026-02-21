# ASCII LSTM Unroll Diagram (Time‑Major ↔ Batch‑First)

This diagram shows how the LSTM unrolls over time, how PRE‑STEP and POST‑STEP states flow, and how the policy switches between:

- **Time‑major** `(T, B, …)` — rollout buffer, PPO, masks
- **Batch‑first** `(B, T, …)` — internal LSTM core

It’s here so future‑you never has to reconstruct this from memory.

---

## 1. Time‑Major View (Rollout / PPO Storage)

```
Time axis →
t = 0      t = 1      t = 2      ...      t = T-1
│          │          │                     │
▼          ▼          ▼                     ▼

obs[0,b]   obs[1,b]   obs[2,b]   ...   obs[T-1,b]
  │          │          │                     │
  ▼          ▼          ▼                     ▼

(h0,b,c0,b) → (h1,b,c1,b) → (h2,b,c2,b) → ... → (hT,b,cT,b)
      │            │            │                     │
      ▼            ▼            ▼                     ▼
 logits[0,b]  logits[1,b]  logits[2,b]        logits[T-1,b]
 values[0,b]  values[1,b]  values[2,b]        values[T-1,b]
 actions[0,b] actions[1,b] actions[2,b]       actions[T-1,b]
```

Rollout buffer stores everything as **(T, B, …)**.

---

## 2. Batch‑First View (Internal LSTM Core)

`_forward_core` operates in **batch‑first** format:

```
Input:
    obs_bt: (B, T, F)
    h0:     (B, H)
    c0:     (B, H)

Unroll:
    for t in 0..T-1:
        enc_t = enc[:, t, :]        # (B, F)
        h_{t+1}, c_{t+1} = LSTM(enc_t, (h_t, c_t))

Outputs:
    out:      (B, T, H)
    pred_obs: (B, T, obs_dim)
    pred_rew: (B, T, 1)
    gates:    (B, T, H)  # detached
```

Then everything is transposed back to **time‑major**:

```
(B, T, A) → (T, B, A)
(B, T)    → (T, B)
(B, T, H) → (T, B, H)
```

---

## 3. Single‑Step vs Full‑Sequence

### Rollout (single‑step)
```
obs_t: (B, F)
enc_t = encoder(obs_t)
h_{t+1}, c_{t+1} = LSTM(enc_t, (h_t, c_t))

logits_t = actor(h_{t+1})
value_t  = critic(h_{t+1})
```

### Training (full‑sequence)
```
obs_bt: (B, T, F)
Unroll inside _forward_core
Apply heads, AR/TAR, diagnostics
Return (B, T, …) → transpose to (T, B, …)
```

---

## 4. PRE‑STEP vs POST‑STEP Semantics

```
PRE‑STEP:  h_t, c_t  → used to compute logits[t]
POST‑STEP: h_{t+1}, c_{t+1} → fed into next step
```

`evaluate_actions_sequence` must use the **same PRE‑STEP states** that rollout used.

---

## 5. Summary Diagram

```
Rollout Path (fast)                Training Path (rich)
--------------------               ---------------------
forward_step                        evaluate_actions_sequence
  │                                       │
  ▼                                       ▼
Single-step LSTM                     Full-sequence LSTM
No LN                                LayerNorm
No aux heads                         Aux heads (obs/reward)
No AR/TAR                            AR/TAR
Time-major outputs                   Batch-first → time-major
```

This diagram is the “don’t break this” reference for future contributors.
