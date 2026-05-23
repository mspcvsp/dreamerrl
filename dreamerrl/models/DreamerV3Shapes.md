# Dreamer‑V3 Shape Sanity Checklist

This document defines the canonical tensor shapes used throughout the Dreamer‑V3 world model, actor, critic, and
training loop. Use it as a reference when debugging shape mismatches or writing new components.
---

## 📦 Core Latent Variables

### Deterministic State
```
h_t : (B, deter_size)
```

### Stochastic Categorical Latent
```
z_t : (B, K, C)
```

Where:

- **B** — batch size
- **K** — number of categorical factors (`stoch_size`)
- **C** — number of classes per factor (`num_classes`)

Each factor `z_t[k]` is a one‑hot categorical variable.

---

## 🧠 RSSM Rollout Shapes

For a sequence of length **L**:

```
h_seq : (B, L, deter_size)
z_seq : (B, L, K, C)
```

Flattened for heads:

```
h_flat     : (B*L, deter_size)
z_factored : (B*L, K, C)
```

---

## 🖼 Decoder (Reconstruction Head)

### Input
```
h_flat     : (B*L, deter_size)
z_factored : (B*L, K, C)
```

### Output
```
recon_flat : (B*L, obs_dim)
recon      : (B, L, obs_dim)
```

---

## 🎯 Reward Head (Distributional)

### Output logits
```
reward_logits_flat : (B*L, value_bins)
reward_logits      : (B, L, value_bins)
```

### Readout (expectation over bins)
```
reward_pred : (B*L,)
```

---

## 🔁 Continue Head (Distributional)

### Output logits
```
cont_logits_flat : (B*L, value_bins)
cont_logits      : (B, L, value_bins)
```

### Readout (optional)
```
cont_pred : (B*L,)
```

---

## 📈 Value Head (Distributional Critic)

### Output logits
```
value_logits : (B, value_bins)
```

### Readout
```
value_pred : (B,)
```

---

## 🎮 Actor (Policy Network)

### Input
```
h : (B, deter_size)
z : (B, K, C)
```

### Output
```
action_logits : (B, action_dim)
actions       : (B,)   # sampled or argmax
```

---

## 📚 Loss Inputs

### Reconstruction
```
recon        : (B, L, obs_dim)
recon_target : (B, L, obs_dim)
```

### Reward
```
reward_logits : (B, L, value_bins)
reward        : (B, L)
```

### Continue
```
cont_logits : (B, L, value_bins)
cont_target : (B, L)
```

### KL Terms (from RSSM)
```
kl_dyn : (B, L)
kl_rep : (B, L)
```

---

## 🧮 World Model Training Step Summary

| Component      | Input Shape                          | Output Shape            |
|----------------|---------------------------------------|--------------------------|
| Decoder        | `(B*L, deter), (B*L, K, C)`           | `(B, L, obs_dim)`        |
| RewardHead     | `(B*L, deter), (B*L, K, C)`           | `(B, L, value_bins)`     |
| ContinueHead   | `(B*L, deter), (B*L, K, C)`           | `(B, L, value_bins)`     |
| KL Terms       | —                                     | `(B, L)`                 |
| Total Loss     | —                                     | scalar                   |

---

## 🧩 End‑to‑End Shape Flow

```
(B, L, obs_dim)
    ↓ encode
(B, L, deter), (B, L, K, C)
    ↓ flatten
(B*L, deter), (B*L, K, C)
    ↓ heads
(B, L, obs_dim)
(B, L, value_bins)
(B, L, value_bins)
    ↓ losses
scalar
```

---
