markdown

# Dreamer‑V3 Latent Structure — Mental Model Guide

This document summarizes the **factored discrete latent model** used in Dreamer‑V3.
It exists to give future‑you a crisp, stable mental model of how V3 represents, samples, and uses latent variables.

---

## 1. Overview

Dreamer‑V3 replaces the old Gaussian latent with a **factored discrete latent**:

z_t = [ z_t¹, z_t², …, z_tᴷ ]
Code


Each factor is a **categorical variable** with **C** classes.

This gives the latent a **3‑D structure**:

(B, K, C)
Code


Where:

- **B** — batch size
- **K** — number of latent factors (`stoch_size`)
- **C** — number of categories per factor (`num_classes`)

Example:

K = 30
C = 32
z.shape = (B, 30, 32)
Code


---

## 2. Why Dreamer‑V3 uses factored discrete latents

### 2.1 Expressiveness

A factored categorical latent has capacity:

C^K
Code


Example:

32^30 ≈ 10^45 states
Code


This is vastly more expressive than any Gaussian latent.

### 2.2 Stability

Discrete latents + straight‑through Gumbel‑Softmax avoid:

- posterior collapse
- KL explosions
- mode collapse
- unstable gradients

### 2.3 Interpretability

Each factor can specialize in a different aspect of the environment.

### 2.4 Clean KL geometry

KL splits cleanly into:

- **KL_dyn** — “Did the dynamics predict the right latent?”
- **KL_rep** — “Did the encoder add extra information?”

This split only makes sense when z is factored.

---

## 3. Shapes at a glance

### Posterior / Prior outputs

logits: (B, K, C)
probs:  (B, K, C)
z:      (B, K, C)   # one-hot or straight-through
Code


### WorldModelState

h: (B, deter_size)
z: (B, K, C)
Code


### Decoder / RewardHead / ContinueHead / Actor / Critic

All consume:

h: (B, deter_size)
z: (B, K, C)
Code


And internally do:

z_e = Linear(C → H) applied per factor → (B, K, H)
z_sum = z_e.sum(dim=1) → (B, H)
h_e = Linear(deter_size → H)
features = h_e + z_sum
Code


This is the **V3 fusion rule**.

---

## 4. Why flattening z is forbidden

Flattening:

(B, K, C) → (B, K*C)
Code


destroys:

- factor independence
- categorical structure
- KL geometry
- actor/critic conditioning
- imagination stability
- determinism invariants

**V3 rule:**
**Never flatten z.**
Only merge time and batch dimensions.

---

## 5. Sampling in V3

Dreamer‑V3 uses **Gumbel‑Softmax straight‑through**:

g = -log(-log(U))
y = softmax((logits + g) / temperature)
z_hard = one_hot(argmax(y))
z = z_hard + (y - y.detach())
Code


This gives:

- discrete forward pass
- differentiable backward pass

---

## 6. KL structure

For each factor:

KL_dyn = KL[ sg(q) || p ]
KL_rep = KL[ q || sg(p) ]
KL_total = KL_dyn + KL_rep
Code


Shapes:

KL_dyn: (B, K)
KL_rep: (B, K)
KL_total: (B, K)
Code


Then averaged over batch and factors.

---

## 7. Deterministic state h

Dreamer‑V3 uses:

h_{t+1} = f(h_t, action_t)
Code


with:

- no GRU
- no z in the deterministic update
- pure MLP + LayerNorm
- fully deterministic across CPU/GPU

This keeps imagination stable.

---

## 8. Summary Diagram

┌──────────────┐
│  Observation  │
└───────┬──────┘
│
▼
┌──────────────┐
│   Encoder     │
└───────┬──────┘
│ embed
▼
┌──────────────────────┐
│ Posterior q(z | h,e) │
└──────┬──────────────┘
│ z_post (B,K,C)
▼
h_prev ──► RSSMCore ──► h_next
▲
│
┌──────┴──────────────┐
│ Prior p(z | h_prev) │
└──────────────────────┘

Decoder / Reward / Continue / Actor / Critic all consume:
(h_next, z_post) or (h_next, z_prior)
Code


---

## 9. TL;DR mental model

- **K** = number of latent *factors*
- **C** = number of *categories* per factor
- **z** is a **grid of categorical variables**
- **never flatten z**
- **sum embeddings across factors**
- **add h‑embedding + z‑embedding**
- **KL is per‑factor**
- **RSSMCore is deterministic and z‑free**

This is the entire Dreamer‑V3 latent philosophy in one page.
