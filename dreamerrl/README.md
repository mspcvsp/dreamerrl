# Dreamer‑V3 Reinforcement Learning Pipeline

This repository implements a clean, test‑aligned, research‑grade version of **Dreamer‑V3**, a model‑based reinforcement learning algorithm that learns a world model, imagines trajectories in latent space, and trains an actor–critic agent entirely from imagination.

This document explains the **full pipeline**, from environment interaction to world‑model learning to imagination‑based actor–critic updates.

---

# 🌐 High‑Level Overview

Dreamer‑V3 consists of three interacting systems:

1. **Environment Loop**
   Collects real transitions, updates the latent RSSM state, and stores raw transitions in replay.

2. **World Model Learning**
   Learns to predict next latent state, reward, continuation, and reconstruction using real data.

3. **Imagination Actor–Critic**
   Rolls out trajectories in latent space, computes λ‑returns, trains a distributional critic, and trains an actor using normalized advantages.

These systems form a closed loop:

Environment → Replay → World Model → Imagination → Actor/Critic → Environment

---

# 🧠 World Model (RSSM)

The world model consists of:

- **Encoder**: embeds observations
- **RSSMCore**: deterministic transition `h_{t+1} = f(h_t, z_t)`
- **Posterior**: q(z_t | h_t, embed_t)
- **Prior**: p(z_t | h_t)
- **Decoder**: reconstructs observations
- **RewardHead**: predicts reward
- **ContinueHead**: predicts non‑terminal probability

The world model maintains a latent state:

WorldModelState:
    h : deterministic hidden state
    z : stochastic latent

### Observe Step (real data)

prev_state + obs → encoder → posterior → RSSM → new latent state

### Imagine Step (latent rollout)

prev_state → prior → RSSM → next latent state

---

# 🔁 Environment Interaction Loop

1. Select action
2. Step environment
3. Update latent state via `observe_step`
4. Store raw transition in replay

This loop produces the data used to train the world model.

---

# 📦 Replay Buffer

Stores sequences of:

state (obs), action, reward, is_first, is_last, is_terminal

The world model is trained on sampled sequences of length `seq_len`.

---

# 🏗️ World Model Training

The world model is trained using:

L = pred + kl_scale * (dyn + rep)

Where:

- pred = reconstruction + reward + continuation losses
- dyn = KL[ sg(q) || p ]
- rep = KL[ q || sg(p) ]
- sg(·) = stop‑gradient
- free bits applied per latent dimension

Free bits prevent posterior collapse.

---

# 🌌 Imagination (Latent Rollout)

Dreamer never rolls out in pixel space.

Instead:

1. Start from a latent state
2. Predict reward
3. Sample action
4. Apply imagine_step
5. Repeat

Produces imagined trajectories: h[t], z[t], reward[t], action[t].

---

# 📈 Critic Learning (Distributional)

The critic predicts a **two‑hot distribution** over symlog‑transformed returns.

Steps:

1. Compute λ‑returns
2. Apply symlog
3. Encode as two‑hot
4. Train with cross‑entropy

---

# 🎮 Actor Learning (Advantage‑Weighted)

L_actor = -E[ log π(a|s) * adv_norm ] - entropy_bonus

- adv_norm = normalized advantage (p5–p95)
- entropy bonus encourages exploration

---

# 🧩 Full Pipeline Diagram

Environment → World Model → Replay → World Model Training → Imagination → Critic → Actor → Environment

---

# 🧪 Test Suite Alignment

This implementation passes:

- RSSM invariants
- Posterior/prior correctness
- KL structure
- Free bits behavior
- No‑mutation guarantees
- CPU/GPU equivalence
- Actor–critic invariants
- Imagination correctness
- World model update finiteness

All 70 tests green.

---

# 🏁 Summary

This repository implements a clean, modular, and fully test‑aligned Dreamer‑V3 pipeline:

- World Model learns environment dynamics
- Imagination generates trajectories in latent space
- Critic learns distributional values
- Actor learns from normalized advantages
- Trainer orchestrates environment interaction and updates

The architecture is faithful to the Dreamer‑V3 paper and suitable for research‑grade experimentation.
