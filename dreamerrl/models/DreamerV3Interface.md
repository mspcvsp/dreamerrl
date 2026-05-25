# Dreamer‑V3 Minimal Interface Diagram
A clean overview of how data flows through Dreamer‑V3:
environment → trainer → replay buffer → world model → actor/critic → imagination → updates.

---

## 1. Environment Step (runtime interface)

PopGymVecEnv.step() produces:

    state        (obs_t)
    reward       (r_t)
    is_first     (episode boundary)
    is_last      (termination or truncation)
    is_terminal  (true terminal vs. time-limit)
    info

These flags are **runtime control signals**, not training targets.

---

## 2. Trainer Collects Transitions

trainer.collect_env_steps() converts env output into replay‑buffer‑ready tensors:

    obs_t
    action_t
    reward_t
    done_t = is_last.float()

Only **done** is stored.

---

## 3. Replay Buffer (training interface)

ReplayBuffer stores **only what training needs**:

    obs[t]
    action[t]
    reward[t]
    done[t]

No is_first, is_last, is_terminal.

Dreamer‑V3 reconstructs continuation from:

    cont_target = 1 - done

---

## 4. World Model Observe Step (RSSM update)

world.observe_step(prev_state, obs_t, action_t, reward_t):

    encode obs_t → embed_t
    posterior q(z_t | h_{t-1}, embed_t)
    prior     p(z_t | h_{t-1})
    deterministic transition h_t = f(h_{t-1}, action_t)
    decode obs_t
    predict reward_t
    predict continuation_t
    compute KL(q || p)

Flags are **not used** here.

---

## 5. World Model Training Step

Takes sequences from replay buffer:

    batch:
        obs[B, L]
        action[B, L]
        reward[B, L]
        done[B, L]

Computes:

    reconstruction loss
    reward loss
    continuation loss
    KL_dyn + KL_rep

No flags.

---

## 6. Actor & Critic Training (Imagination)

world.imagine_step():

    actor(h, z) → action logits
    sample action
    RSSM transition → h'
    prior → z'
    rollout imagined trajectory

Critic learns value distribution over imagined rollouts.

---

## 7. Full Dreamer‑V3 Loop

Environment → (state, reward, flags)
        ↓
Trainer.collect_env_steps()
        ↓
ReplayBuffer.add(obs, action, reward, done)
        ↓
ReplayBuffer.sample(batch)
        ↓
WorldModelTrainingStep(batch)
        ↓
ActorCriticUpdate(imagined_rollouts)
        ↓
Repeat

Flags are used **only** at the environment boundary.
Training uses **only** obs, action, reward, done.
