# Policy‑Level Invariants

These tests validate the behavior of the policy network:

- Output shapes
- Minibatch consistency
- Determinism in eval mode
- LSTM‑only mode correctness
- Encoder identity behavior
- Actor identity behavior
- Combined identity path

These invariants ensure the policy is stable, predictable, and regression‑proof.

LSTM State‑Flow Invariant
=========================

Each timestep t has two distinct LSTM states:

    PRE‑STEP:  (h_t,   c_t)
    POST‑STEP: (h_{t+1}, c_{t+1})

Rollout-time:
-------------
    (h_t, c_t) --forward_step--> (h_{t+1}, c_{t+1})

Training-time (evaluate_actions_sequence):
-----------------------------------------
    PRE‑STEP states come from the rollout buffer:
        hxs[t] = h_t
        cxs[t] = c_t

    POST‑STEP states come from the LSTM unroll:
        new_hxs[t] = h_{t+1}
        new_cxs[t] = c_{t+1}

TBPTT:
------
Chunks must begin at PRE‑STEP states:

    chunk k starts at t0:
        hxs0 = h_t0
        cxs0 = c_t0

This ensures:
    • deterministic state-flow
    • correct PPO logprobs/values
    • correct auxiliary predictions
    • correct drift/saturation/entropy diagnostics
    • CPU/GPU equivalence
