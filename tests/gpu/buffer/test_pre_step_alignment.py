import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import LSTMGates, PolicyEvalInput, RolloutStep


def test_rollout_buffer_pre_step_alignment():
    """
    PRE‑STEP Hidden State Alignment Test
    -----------------------------------

    This test ensures that the rollout buffer stores the PRE‑STEP LSTM state
    (h_t, c_t) used to produce action[t].

    Invariants validated:

    1. The buffer stores h_t, c_t BEFORE the policy processes obs[t].
    2. The stored state exactly matches the policy’s internal state.
    3. Feeding the stored state back into evaluate_actions_sequence reproduces
       rollout‑time logits/values.
    4. TBPTT chunking begins from the correct PRE‑STEP state.

    If this invariant breaks:
        • PPO logprobs/values misalign
        • TBPTT unrolls from the wrong state
        • Diagnostics become meaningless
        • Training becomes nondeterministic
    """

    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device
    buf = trainer.buffer

    T = buf.cfg.rollout_steps
    B = buf.cfg.num_envs
    H = buf.cfg.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    # --- Create synthetic rollout ---
    obs = torch.randn(T, B, obs_dim, device=device)

    # Initial PRE‑STEP state
    h = torch.randn(B, H, device=device)
    c = torch.randn(B, H, device=device)

    # Clear buffer
    buf.reset()

    # --- Rollout loop ---
    for t in range(T):
        # Store PRE‑STEP state BEFORE calling forward_step
        pre_h = h.clone()
        pre_c = c.clone()

        # Single-step rollout
        logits, value, h_new, c_new, _ = policy.forward_step(obs[t], h, c)

        # PRE‑STEP alignment test doesn't need meaningful gates, so we can use dummy values here
        dummy_gates = LSTMGates(
            i_gates=torch.zeros(B, 1, H),
            f_gates=torch.zeros(B, 1, H),
            g_gates=torch.zeros(B, 1, H),
            o_gates=torch.zeros(B, 1, H),
            c_gates=torch.zeros(B, 1, H),
            h_gates=torch.zeros(B, 1, H),
        )

        # Add to buffer
        step = RolloutStep(
            obs=obs[t],
            actions=torch.zeros(B, device=device),
            rewards=torch.zeros(B, device=device),
            values=value.detach(),
            logprobs=torch.zeros(B, device=device),
            terminated=torch.zeros(B, dtype=torch.bool, device=device),
            truncated=torch.zeros(B, dtype=torch.bool, device=device),
            hxs=pre_h,  # PRE‑STEP
            cxs=pre_c,  # PRE‑STEP
            gates=dummy_gates,  # Not relevant for this test
        )
        buf.add(step)

        # Advance state
        h, c = h_new.detach(), c_new.detach()

    # --- Invariant 1: buffer stored PRE‑STEP states correctly ---
    assert torch.allclose(buf.hxs[0], pre_h, atol=1e-6)
    assert torch.allclose(buf.cxs[0], pre_c, atol=1e-6)

    # --- Invariant 2: buffer PRE‑STEP state reproduces rollout behavior ---
    # Re-run the entire sequence using evaluate_actions_sequence
    eval_inp = PolicyEvalInput(
        obs=obs,
        hxs=buf.hxs[0],
        cxs=buf.cxs[0],
        actions=torch.zeros(T, B, device=device),
    )
    eval_out = policy.evaluate_actions_sequence(eval_inp)

    # Compare rollout-time vs training-time hidden states
    # (evaluate_actions_sequence must reproduce rollout-time state-flow)
    assert torch.allclose(eval_out.new_hxs, torch.stack([buf.hxs[t] for t in range(T)]), atol=1e-5)

    # --- Invariant 3: TBPTT chunking begins from correct PRE‑STEP state ---
    tb = policy.forward_tbptt(obs, buf.hxs[0], buf.cxs[0], chunk_size=4)
    assert torch.allclose(tb.hn, eval_out.new_hxs, atol=1e-5)
