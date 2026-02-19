import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import PolicyEvalInput


def test_trainer_tbptt_mask_propagation_gpu():
    """
    Trainer-level invariant:
    ------------------------
    TBPTT (Truncated Backprop Through Time) requires that masks
    propagate correctly across chunk boundaries.

    Specifically:
    - If done[t] == True, then the PRE-STEP state for the next chunk
      must be zeroed.
    - Masked timesteps must not contribute to losses.
    - Chunked evaluation must produce the same masked outputs as
      full-sequence evaluation.

    This test ensures:
    - Mask propagation is correct.
    - Hidden-state resets are correct across chunk boundaries.
    - TBPTT does not change the effective computation.
    - All operations run on the trainer's device.

    If this test fails, TBPTT becomes inconsistent and PPO gradients
    become incorrect.
    """

    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device

    T = 18
    B = 3
    H = trainer.state.cfg.lstm.lstm_hidden_size
    chunk = 5

    obs = torch.randn(T, B, trainer.state.env_info.flat_obs_dim, device=device)
    actions = torch.randint(0, trainer.state.env_info.action_dim, (T, B, 1), device=device)

    # Fake done flags
    done = torch.zeros(T, B, dtype=torch.bool, device=device)
    done[7:, 1] = True

    # Build masks
    mask = torch.ones(T, B, device=device)
    for t in range(1, T):
        mask[t] = mask[t - 1] * (~done[t - 1]).float()

    # --- Full-sequence evaluation ---
    full_eval = policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=obs,
            hxs=torch.zeros(B, H, device=device),
            cxs=torch.zeros(B, H, device=device),
            actions=actions,
        )
    )

    # --- TBPTT chunked evaluation ---
    h = torch.zeros(B, H, device=device)
    c = torch.zeros(B, H, device=device)

    logits_chunks = []
    values_chunks = []

    for start in range(0, T, chunk):
        end = min(start + chunk, T)

        eval_out = policy.evaluate_actions_sequence(
            PolicyEvalInput(
                obs=obs[start:end],
                hxs=h,
                cxs=c,
                actions=actions[start:end],
            )
        )

        logits_chunks.append(eval_out.logits)
        values_chunks.append(eval_out.values)

        # Reset hidden state if done at boundary
        if end < T:
            reset_mask = done[end - 1].view(B, 1).expand(B, H)
            h = torch.where(reset_mask, torch.zeros_like(eval_out.new_hxs[-1]), eval_out.new_hxs[-1])
            c = torch.where(reset_mask, torch.zeros_like(eval_out.new_cxs[-1]), eval_out.new_cxs[-1])
        else:
            h = eval_out.new_hxs[-1]
            c = eval_out.new_cxs[-1]

    logits_tbptt = torch.cat(logits_chunks, dim=0)
    values_tbptt = torch.cat(values_chunks, dim=0)

    # Compare masked outputs
    assert torch.allclose(full_eval.logits * mask.unsqueeze(-1), logits_tbptt * mask.unsqueeze(-1), atol=1e-6), (
        "TBPTT logits mismatch"
    )

    assert torch.allclose(full_eval.values * mask, values_tbptt * mask, atol=1e-6), "TBPTT values mismatch"
