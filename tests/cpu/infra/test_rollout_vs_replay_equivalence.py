import pytest
import torch

pytestmark = pytest.mark.cpu


def test_rollout_vs_replay_equivalence(deterministic_trainer):
    """
    Rollout and replay should be numerically consistent:

    - logprobs: small mean/max absolute difference
    - final hidden states: small mean absolute difference
    """

    trainer = deterministic_trainer

    trainer.state.cfg.lstm.dropconnect_p = 0.0
    trainer.policy.eval()

    trainer.collect_rollout()
    buf = trainer.buffer

    rollout_logprobs = buf.logprobs.clone()  # (T, B, 1)
    rollout_hxs = buf.hxs.clone()  # (T+1, B, H)
    rollout_cxs = buf.cxs.clone()  # (T+1, B, H)

    eval_output = trainer.replay_policy_on_rollout()
    lp_roll = rollout_logprobs.cpu()
    lp_repl = eval_output.logprobs.cpu()

    # --- logprobs: structural + statistical ---
    assert lp_roll.shape == lp_repl.shape
    assert torch.isfinite(lp_roll).all()
    assert torch.isfinite(lp_repl).all()

    diff = (lp_roll - lp_repl).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    assert mean_diff < 0.02, f"mean logprob diff too large: {mean_diff}"
    assert max_diff < 0.10, f"max logprob diff too large: {max_diff}"

    # --- hidden states: compare final step only ---
    # rollout final: (B, H)
    h_roll = rollout_hxs[-1].cpu()
    c_roll = rollout_cxs[-1].cpu()

    # replay returns (T, B, H) → take last time step
    h_repl = eval_output.new_hxs[-1].cpu()
    c_repl = eval_output.new_cxs[-1].cpu()

    assert h_roll.shape == h_repl.shape
    assert c_roll.shape == c_repl.shape

    h_diff = (h_roll - h_repl).abs().mean().item()
    c_diff = (c_roll - c_repl).abs().mean().item()

    assert h_diff < 0.02, f"mean hxs diff too large: {h_diff}"
    assert c_diff < 0.02, f"mean cxs diff too large: {c_diff}"
