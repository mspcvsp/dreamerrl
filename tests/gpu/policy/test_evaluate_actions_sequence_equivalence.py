import torch

from lstmppo.policy import PolicyEvalInput
from lstmppo.trainer import LSTMPPOTrainer


def test_evaluate_actions_sequence_equivalence():
    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device

    T = 12
    B = 3
    H = trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim
    act_dim = trainer.state.env_info.action_dim

    obs = torch.randn(T, B, obs_dim, device=device)
    h0 = torch.randn(B, H, device=device)
    c0 = torch.randn(B, H, device=device)

    # --- Rollout path (authoritative) ---
    full = policy.forward_sequence(obs, h0, c0)
    logits_full = full.logits  # (T, B, A)
    values_full = full.value  # (T, B)
    h_full = full.hn  # (T, B, H)
    c_full = full.cn  # (T, B, H)

    # --- Training path (must match rollout) ---
    actions = torch.randint(0, act_dim, (T, B, 1), device=device)

    eval_in = PolicyEvalInput(
        obs=obs,
        hxs=h0,  # (B, H)
        cxs=c0,  # (B, H)
        actions=actions,  # (T, B, 1)
    )

    eval_out = policy.evaluate_actions_sequence(eval_in)

    # Eval outputs may be laid out differently; enforce same shape before compare
    assert eval_out.logits.numel() == logits_full.numel()
    assert eval_out.values.numel() == values_full.numel()
    assert eval_out.new_hxs.numel() == h_full.numel()
    assert eval_out.new_cxs.numel() == c_full.numel()

    logits_eval = eval_out.logits.view_as(logits_full)
    values_eval = eval_out.values.view_as(values_full)
    h_eval = eval_out.new_hxs.view_as(h_full)
    c_eval = eval_out.new_cxs.view_as(c_full)

    assert torch.allclose(logits_full, logits_eval, atol=1e-6)
    assert torch.allclose(values_full, values_eval, atol=1e-6)
    assert torch.allclose(h_full, h_eval, atol=1e-6)
    assert torch.allclose(c_full, c_eval, atol=1e-6)
