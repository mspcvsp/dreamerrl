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

    actions = torch.randint(0, act_dim, (T, B), device=device)

    # --- Rollout path ---
    full = policy.forward_sequence(obs, h0, c0)
    logits_full = full.logits
    values_full = full.value
    h_full = full.hn
    c_full = full.cn

    # --- Training path ---
    eval_in = PolicyEvalInput(
        obs=obs,
        hxs=h0.expand(T, B, H),
        cxs=c0.expand(T, B, H),
        actions=actions,
    )

    eval_out = policy.evaluate_actions_sequence(eval_in)

    logits_eval = eval_out.logits
    values_eval = eval_out.values
    h_eval = eval_out.new_hxs
    c_eval = eval_out.new_cxs

    assert torch.allclose(logits_full, logits_eval, atol=1e-6)
    assert torch.allclose(values_full, values_eval, atol=1e-6)
    assert torch.allclose(h_full, h_eval, atol=1e-6)
    assert torch.allclose(c_full, c_eval, atol=1e-6)
