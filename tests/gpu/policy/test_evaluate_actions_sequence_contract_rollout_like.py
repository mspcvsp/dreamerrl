import torch

from dreamerrl.policy import PolicyEvalInput
from dreamerrl.trainer import LSTMPPOTrainer


def test_evaluate_actions_sequence_contract_rollout_like():
    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device

    T = 8
    B = 4
    H = trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim
    act_dim = trainer.state.env_info.action_dim

    obs = torch.randn(T, B, obs_dim, device=device, requires_grad=True)
    h0 = torch.randn(B, H, device=device, requires_grad=True)
    c0 = torch.randn(B, H, device=device, requires_grad=True)

    # Rollout‑style actions: (T, B, 1)
    actions = torch.randint(0, act_dim, (T, B, 1), device=device)

    eval_in = PolicyEvalInput(
        obs=obs,
        hxs=h0,
        cxs=c0,
        actions=actions,
    )

    out = policy.evaluate_actions_sequence(eval_in)

    # Shapes: time‑major, PPO‑ready
    assert out.logits.shape[:2] == (T, B)
    assert out.values.shape == (T, B)
    assert out.logprobs.shape == (T, B)
    assert out.entropy.shape == (T, B)
    assert out.new_hxs.shape == (T, B, H)
    assert out.new_cxs.shape == (T, B, H)

    # Gradients: PPO needs these
    assert out.logits.requires_grad
    assert out.values.requires_grad
    assert out.logprobs.requires_grad
    assert out.entropy.requires_grad

    # Hidden states must be detached (no grad across rollout boundaries)
    assert not out.new_hxs.requires_grad
    assert not out.new_cxs.requires_grad

    # Gates are diagnostics only
    g = out.gates
    for g_tensor in (g.i_gates, g.f_gates, g.g_gates, g.o_gates, g.h_gates, g.c_gates):
        assert g_tensor.shape[:2] == (T, B)
        assert not g_tensor.requires_grad
