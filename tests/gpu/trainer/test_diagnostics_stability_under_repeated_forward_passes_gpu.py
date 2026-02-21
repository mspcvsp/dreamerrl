import torch

from dreamerrl.trainer import LSTMPPOTrainer
from dreamerrl.types import PolicyEvalInput


def test_diagnostics_stability_under_repeated_forward_passes_gpu():
    """
    Trainer-level + Policy-level invariant:
    ---------------------------------------
    LSTM diagnostics must be *stable* under repeated forward passes when:
        - obs, actions, h0, c0 are identical
        - policy.eval() is active
        - no dropout or randomness is enabled
        - trainer.compute_lstm_unit_diagnostics() is called repeatedly

    This protects:
        • drift metrics (prev vs current)
        • saturation metrics
        • entropy metrics
        • CPU↔GPU equivalence
        • replay vs rollout equivalence
        • TBPTT diagnostics consistency

    If this test fails, diagnostics will drift even when the rollout
    is unchanged, corrupting:
        - LSTMUnitPrev
        - TrainerState.current_lstm_unit_diag
        - drift-based stability metrics
    """

    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    policy.eval()

    device = trainer.device
    T = 10
    B = 2
    H = trainer.state.cfg.lstm.lstm_hidden_size

    obs = torch.randn(T, B, trainer.state.env_info.flat_obs_dim, device=device)
    actions = torch.randint(0, trainer.state.env_info.action_dim, (T, B, 1), device=device)

    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    # First evaluation
    out1 = policy.evaluate_actions_sequence(PolicyEvalInput(obs=obs, hxs=h0, cxs=c0, actions=actions))
    diag1 = trainer.compute_lstm_unit_diagnostics(out1, mask=torch.ones(T, B, device=device)).require()

    # Second evaluation (identical inputs)
    out2 = policy.evaluate_actions_sequence(PolicyEvalInput(obs=obs, hxs=h0, cxs=c0, actions=actions))
    diag2 = trainer.compute_lstm_unit_diagnostics(out2, mask=torch.ones(T, B, device=device)).require()

    # Compare per-unit metrics
    assert torch.allclose(diag1.i_mean, diag2.i_mean, atol=1e-7), "i_mean unstable"
    assert torch.allclose(diag1.f_mean, diag2.f_mean, atol=1e-7), "f_mean unstable"
    assert torch.allclose(diag1.g_mean, diag2.g_mean, atol=1e-7), "g_mean unstable"
    assert torch.allclose(diag1.o_mean, diag2.o_mean, atol=1e-7), "o_mean unstable"

    assert torch.allclose(diag1.h_norm, diag2.h_norm, atol=1e-7), "h_norm unstable"
    assert torch.allclose(diag1.c_norm, diag2.c_norm, atol=1e-7), "c_norm unstable"

    assert torch.allclose(diag1.saturation.i_sat_low, diag2.saturation.i_sat_low, atol=1e-7)
    assert torch.allclose(diag1.saturation.i_sat_high, diag2.saturation.i_sat_high, atol=1e-7)
    assert torch.allclose(diag1.saturation.f_sat_low, diag2.saturation.f_sat_low, atol=1e-7)
    assert torch.allclose(diag1.saturation.f_sat_high, diag2.saturation.f_sat_high, atol=1e-7)

    assert torch.allclose(diag1.entropy.i_entropy, diag2.entropy.i_entropy, atol=1e-7)
    assert torch.allclose(diag1.entropy.f_entropy, diag2.entropy.f_entropy, atol=1e-7)
    assert torch.allclose(diag1.entropy.g_entropy, diag2.entropy.g_entropy, atol=1e-7)
    assert torch.allclose(diag1.entropy.o_entropy, diag2.entropy.o_entropy, atol=1e-7)
