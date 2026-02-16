import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import PolicyEvalInput
from tests.helpers.diagnostics_helpers import EvalOutProxy


def test_per_unit_diagnostics_tbptt_alignment_gpu():
    """
    Trainer-level invariant:
    ------------------------
    Per-unit LSTM diagnostics (means, norms, saturation, entropy) must be
    TBPTT-aligned.

    That is:
        compute_lstm_unit_diagnostics(full_sequence)
    must match:
        compute_lstm_unit_diagnostics(stitched_TBPTT_chunks)

    This ensures:
        - TBPTT does not distort interpretability metrics
        - drift, saturation, entropy remain comparable across training modes
        - replay-based diagnostics match rollout-based diagnostics
        - chunking strategy does not affect diagnostics correctness
    """

    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    policy.eval()

    device = trainer.device
    T = 24
    B = 2
    H = trainer.state.cfg.lstm.lstm_hidden_size
    chunk_len = 5

    obs = torch.randn(T, B, trainer.state.env_info.flat_obs_dim, device=device)
    actions = torch.randint(0, trainer.state.env_info.action_dim, (T, B, 1), device=device)

    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    # -----------------------------
    # Full-sequence evaluation
    # -----------------------------
    full_out = policy.evaluate_actions_sequence(PolicyEvalInput(obs=obs, hxs=h0, cxs=c0, actions=actions))
    mask_full = torch.ones(T, B, device=device)
    full_diag = trainer.compute_lstm_unit_diagnostics(full_out, mask_full).require()

    # -----------------------------
    # TBPTT chunked evaluation
    # -----------------------------
    h = h0
    c = c0

    all_i = []
    all_f = []
    all_g = []
    all_o = []
    all_h = []
    all_c = []

    for start in range(0, T, chunk_len):
        end = min(start + chunk_len, T)

        out = policy.evaluate_actions_sequence(
            PolicyEvalInput(obs=obs[start:end], hxs=h, cxs=c, actions=actions[start:end])
        )

        all_i.append(out.gates.i_gates)
        all_f.append(out.gates.f_gates)
        all_g.append(out.gates.g_gates)
        all_o.append(out.gates.o_gates)
        all_h.append(out.new_hxs)
        all_c.append(out.new_cxs)

        h = out.new_hxs[-1]
        c = out.new_cxs[-1]

    # Stitch back together
    i_tbptt = torch.cat(all_i, dim=0)
    f_tbptt = torch.cat(all_f, dim=0)
    g_tbptt = torch.cat(all_g, dim=0)
    o_tbptt = torch.cat(all_o, dim=0)
    h_tbptt = torch.cat(all_h, dim=0)
    c_tbptt = torch.cat(all_c, dim=0)

    # Build proxy using shared helper
    proxy = EvalOutProxy(
        gates=full_out.gates.__class__(
            i_gates=i_tbptt,
            f_gates=f_tbptt,
            g_gates=g_tbptt,
            o_gates=o_tbptt,
            c_gates=full_out.gates.c_gates,
            h_gates=full_out.gates.h_gates,
        ),
        new_hxs=h_tbptt,
        new_cxs=c_tbptt,
    )

    tbptt_diag = trainer.compute_lstm_unit_diagnostics(proxy, mask_full).require()

    # -----------------------------
    # Compare per-unit metrics
    # -----------------------------
    assert torch.allclose(full_diag.i_mean, tbptt_diag.i_mean, atol=1e-6)
    assert torch.allclose(full_diag.f_mean, tbptt_diag.f_mean, atol=1e-6)
    assert torch.allclose(full_diag.g_mean, tbptt_diag.g_mean, atol=1e-6)
    assert torch.allclose(full_diag.o_mean, tbptt_diag.o_mean, atol=1e-6)

    assert torch.allclose(full_diag.h_norm, tbptt_diag.h_norm, atol=1e-6)
    assert torch.allclose(full_diag.c_norm, tbptt_diag.c_norm, atol=1e-6)

    assert torch.allclose(full_diag.saturation.i_sat_low, tbptt_diag.saturation.i_sat_low, atol=1e-6)
    assert torch.allclose(full_diag.saturation.i_sat_high, tbptt_diag.saturation.i_sat_high, atol=1e-6)
    assert torch.allclose(full_diag.saturation.f_sat_low, tbptt_diag.saturation.f_sat_low, atol=1e-6)
    assert torch.allclose(full_diag.saturation.f_sat_high, tbptt_diag.saturation.f_sat_high, atol=1e-6)

    assert torch.allclose(full_diag.entropy.i_entropy, tbptt_diag.entropy.i_entropy, atol=1e-6)
    assert torch.allclose(full_diag.entropy.f_entropy, tbptt_diag.entropy.f_entropy, atol=1e-6)
    assert torch.allclose(full_diag.entropy.g_entropy, tbptt_diag.entropy.g_entropy, atol=1e-6)
    assert torch.allclose(full_diag.entropy.o_entropy, tbptt_diag.entropy.o_entropy, atol=1e-6)
