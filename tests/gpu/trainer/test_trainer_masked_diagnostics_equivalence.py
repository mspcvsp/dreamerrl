import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import PolicyEvalInput


def reference_masked_diagnostics(h, masks):
    """
    Stateless reference implementation used to validate the trainer.
    h:     (T, B, H)
    masks: (T, B)
    """
    m = masks.unsqueeze(-1)  # (T, B, 1)

    # Saturation
    sat = (h.abs() * m).sum() / m.sum().clamp(min=1)

    # Entropy
    sig = torch.sigmoid(h)
    ent = (-(sig * torch.log(sig + 1e-8)) * m).sum() / m.sum().clamp(min=1)

    # Drift (only valid pairs)
    valid_pairs = (masks[1:] * masks[:-1]).unsqueeze(-1)  # (T-1, B, 1)
    drift = ((h[1:] - h[:-1]).pow(2) * valid_pairs).sum() / valid_pairs.sum().clamp(min=1)

    return {
        "gate_saturation": sat,
        "gate_entropy": ent,
        "drift": drift,
    }


def test_trainer_masked_diagnostics_equivalence():
    """
    Trainer-level mask-aware diagnostics must match the reference implementation.

    This test validates the *training-time* diagnostics path, which is fully
    mask-aware and operates on (T,B,H) tensors produced by evaluate_actions_sequence.

    It ensures:
        • drift ignores transitions across terminated/truncated boundaries
        • saturation/entropy ignore masked timesteps
        • trainer diagnostics == reference implementation
    """
    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device

    T, B, D, H = 6, 2, 4, trainer.state.cfg.lstm.lstm_hidden_size

    obs = torch.randn(T, B, D, device=device)
    h0 = torch.randn(B, H, device=device)
    c0 = torch.randn(B, H, device=device)

    masks = torch.randint(0, 2, (T, B), device=device).float()
    if masks.sum() == 0:
        masks[0, 0] = 1.0

    # Run full sequence evaluation
    eval_out = policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=obs,
            hxs=h0,
            cxs=c0,
            actions=torch.zeros(T, B, dtype=torch.long),
        )
    )

    # Trainer-level diagnostics
    trainer_diag = trainer.compute_scalar_masked_diagnostics(eval_out, masks)

    # Reference diagnostics
    ref_diag = reference_masked_diagnostics(eval_out.new_hxs, masks)

    # Exact equivalence checks
    assert torch.allclose(trainer_diag["gate_saturation"], ref_diag["gate_saturation"])
    assert torch.allclose(trainer_diag["gate_entropy"], ref_diag["gate_entropy"])
    assert torch.allclose(trainer_diag["drift"], ref_diag["drift"])
