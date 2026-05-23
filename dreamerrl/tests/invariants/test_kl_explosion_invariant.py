import torch

from dreamerrl.models.categorical_kl import categorical_kl


def test_kl_explosion_invariant():
    B, K, C = 4, 30, 32
    logits_p = torch.randn(B, K, C) * 50
    logits_q = torch.randn(B, K, C) * 50

    kl = categorical_kl(logits_p, logits_q)
    assert torch.isfinite(kl).all()
