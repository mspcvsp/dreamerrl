import torch

from dreamerrl.models.categorical_kl import categorical_kl


def test_categorical_kl_zero_when_equal():
    B, F, C = 4, 3, 5
    logits = torch.randn(B, F, C)
    p = torch.log_softmax(logits, dim=-1)
    q = p.clone()

    kl = categorical_kl(q, p)  # KL(q || p)
    assert kl.shape == (B,)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)
