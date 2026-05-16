import pytest
import torch

from dreamerrl.models.categorical_kl import categorical_kl


@pytest.mark.functional
def test_categorical_kl_zero_when_equal():
    B, F, C = 4, 3, 5

    # Random logits → softmax → valid categorical distributions
    logits = torch.randn(B, F, C)
    p = torch.log_softmax(logits, dim=-1)
    q = p.clone()

    # KL(q || p) should be exactly zero when q == p
    kl = categorical_kl(q, p)

    # V3: categorical_kl returns (B,) aggregated across factors
    assert kl.shape == (B,)
    assert torch.allclose(kl, torch.zeros(B), atol=1e-6)
