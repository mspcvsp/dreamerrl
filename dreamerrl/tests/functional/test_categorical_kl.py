import pytest
import torch

from dreamerrl.models.categorical_kl import categorical_kl


@pytest.mark.functional
def test_categorical_kl_zero_when_equal():
    B, F, C = 4, 3, 5

    logits = torch.randn(B, F, C)
    p = torch.softmax(logits, dim=-1)
    q = p.clone()

    kl = categorical_kl(q, p)

    # V3: categorical_kl returns per-factor KL: (B, F)
    assert kl.shape == (B, F)

    # Aggregated KL must be zero
    assert torch.allclose(kl.sum(dim=-1), torch.zeros(B))
