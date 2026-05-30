# dreamerrl/tests/invariants/test_kl_freebits_monotonicity.py
import pytest
import torch


@pytest.mark.invariants
def test_kl_freebits_monotonicity(latent):
    """
    Free-bits must never increase KL and must never produce negative KL.
    """
    logits_p = torch.randn(32, latent.num_classes)
    logits_q = torch.randn(32, latent.num_classes)

    kl_raw = latent.kl(logits_p, logits_q, free_bits=0.0)
    kl_fb = latent.kl(logits_p, logits_q, free_bits=0.1)

    assert torch.all(kl_fb <= kl_raw + 1e-12)
    assert torch.all(kl_fb >= 0)
