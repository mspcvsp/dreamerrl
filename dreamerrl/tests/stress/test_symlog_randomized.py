import pytest
import torch

from dreamerrl.utils.transforms import symexp, symlog


@pytest.mark.stress
def test_symlog_randomized():
    """
    Fuzz symlog/symexp with extreme random values.
    """
    torch.manual_seed(0)

    x = torch.randn(10_000) * 1000  # huge range
    y = symexp(symlog(x))

    assert torch.isfinite(y).all()
    # Round-trip should be close for moderate values
    mask = x.abs() < 50
    assert torch.allclose(x[mask], y[mask], atol=1e-2)
