import torch

from dreamerrl.utils.transforms import symexp, symlog


def test_symlog_inverse_consistency():
    x = torch.linspace(-50, 50, 500)
    y = symexp(symlog(x))
    assert torch.allclose(x, y, atol=1e-3)
