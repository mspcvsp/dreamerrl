import torch

from dreamerrl.utils.transforms import symexp, symlog


def test_symlog_symexp_roundtrip():
    x = torch.linspace(-100, 100, steps=257)
    y = symexp(symlog(x))
    assert torch.allclose(x, y, atol=1e-4, rtol=1e-4)
