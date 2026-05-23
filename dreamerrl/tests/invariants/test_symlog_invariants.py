import pytest
import torch

from dreamerrl.utils.transforms import symexp, symlog


@pytest.mark.invariants
def test_symlog_symmetry():
    x = torch.linspace(-10, 10, 200)
    assert torch.allclose(symlog(x), -symlog(-x), atol=1e-6)


@pytest.mark.invariants
def test_symlog_small_value_linearity():
    x = torch.linspace(-1e-4, 1e-4, 100)
    assert torch.allclose(symlog(x), x, atol=1e-6)


@pytest.mark.invariants
def test_symlog_roundtrip():
    x = torch.linspace(-10, 10, 200)
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)
