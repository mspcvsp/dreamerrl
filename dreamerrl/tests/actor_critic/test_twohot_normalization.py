import torch

from dreamerrl.utils.twohot import twohot_encode


def test_twohot_normalization():
    y = torch.randn(10)
    out = twohot_encode(y)
    assert torch.allclose(out.sum(dim=-1), torch.ones(10))
