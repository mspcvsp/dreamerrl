import torch

from dreamerrl.utils.twohot import value_from_logits


def test_value_from_logits_shape():
    logits = torch.randn(8, 41)
    v = value_from_logits(logits)
    assert v.shape == (8,)
