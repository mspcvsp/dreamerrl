import torch

from dreamerrl.models.categorical_kl import structured_kl
from dreamerrl.utils.types import KLConfig


def test_structured_kl_behavior():
    q = torch.tensor([[[0.8, 0.2]]])
    p = torch.tensor([[[0.4, 0.6]]])
    reference = (q * (q.log() - p.log())).sum()
    for factors in [1, 4, 16]:
        out = structured_kl(q.repeat(1, factors, 1), p.repeat(1, factors, 1), KLConfig(free_nats=0))
        torch.testing.assert_close(out["kl_dyn"], (reference * factors).reshape(1))
        torch.testing.assert_close(out["kl_rep"], (reference * factors).reshape(1))
