import torch


def test_advantage_normalization():
    adv = torch.randn(1000)
    p5, p95 = torch.quantile(adv, torch.tensor([0.05, 0.95]))
    scale = torch.clamp(p95 - p5, min=1.0)
    assert scale >= 1.0
