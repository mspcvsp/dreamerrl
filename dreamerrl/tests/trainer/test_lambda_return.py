import pytest
import torch

from dreamerrl.training.test_trainer import lambda_return


@pytest.mark.trainer
def test_lambda_return(device):
    T, B = 5, 2

    reward = torch.randn(T, B, device=device)
    value = torch.randn(T + 1, B, device=device)
    discount = 0.99

    # General λ-return
    out = lambda_return(reward, value, discount, lam=0.95)
    assert out.shape == (T, B)
    assert torch.isfinite(out).all()

    # λ = 1 → Monte Carlo return
    mc = lambda_return(reward, value, discount, lam=1.0)
    expected_mc = torch.zeros_like(reward)
    for t in reversed(range(T)):
        expected_mc[t] = reward[t] + discount * (expected_mc[t + 1] if t + 1 < T else value[-1])
    assert torch.allclose(mc, expected_mc, atol=1e-5)

    # λ = 0 → TD(0)
    td0 = lambda_return(reward, value, discount, lam=0.0)
    expected_td0 = reward + discount * value[1:]
    assert torch.allclose(td0, expected_td0, atol=1e-5)
