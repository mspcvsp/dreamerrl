import torch

from dreamerrl.training.core.lambda_return import lambda_return


def test_lambda_return_limits():
    T, B = 5, 3
    reward = torch.ones(T, B)
    value = torch.zeros(T + 1, B)

    # λ = 0 → TD(0)
    out0 = lambda_return(reward, value, 1.0, 0.0)
    assert torch.allclose(out0, torch.ones(T, B))

    # λ = 1 → Monte Carlo
    out1 = lambda_return(reward, value, 1.0, 1.0)
    assert torch.allclose(out1, torch.tensor([[5], [4], [3], [2], [1]]).repeat(1, B))
