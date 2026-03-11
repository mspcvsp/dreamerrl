import torch

from dreamerrl.models.reward_head import RewardHead


def test_reward_head_gradients_finite():
    B, deter, stoch, hidden = 4, 32, 16, 64
    head = RewardHead(deter_size=deter, stoch_size=stoch, hidden_size=hidden)

    h = torch.randn(B, deter, requires_grad=True)
    z = torch.randn(B, stoch, requires_grad=True)

    out = head(h, z).sum()
    out.backward()

    for p in head.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
