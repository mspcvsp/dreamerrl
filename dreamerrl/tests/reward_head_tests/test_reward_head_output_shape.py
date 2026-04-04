import torch

from dreamerrl.models.reward_head import RewardHead


def test_reward_head_output_shape():
    B, deter, stoch, hidden = 5, 32, 16, 64
    head = RewardHead(deter_size=deter, stoch_size=stoch, hidden_size=hidden)

    h = torch.randn(B, deter)
    z = torch.randn(B, stoch)

    out = head(h, z)
    assert out.shape == (B, head.num_bins)
