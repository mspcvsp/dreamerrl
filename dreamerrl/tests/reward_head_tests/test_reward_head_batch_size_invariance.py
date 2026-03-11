import torch

from dreamerrl.models.reward_head import RewardHead


def test_reward_head_batch_size_invariance():
    deter, stoch, hidden = 32, 16, 64
    head = RewardHead(deter_size=deter, stoch_size=stoch, hidden_size=hidden)

    h1 = torch.randn(1, deter)
    z1 = torch.randn(1, stoch)

    h4 = h1.repeat(4, 1)
    z4 = z1.repeat(4, 1)

    out1 = head(h1, z1)
    out4 = head(h4, z4)

    torch.testing.assert_close(out4, out1.repeat(4, 1))
