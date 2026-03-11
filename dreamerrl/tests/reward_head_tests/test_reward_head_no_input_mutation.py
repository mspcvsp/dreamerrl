import torch

from dreamerrl.models.reward_head import RewardHead


def test_reward_head_no_input_mutation():
    B, deter, stoch, hidden = 4, 32, 16, 64
    head = RewardHead(deter_size=deter, stoch_size=stoch, hidden_size=hidden)

    h = torch.randn(B, deter)
    z = torch.randn(B, stoch)

    h_clone = h.clone()
    z_clone = z.clone()

    _ = head(h, z)

    torch.testing.assert_close(h, h_clone)
    torch.testing.assert_close(z, z_clone)
