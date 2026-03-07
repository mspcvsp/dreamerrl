import torch

from dreamerrl.models.reward_head import RewardHead


def test_reward_head_deterministic():
    B, deter_size, stoch_size, hidden_size = 4, 32, 16, 64

    torch.manual_seed(0)
    head1 = RewardHead(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    )

    torch.manual_seed(0)
    head2 = RewardHead(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    )

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    out1 = head1(h, z)
    out2 = head2(h, z)

    torch.testing.assert_close(out1, out2)
