import torch

from dreamerrl.models.reward_head import RewardHead


def test_reward_head_cpu_gpu_equivalence():
    if not torch.cuda.is_available():
        return

    B, deter_size, stoch_size, hidden_size = 4, 32, 16, 64

    torch.manual_seed(0)
    head_cpu = RewardHead(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    ).cpu()

    head_gpu = RewardHead(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    ).cuda()

    head_gpu.load_state_dict(head_cpu.state_dict())

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    out_cpu = head_cpu(h, z)
    out_gpu = head_gpu(h.cuda(), z.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu)
