import torch

from lstmppo.buffer import RecurrentRolloutBuffer
from lstmppo.trainer_state import TrainerState  # type hint only


def test_gae_gpu(trainer_state: TrainerState):
    cfg = trainer_state.cfg
    buf_cfg = cfg.buffer_config
    device = torch.device("cuda")

    buf = RecurrentRolloutBuffer(trainer_state, device)

    T = buf_cfg.rollout_steps
    B = buf_cfg.num_envs

    rewards = torch.linspace(0, 1, T, device=device).unsqueeze(1).repeat(1, B)
    buf.rewards.copy_(rewards)

    buf.values.zero_()
    buf.terminated.zero_()
    buf.truncated.zero_()

    last_value = torch.zeros(B, device=device)

    buf.compute_returns_and_advantages(last_value)

    assert abs(buf.rewards.mean().item()) < 1e-6
    assert abs(buf.rewards.std(unbiased=False).item() - 1.0) < 1e-6

    assert buf.advantages.shape == (T, B)
    assert buf.returns.shape == (T, B)

    assert abs(buf.returns.mean().item()) < 1e-6
    assert abs(buf.returns.std(unbiased=False).item() - 1.0) < 1e-6

    flat_adv = buf.advantages.flatten()
    flat_ret = buf.returns.flatten()
    corr = torch.corrcoef(torch.stack([flat_adv, flat_ret]))[0, 1]
    assert corr > 0.999
