import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import Config


def test_ppo_losses_gpu():
    cfg = Config()
    cfg.trainer.cuda = True

    trainer = LSTMPPOTrainer(cfg)

    T = cfg.trainer.rollout_steps
    B = cfg.env.num_envs
    device = torch.device("cuda")

    values = torch.randn(T, B, device=device)
    old_values = torch.randn(T, B, device=device)
    returns = torch.randn(T, B, device=device)
    adv = torch.randn(T, B, device=device)

    new_logp = torch.randn(T, B, device=device)
    old_logp = torch.randn(T, B, device=device)

    mask = torch.ones(T, B, device=device)

    policy_loss, value_loss, approx_kl, clip_frac = trainer.compute_losses(
        values=values,
        new_logp=new_logp,
        old_logp=old_logp,
        old_values=old_values,
        returns=returns,
        adv=adv,
        mask=mask,
    )

    assert policy_loss.ndim == 0
    assert value_loss.ndim == 0
    assert approx_kl.ndim == 0
    assert clip_frac.ndim == 0

    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(approx_kl)
    assert torch.isfinite(clip_frac)
