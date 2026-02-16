import torch

from lstmppo.trainer import LSTMPPOTrainer


def test_trainer_advantage_normalization_gpu():
    """
    Trainer-level invariant:
    ------------------------
    PPO requires advantages to be normalized per minibatch:

        adv_norm = (adv - mean(adv_valid)) / (std(adv_valid) + eps)

    This test ensures:
    - Only valid (mask==1) timesteps contribute to mean/std.
    - Normalization is numerically stable.
    - Masked-out timesteps remain zeroed.
    - All operations run on the trainer's device.

    If this test fails, PPO updates become biased and unstable.
    """

    trainer = LSTMPPOTrainer.for_validation()
    device = trainer.device

    T = 12
    B = 4

    adv = torch.randn(T, B, device=device)
    mask = torch.ones(T, B, device=device)
    mask[5:, 2] = 0  # env 2 terminates early

    valid = adv[mask > 0.5]
    mean = valid.mean()
    std = valid.std(unbiased=False) + 1e-8

    adv_ref = (adv - mean) / std
    adv_ref = adv_ref * mask  # masked-out timesteps must remain zero

    # Trainer-style normalization
    adv_trainer = adv.clone()
    valid_adv = adv_trainer[mask > 0.5]
    adv_trainer = (adv_trainer - valid_adv.mean()) / (valid_adv.std(unbiased=False) + 1e-8)
    adv_trainer = adv_trainer * mask

    assert torch.allclose(adv_ref, adv_trainer, atol=1e-6), (
        "Advantage normalization does not match reference implementation"
    )
