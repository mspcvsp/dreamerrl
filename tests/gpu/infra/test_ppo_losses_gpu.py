import torch


def test_ppo_losses_gpu(deterministic_trainer, fake_batch):
    device = torch.device("cuda")
    trainer = deterministic_trainer(device=device)

    batch = fake_batch(device=device, batch_size=32, seq_len=16)

    losses = trainer.compute_ppo_loss(
        obs=batch.obs,
        actions=batch.actions,
        old_logp=batch.old_logp,
        advantages=batch.advantages,
        returns=batch.returns,
        values=batch.values,
        h0=batch.h0,
        c0=batch.c0,
    )

    # Losses must be finite
    for k, v in losses.items():
        assert torch.isfinite(v).all()

    # Policy loss should not explode
    assert abs(losses["policy_loss"].item()) < 10.0

    # Value loss should be positive
    assert losses["value_loss"].item() >= 0
