import torch


def test_gae_gpu(deterministic_trainer, fake_rollout):
    device = torch.device("cuda")
    trainer = deterministic_trainer(device=device)

    rollout = fake_rollout(device=device, batch_size=8, seq_len=16)

    advantages, returns = trainer.compute_gae(
        rollout.value,
        rollout.reward,
        rollout.done,
        rollout.mask,
        gamma=0.99,
        lam=0.95,
    )

    # Basic invariants
    assert advantages.shape == rollout.reward.shape
    assert returns.shape == rollout.reward.shape

    # No NaNs
    assert not torch.isnan(advantages).any()
    assert not torch.isnan(returns).any()

    # Advantage mean should be near zero (GAE property)
    assert abs(advantages.mean().item()) < 1e-3
