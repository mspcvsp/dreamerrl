import torch


def test_hidden_state_lifecycle_gpu(deterministic_trainer, fake_rollout):
    device = torch.device("cuda")
    trainer = deterministic_trainer.to(device)

    # Create rollout with a done in the middle
    rollout = fake_rollout(
        device=device,
        batch_size=4,
        seq_len=10,
        force_done_at=5,
    )

    out = trainer.policy.forward_sequence(rollout.obs, rollout.h0, rollout.c0, done=rollout.done)

    # Hidden state after done must reset
    assert torch.allclose(out.hn[:, :, 5], torch.zeros_like(out.hn[:, :, 5]))
    assert torch.allclose(out.cn[:, :, 5], torch.zeros_like(out.cn[:, :, 5]))

    # Hidden state before done must not reset
    assert not torch.allclose(out.hn[:, :, 4], torch.zeros_like(out.hn[:, :, 4]))
