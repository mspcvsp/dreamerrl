import torch


def test_rollout_vs_replay_equivalence_gpu(deterministic_trainer, fake_rollout, fake_buffer_loader):
    device = torch.device("cuda")
    trainer = deterministic_trainer
    trainer.policy.to(device)

    # Generate a fake rollout on GPU
    rollout = fake_rollout(device=device, batch_size=8, seq_len=16)

    # Load rollout into replay buffer
    replay = fake_buffer_loader(rollout, device=device)

    # Compute policy outputs from rollout
    rollout_out = trainer.policy.forward_sequence(rollout.obs, rollout.h0, rollout.c0)

    # Compute policy outputs from replay
    replay_out = trainer.policy.forward_sequence(replay.obs, replay.h0, replay.c0)

    # Compare logits, values, and hidden states
    assert torch.allclose(rollout_out.logits, replay_out.logits, atol=1e-6)
    assert torch.allclose(rollout_out.value, replay_out.value, atol=1e-6)
    assert torch.allclose(rollout_out.hn, replay_out.hn, atol=1e-6)
    assert torch.allclose(rollout_out.cn, replay_out.cn, atol=1e-6)
