import torch


def test_tbptt_replay_equivalence_gpu(deterministic_trainer, fake_rollout, fake_buffer_loader):
    device = torch.device("cuda")
    trainer = deterministic_trainer.to(device)

    rollout = fake_rollout(device=device, batch_size=4, seq_len=32)
    replay = fake_buffer_loader(rollout, device=device, chunk_size=8)

    # TBPTT forward pass
    tbptt_out = trainer.policy.forward_tbptt(replay.obs, replay.h0, replay.c0, chunk_size=8)

    # Full unrolled forward pass
    full_out = trainer.policy.forward_sequence(rollout.obs, rollout.h0, rollout.c0)

    # Compare chunked vs full sequence
    assert torch.allclose(tbptt_out.logits, full_out.logits, atol=1e-6)
    assert torch.allclose(tbptt_out.value, full_out.value, atol=1e-6)
