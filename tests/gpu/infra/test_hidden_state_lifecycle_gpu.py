import torch


def test_hidden_state_lifecycle_gpu(synthetic_trainer, fake_rollout):
    device = torch.device("cuda")
    trainer = synthetic_trainer
    trainer.policy.to(device)

    rollout = fake_rollout(
        device=device,
        batch_size=4,
        seq_len=10,
        force_done_at=5,
    )

    done_tb = rollout.done.transpose(0, 1)  # (T, B)
    out = trainer.policy.forward_sequence(rollout.obs, rollout.h0, rollout.c0, done=done_tb)

    # out.hn: (T, B, H)
    t_done = 5
    t_reset = t_done + 1

    # 1) Hidden state just before done should be non-zero
    assert not torch.allclose(out.hn[t_done - 1], torch.zeros_like(out.hn[t_done - 1]))

    # 2) After reset, hidden state should match a fresh unroll from zero at that timestep
    with torch.no_grad():
        h0 = torch.zeros_like(rollout.h0)
        c0 = torch.zeros_like(rollout.c0)
        fresh = trainer.policy.forward_sequence(
            rollout.obs[t_reset : t_reset + 1],  # single step
            h0,
            c0,
        )
    assert torch.allclose(out.hn[t_reset], fresh.hn[0], atol=1e-6)
