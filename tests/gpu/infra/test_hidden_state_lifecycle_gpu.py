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
    # Reset happens at the *first step after* done, i.e. t = 6
    t_reset = 6
    assert torch.allclose(out.hn[t_reset], torch.zeros_like(out.hn[t_reset]))
    assert torch.allclose(out.cn[t_reset], torch.zeros_like(out.cn[t_reset]))
