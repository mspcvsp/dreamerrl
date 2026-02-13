import torch


def test_diagnostics_mean_aggregation_gpu(deterministic_trainer, fake_batch):
    device = torch.device("cuda")
    trainer = deterministic_trainer(device=device)

    batch = fake_batch(device=device, batch_size=16, seq_len=32)

    diagnostics = trainer.policy.compute_diagnostics(batch.obs, batch.h0, batch.c0)

    # Ensure diagnostics are scalars (mean over hidden units)
    for name, value in diagnostics.items():
        assert value.dim() == 0, f"{name} must be scalar after mean aggregation"

    # No NaNs
    assert all(torch.isfinite(v) for v in diagnostics.values())
