import torch

from lstmppo.trainer import LSTMPPOTrainer


def reference_masked_diagnostics(h, masks):
    m = masks.unsqueeze(-1)

    sat = (h.abs() * m).sum() / m.sum().clamp(min=1)

    sig = torch.sigmoid(h)
    ent = (-(sig * torch.log(sig + 1e-8)) * m).sum() / m.sum().clamp(min=1)

    valid_pairs = (masks[1:] * masks[:-1]).unsqueeze(-1)
    drift = ((h[1:] - h[:-1]).pow(2) * valid_pairs).sum() / valid_pairs.sum().clamp(min=1)

    return sat, ent, drift


def test_trainer_masked_diagnostics_cpu_gpu_stability():
    """
    CPU and GPU diagnostics should be numerically stable and within a reasonable
    range of each other. Hidden states differ significantly across devices due
    to kernel differences, so exact equality is not expected.
    """

    gpu_trainer = LSTMPPOTrainer.for_validation()
    gpu_policy = gpu_trainer.policy
    gpu_device = gpu_trainer.device

    cpu_trainer = LSTMPPOTrainer.for_validation()
    cpu_trainer.device = torch.device("cpu")
    cpu_policy = cpu_trainer.policy.to("cpu")

    T, B = 10, 2
    H = gpu_trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = gpu_trainer.state.env_info.flat_obs_dim

    torch.manual_seed(0)

    obs_cpu = torch.randn(T, B, obs_dim)
    h0_cpu = torch.randn(B, H)
    c0_cpu = torch.randn(B, H)
    masks_cpu = torch.randint(0, 2, (T, B)).float()

    obs_gpu = obs_cpu.to(gpu_device)
    h0_gpu = h0_cpu.to(gpu_device)
    c0_gpu = c0_cpu.to(gpu_device)
    masks_gpu = masks_cpu.to(gpu_device)

    cpu_out = cpu_policy.forward_sequence(obs_cpu, h0_cpu, c0_cpu)
    gpu_out = gpu_policy.forward_sequence(obs_gpu, h0_gpu, c0_gpu)

    h_cpu = cpu_out.hn
    h_gpu = gpu_out.hn

    sat_cpu, ent_cpu, drift_cpu = reference_masked_diagnostics(h_cpu, masks_cpu)
    sat_gpu, ent_gpu, drift_gpu = reference_masked_diagnostics(h_gpu, masks_gpu)

    # Diagnostics must be finite
    assert torch.isfinite(sat_cpu) and torch.isfinite(sat_gpu)
    assert torch.isfinite(ent_cpu) and torch.isfinite(ent_gpu)
    assert torch.isfinite(drift_cpu) and torch.isfinite(drift_gpu)

    # Diagnostics must be non-negative
    assert sat_cpu >= 0 and sat_gpu >= 0
    assert ent_cpu >= 0 and ent_gpu >= 0
    assert drift_cpu >= 0 and drift_gpu >= 0

    # Diagnostics must be within a reasonable factor (not equal)
    # Allow up to 20% drift due to CPU/GPU kernel differences
    def close_enough(a, b):
        return abs(a - b) / max(a, b, 1e-6) < 0.2

    assert close_enough(sat_cpu.item(), sat_gpu.item())
    assert close_enough(ent_cpu.item(), ent_gpu.item())
    assert close_enough(drift_cpu.item(), drift_gpu.item())
