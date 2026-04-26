import torch

from dreamerrl.models.decoder import ObsDecoder


def test_decoder_cpu_gpu_equivalence():
    if not torch.cuda.is_available():
        return

    B, deter_size, stoch_size, nclasses, hidden_size, obs_dim = 4, 32, 16, 32, 64, 8

    # Initialize on CPU for deterministic weights
    torch.manual_seed(0)
    decoder_cpu = ObsDecoder(
        deter_size=deter_size,
        stoch_size=stoch_size,
        num_classes=nclasses,
        hidden_size=hidden_size,
        obs_shape=obs_dim,
    ).cpu()

    # Create GPU copy with identical weights
    decoder_gpu = ObsDecoder(
        deter_size=deter_size,
        stoch_size=stoch_size,
        num_classes=nclasses,
        hidden_size=hidden_size,
        obs_shape=obs_dim,
    ).cuda()

    decoder_gpu.load_state_dict(decoder_cpu.state_dict())

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    out_cpu = decoder_cpu(h, z)
    out_gpu = decoder_gpu(h.cuda(), z.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu)
