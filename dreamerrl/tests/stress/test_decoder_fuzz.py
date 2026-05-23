import pytest
import torch


@pytest.mark.stress
def test_decoder_fuzz(world_model):
    """
    Fuzz decoder with extreme latent noise.
    """
    torch.manual_seed(0)

    B = 32
    for _ in range(50):
        h = torch.randn(B, world_model.latent.deter_size) * 10
        z = torch.randn(B, world_model.latent.stoch_size, world_model.latent.num_classes) * 10

        recon = world_model.decoder(h, z)
        assert torch.isfinite(recon).all()
