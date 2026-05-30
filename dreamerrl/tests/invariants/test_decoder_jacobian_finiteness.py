import pytest
import torch


@pytest.mark.invariants
@pytest.mark.decoder_invariants
def test_decoder_jacobian_finiteness(world_model):
    """
    Decoder Jacobian must be finite for stability.
    """
    B = 4
    h = torch.randn(B, world_model.latent.deter_size, requires_grad=True)
    z = torch.randn(B, world_model.latent.num_classes, world_model.latent.stoch_size, requires_grad=True)

    recon = world_model.decoder(h, z).sum()
    recon.backward()

    assert h.grad is not None
    assert z.grad is not None

    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(z.grad).all()
