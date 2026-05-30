# dreamerrl/tests/invariants/test_decoder_symlog_roundtrip.py
import pytest
import torch

from dreamerrl.utils.transforms import symlog


@pytest.mark.invariants
@pytest.mark.decoder_invariants
def test_decoder_symlog_roundtrip(world_model, roundtrip_obs):
    """
    Decoder must predict symlog(obs).
    This test checks:
        obs → symlog → decoder(h,z) ≈ symlog(obs)
    """

    # Target in symlog space
    obs_symlog = symlog(roundtrip_obs)

    B = roundtrip_obs.size(0)
    h = torch.randn(B, world_model.latent.deter_size)
    z = torch.randn(B, world_model.latent.num_classes, world_model.latent.stoch_size)

    # Decoder predicts symlog(obs)
    pred_symlog = world_model.decoder(h, z)

    # Compare directly in symlog space
    assert torch.allclose(pred_symlog, obs_symlog, atol=0.1), "Decoder failed to reconstruct symlog(obs)"
