import torch

from dreamerrl.models.posterior import Posterior
from dreamerrl.models.prior import Prior
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def test_prior_posterior_shapes():
    B = 8
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=200)

    prior = Prior(latent=latent, net=net)
    posterior = Posterior(latent=latent, net=net)

    h = torch.zeros(B, latent.deter_size)

    # Determine embed size dynamically from the Posterior module
    embed_dim = posterior.fc1.in_features - latent.deter_size
    embed = torch.zeros(B, embed_dim)

    prior_stats = prior(h)
    post_stats = posterior(h, embed)

    assert prior_stats["logits"].shape == (B, latent.stoch_size, latent.num_classes)
    assert post_stats["logits"].shape == (B, latent.stoch_size, latent.num_classes)
    assert prior_stats["z"].shape == (B, latent.z_dim)
    assert post_stats["z"].shape == (B, latent.z_dim)
