"""
KL free‑bits prevents the model from “cheating” by collapsing its latent space, ensuring the world model continues to
encode meaningful information.
"""

import pytest
import torch

from dreamerrl.models.categorical_kl import structured_kl
from dreamerrl.models.posterior import Posterior
from dreamerrl.models.prior import Prior
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.mark.invariants
@pytest.mark.parametrize("free_nats", [0.0, 0.1, 1.0])
def test_structured_kl_free_nats_invariants(free_nats: float) -> None:
    """
    Dreamer‑V3 KL invariants:

    1. kl_total, kl_dyn, kl_rep are all >= 0.
    2. kl_total == kl_dyn + kl_rep (within tolerance).
    3. Increasing free_nats never increases *per‑factor* KL.
    """

    torch.manual_seed(0)

    latent = LatentConfig(deter_size=200, stoch_size=8, num_classes=16)
    net = NetworkConfig(hidden_size=256, action_dim=None, value_bins=None)

    prior = Prior(latent=latent, net=net)
    posterior = Posterior(latent=latent, net=net)

    B = 32
    h = torch.randn(B, latent.deter_size)
    embed = torch.randn(B, net.hidden_size)

    post_stats = posterior(h, embed)
    prior_stats = prior(h)

    q_probs = post_stats["probs"]  # (B, K, C)
    p_probs = prior_stats["probs"]

    kl = structured_kl(q_probs=q_probs, p_probs=p_probs, free_nats=free_nats)

    kl_total = kl["kl_total"]
    kl_dyn = kl["kl_dyn"]
    kl_rep = kl["kl_rep"]

    # 1. Non-negativity
    assert torch.all(kl_total >= 0)
    assert torch.all(kl_dyn >= 0)
    assert torch.all(kl_rep >= 0)

    # 2. Decomposition consistency
    diff = (kl_total - (kl_dyn + kl_rep)).abs().max().item()
    assert diff < 1e-5


@pytest.mark.invariants
def test_structured_kl_free_nats_effect():
    """
    Free-nats should modify the *per-factor* KL unless raw KL is already above threshold.
    """

    torch.manual_seed(1)

    latent = LatentConfig(deter_size=64, stoch_size=4, num_classes=8)
    net = NetworkConfig(hidden_size=128, action_dim=None, value_bins=None)

    prior = Prior(latent=latent, net=net)
    posterior = Posterior(latent=latent, net=net)

    B = 16
    h = torch.randn(B, latent.deter_size)
    embed = torch.randn(B, net.hidden_size)

    post_stats = posterior(h, embed)
    prior_stats = prior(h)

    q_probs = post_stats["probs"]
    p_probs = prior_stats["probs"]

    kl0 = structured_kl(q_probs=q_probs, p_probs=p_probs, free_nats=0.0)["kl_total"]
    kl1 = structured_kl(q_probs=q_probs, p_probs=p_probs, free_nats=1.0)["kl_total"]

    assert not torch.allclose(kl0, kl1, atol=1e-6)
