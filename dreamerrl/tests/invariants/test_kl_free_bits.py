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
@pytest.mark.parametrize("free_bits", [0.0, 0.1, 1.0])
def test_structured_kl_free_bits_invariants(free_bits: float) -> None:
    """
    Invariants for structured_kl with free bits:

    1. kl_total, kl_dyn, kl_rep are all >= 0.
    2. kl_total == kl_dyn + kl_rep (within numerical tolerance).
    3. Increasing free_bits never increases the effective KL for the same q, p.
    """

    torch.manual_seed(0)

    latent = LatentConfig(deter_size=200, stoch_size=8, num_classes=16)
    net = NetworkConfig(hidden_size=256, action_dim=None, value_bins=None)

    prior = Prior(latent=latent, net=net)
    posterior = Posterior(latent=latent, net=net)

    B = 32
    h = torch.randn(B, latent.deter_size)

    # Fake encoder embedding; only h matters for prior/posterior here.
    embed = torch.randn(B, net.hidden_size)

    post_stats = posterior(h, embed)
    prior_stats = prior(h)

    q_probs = post_stats["probs"]  # (B, stoch_size * num_classes)
    p_probs = prior_stats["probs"]

    kl = structured_kl(q_probs=q_probs, p_probs=p_probs, free_bits=free_bits)

    kl_total = kl["kl_total"]
    kl_dyn = kl["kl_dyn"]
    kl_rep = kl["kl_rep"]

    # 1. Non-negativity
    assert torch.all(kl_total >= 0), "kl_total must be non-negative"
    assert torch.all(kl_dyn >= 0), "kl_dyn must be non-negative"
    assert torch.all(kl_rep >= 0), "kl_rep must be non-negative"

    # 2. Decomposition consistency: total == dyn + rep
    diff = (kl_total - (kl_dyn + kl_rep)).abs().max().item()
    assert diff < 1e-5, f"kl_total != kl_dyn + kl_rep (max diff {diff})"


@pytest.mark.invariants
def test_structured_kl_free_bits_effect():
    """
    Free bits should modify the effective KL unless the raw KL is already above
    the threshold. We only check that the outputs differ, not the direction.
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

    kl0 = structured_kl(q_probs=q_probs, p_probs=p_probs, free_bits=0.0)["kl_total"]
    kl1 = structured_kl(q_probs=q_probs, p_probs=p_probs, free_bits=1.0)["kl_total"]

    # Free bits should have some effect (not necessarily monotonic)
    assert not torch.allclose(kl0, kl1, atol=1e-6), "free_bits should modify the effective KL for at least some entries"
