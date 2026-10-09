from dataclasses import replace

import torch

from dreamerrl.models.categorical_kl import structured_kl
from dreamerrl.utils.types import KLConfig


def test_structured_kl_behavior():
    """
    KL must:
      • be finite
      • be in a reasonable Dreamer-V3 range
      • NOT scale linearly with stoch_size
    """
    B = 4
    C = 3  # num_classes

    # Two different stoch sizes
    stoch_sizes = [16, 32]

    kl_values = []

    for K in stoch_sizes:
        # Fake logits
        q_logits = torch.randn(B, K, C)
        p_logits = torch.randn(B, K, C)

        # Convert to probs
        q_probs = q_logits.softmax(dim=-1)
        p_probs = p_logits.softmax(dim=-1)

        kl_cfg = replace(
            KLConfig(),
            free_nats=1.0,
        )

        out = structured_kl(
            q_probs,
            p_probs,
            kl_cfg=kl_cfg,
        )

        kl = out["kl_total"].mean().item()
        kl_values.append(kl)

        # Basic sanity
        assert torch.isfinite(torch.tensor(kl)), "KL must be finite"
        assert 0.01 < kl < 20.0, f"KL magnitude unreasonable: {kl}"

    kl_small, kl_large = kl_values

    # 🔥 The key invariant:
    # KL MUST NOT double when stoch_size doubles.
    ratio = kl_large / kl_small
    assert ratio < 1.5, f"KL scales with stoch_size (ratio={ratio}) — reduction bug"
