from typing import cast

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from dreamerrl.models.world_model import WorldModel
from dreamerrl.models.world_model_core import RSSMCore
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rssm_core_cpu_gpu_determinism():
    """
    RSSMCore should produce numerically identical outputs on CPU and GPU when
    given identical parameters and inputs. This test verifies that determinism.
    """

    torch.manual_seed(0)

    # Pylance requires explicit dtype from numpy, not Python float
    obs_space = Box(
        low=np.float32(0.0),
        high=np.float32(1.0),
        shape=(8,),
        dtype=np.float32,
    )

    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=3, value_bins=41)

    wm_cpu = WorldModel(obs_space=obs_space, latent=latent, net=net, device=torch.device("cpu"))
    wm_gpu = WorldModel(obs_space=obs_space, latent=latent, net=net, device=torch.device("cuda"))

    # Copy parameters exactly
    wm_gpu.load_state_dict(wm_cpu.state_dict())

    B = 4

    # Explicit dtype + device for Pylance
    h_cpu = torch.randn(B, latent.deter_size, dtype=torch.float32, device="cpu")

    # Pylance requires explicit int for high and num_classes
    assert net.action_dim is not None, "action_dim must be specified in net config for reproducibility check"

    action_indices = torch.randint(
        low=0,
        high=int(net.action_dim),
        size=(B,),
        dtype=torch.int64,
        device="cpu",
    )

    a_cpu = torch.nn.functional.one_hot(
        action_indices,
        num_classes=int(net.action_dim),
    ).float()

    # Move to GPU
    h_gpu = h_cpu.to("cuda")
    a_gpu = a_cpu.to("cuda")

    # RSSMCore is exposed as wm.core
    core_cpu = cast(RSSMCore, wm_cpu.rssm)
    core_gpu = cast(RSSMCore, wm_gpu.rssm)

    with torch.no_grad():
        h_next_cpu = core_cpu(h_cpu, a_cpu)
        h_next_gpu = core_gpu(h_gpu, a_gpu).to("cpu")

    assert h_next_cpu.shape == h_next_gpu.shape

    max_diff = (h_next_cpu - h_next_gpu).abs().max().item()
    assert max_diff < 1e-5, f"CPU/GPU mismatch: max diff {max_diff}"
