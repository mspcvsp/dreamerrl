import numpy as np
import torch
from gymnasium.spaces import Box

from dreamerrl.models.obs_encoder import build_obs_encoder


def test_encoder_cpu_gpu_equivalence():
    if not torch.cuda.is_available():
        return

    B, obs_dim = 4, 8
    obs_space = Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype=np.float32)

    # Initialize on CPU for deterministic weights
    torch.manual_seed(0)
    encoder_cpu = build_obs_encoder(obs_space, embed_dim=64).cpu()

    # Create GPU copy with identical weights
    encoder_gpu = build_obs_encoder(obs_space, embed_dim=64).cuda()
    encoder_gpu.load_state_dict(encoder_cpu.state_dict())

    obs = torch.randn(B, obs_dim)

    out_cpu = encoder_cpu(obs)
    out_gpu = encoder_gpu(obs.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu)
