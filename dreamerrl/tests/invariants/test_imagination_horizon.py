import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.world_model import WorldModel, WorldModelState
from dreamerrl.tests.conftest import DummyActor
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def _build_world_model(device: torch.device) -> WorldModel:
    obs_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(8,),
        dtype=np.float32,
    )
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5, value_bins=41)
    return WorldModel(obs_space=obs_space, latent=latent, net=net, device=device)


@pytest.mark.invariants
@pytest.mark.imagination_invariants
def test_imagine_step_horizon_consistency() -> None:
    """
    Unrolling imagine_step H times must produce exactly H states, with consistent
    shapes and device, and match a manual single-step unroll.
    """

    torch.manual_seed(0)

    device = torch.device("cpu")
    wm = _build_world_model(device=device)

    latent = wm.latent
    net = wm.net_cfg
    assert net.action_dim is not None

    actor = DummyActor(latent=latent, net=net).to(device)

    B = 4
    H = 5

    # Initial state
    state0 = wm.init_state(batch_size=B)
    assert isinstance(state0, WorldModelState)

    # Manual unroll using imagine_step
    states = [state0]
    cur = state0
    for _ in range(H):
        cur = wm.imagine_step(cur, actor=actor, deterministic_imagination=True)
        states.append(cur)

    # We expect H transitions → H next states (excluding initial)
    assert len(states) == H + 1
    for s in states:
        assert s.h.shape == (B, latent.deter_size)
        assert s.z.shape == (B, latent.stoch_size, latent.num_classes)
        assert s.h.device == device
        assert s.z.device == device

    # Deterministic unroll: compare last state with recomputed chain
    cur2 = state0
    for _ in range(H):
        cur2 = wm.imagine_step(cur2, actor=actor, deterministic_imagination=True)

    assert torch.allclose(states[-1].h, cur2.h, atol=1e-6)
    assert torch.allclose(states[-1].z, cur2.z, atol=1e-6)


@pytest.mark.invariants
@pytest.mark.imagination_invariants
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_imagine_step_cpu_gpu_determinism_horizon() -> None:
    """
    For a fixed initial state and actor parameters, unrolling imagine_step for H
    steps must be deterministic across CPU and GPU.
    """

    torch.manual_seed(1)

    wm_cpu = _build_world_model(device=torch.device("cpu"))
    wm_gpu = _build_world_model(device=torch.device("cuda"))

    # Sync world model parameters
    wm_gpu.load_state_dict(wm_cpu.state_dict())

    latent = wm_cpu.latent
    net = wm_cpu.net_cfg
    assert net.action_dim is not None

    actor_cpu = DummyActor(latent=latent, net=net).to("cpu")
    actor_gpu = DummyActor(latent=latent, net=net).to("cuda")
    actor_gpu.load_state_dict(actor_cpu.state_dict())

    B = 3
    H = 4

    state0_cpu = wm_cpu.init_state(batch_size=B)
    state0_gpu = WorldModelState(
        h=state0_cpu.h.to("cuda"),
        z=state0_cpu.z.to("cuda"),
        prior_stats=None,
        post_stats=None,
    )

    cur_cpu = state0_cpu
    cur_gpu = state0_gpu

    with torch.no_grad():
        for _ in range(H):
            cur_cpu = wm_cpu.imagine_step(cur_cpu, actor=actor_cpu, deterministic_imagination=True)
            cur_gpu = wm_gpu.imagine_step(cur_gpu, actor=actor_gpu, deterministic_imagination=True)

    h_cpu = cur_cpu.h
    z_cpu = cur_cpu.z
    h_gpu = cur_gpu.h.to("cpu")
    z_gpu = cur_gpu.z.to("cpu")

    assert h_cpu.shape == h_gpu.shape
    assert z_cpu.shape == z_gpu.shape

    max_diff_h = (h_cpu - h_gpu).abs().max().item()
    max_diff_z = (z_cpu - z_gpu).abs().max().item()

    assert max_diff_h < 1e-5, f"h mismatch across devices: {max_diff_h}"
    assert max_diff_z < 1e-5, f"z mismatch across devices: {max_diff_z}"


@pytest.mark.invariants
@pytest.mark.imagination_invariants
def test_stochastic_imagination_distributional_invariants() -> None:
    torch.manual_seed(0)

    wm_cpu = _build_world_model(device=torch.device("cpu"))
    wm_gpu = _build_world_model(device=torch.device("cuda"))

    wm_gpu.load_state_dict(wm_cpu.state_dict())

    latent = wm_cpu.latent
    net = wm_cpu.net_cfg
    assert net.action_dim is not None

    actor_cpu = DummyActor(latent=latent, net=net).to("cpu")
    actor_gpu = DummyActor(latent=latent, net=net).to("cuda")
    actor_gpu.load_state_dict(actor_cpu.state_dict())

    B = 64
    H = 10

    # Collect samples
    samples_cpu = []
    samples_gpu = []

    for _ in range(200):  # enough for stable statistics
        s0_cpu = wm_cpu.init_state(batch_size=B)
        s0_gpu = s0_cpu.to(torch.device("cuda"))

        cur_cpu = s0_cpu
        cur_gpu = s0_gpu

        for _ in range(H):
            cur_cpu = wm_cpu.imagine_step(cur_cpu, actor=actor_cpu, deterministic_imagination=False)
            cur_gpu = wm_gpu.imagine_step(cur_gpu, actor=actor_gpu, deterministic_imagination=False)

        samples_cpu.append(cur_cpu.h.detach().cpu())
        samples_gpu.append(cur_gpu.h.detach().cpu())

    cpu_stack = torch.stack(samples_cpu)  # (N, B, deter)
    gpu_stack = torch.stack(samples_gpu)

    # 1. Shapes match
    assert cpu_stack.shape == gpu_stack.shape

    # 2. Finiteness
    assert torch.isfinite(cpu_stack).all()
    assert torch.isfinite(gpu_stack).all()

    # 3. Distributional similarity (mean/var)
    mean_diff = (cpu_stack.mean() - gpu_stack.mean()).abs().item()
    var_diff = (cpu_stack.var() - gpu_stack.var()).abs().item()

    assert mean_diff < 1e-2
    assert var_diff < 1e-2

    # 4. Non-degeneracy: stochastic rollouts differ
    assert not torch.allclose(cpu_stack[0], cpu_stack[1])
