import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.tests.conftest import DummyObsSpace
from dreamerrl.training.core.imagination import imagine_trajectory_for_training
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def test_imagination_determinism():
    torch.manual_seed(0)

    # Minimal configs
    latent = LatentConfig(deter_size=32, stoch_size=8, num_classes=4)
    net = NetworkConfig(hidden_size=64, action_dim=5, value_bins=51)

    # World model, actor, critic
    obs_space = DummyObsSpace(8)
    world = WorldModel(obs_space=obs_space, latent=latent, net=net, free_nats=1.0, device=torch.device("cpu"))
    actor = Actor(latent=latent, net=net)
    critic = ValueHead(latent=latent, net=net)

    # Initial state
    s0 = world.init_state(batch_size=3)

    traj1 = imagine_trajectory_for_training(
        world_model=world,
        actor=actor,
        critic=critic,
        state=s0,
        horizon=5,
        deterministic_imagination=True,
    )

    traj2 = imagine_trajectory_for_training(
        world_model=world,
        actor=actor,
        critic=critic,
        state=s0,
        horizon=5,
        deterministic_imagination=True,
    )

    for key in ["h", "z", "reward", "action"]:
        assert torch.allclose(traj1[key], traj2[key]), f"Nondeterministic imagination in {key}"
