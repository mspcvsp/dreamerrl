import gymnasium as gym
import torch
from gymnasium.spaces import Discrete
import numpy as np
from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.core.actor_critic_update import actor_critic_update
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def test_sanity_training_loop():
    env = gym.make("CartPole-v1")
    assert isinstance(env.action_space, Discrete), "This smoke test requires a discrete action space"
    action_dim: int = int(env.action_space.n)

    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net_world = NetworkConfig(hidden_size=256, action_dim=action_dim, value_bins=41)
    net_actor = NetworkConfig(hidden_size=256, action_dim=action_dim)
    net_critic = NetworkConfig(hidden_size=256, value_bins=41)

    wm = WorldModel(obs_space=env.observation_space, latent=latent, net=net_world)
    actor = Actor(latent=latent, net=net_actor)
    critic = ValueHead(latent=latent, net=net_critic)

    opt_wm = torch.optim.Adam(wm.parameters(), lr=3e-4)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)

    def collect_batch(env, steps=50):
        """
        collect_batch is written as a closure so it can implicitly capture the env and rollout settings from the outer
        scope. This keeps the training loop focused on the Dreamer update itself, while the details of stepping the
        environment stay encapsulated. The closure pattern mirrors how full RL agents separate “how to gather
        experience” from “how to update the model.”
        """
        obs, _ = env.reset()
        obs_list, action_list, reward_list, done_list = [], [], [], []

        for _ in range(steps):
            a = env.action_space.sample()
            next_obs, r, done, trunc, _ = env.step(a)

            obs_list.append(obs)
            action_list.append(a)
            reward_list.append(r)
            done_list.append(done or trunc)

            obs = next_obs
            if done or trunc:
                obs, _ = env.reset()

        batch = {
            "state": torch.from_numpy(np.array(obs_list, dtype=np.float32)).unsqueeze(1),
            "action": torch.nn.functional.one_hot(torch.tensor(action_list), num_classes=env.action_space.n)
            .float()
            .unsqueeze(1),
            "reward": torch.tensor(reward_list, dtype=torch.float32).unsqueeze(1),
            "is_terminal": torch.tensor(done_list, dtype=torch.bool).unsqueeze(1),
        }
        return batch

    for step in range(50):
        batch = collect_batch(env, steps=50)

        opt_wm.zero_grad()
        opt_actor.zero_grad()
        opt_critic.zero_grad()

        actor_loss, critic_loss = actor_critic_update(
            world_model=wm,
            actor=actor,
            critic=critic,
            batch=batch,
            imagination_horizon=5,
            discount=0.99,
            lam=0.95,
        )

        (actor_loss + critic_loss).backward()
        opt_wm.step()
        opt_actor.step()
        opt_critic.step()

        print(f"Step {step:03d} | Actor Loss {actor_loss.item():.3f} | Critic Loss {critic_loss.item():.3f}")
