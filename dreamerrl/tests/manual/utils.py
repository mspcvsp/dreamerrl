"""Small configurations and held-out evaluation for the MiniHack experiments."""

from __future__ import annotations

import gymnasium as gym
import torch
from gymnasium.wrappers import TimeLimit

from dreamerrl.env.minihack.minihack_utils import extract_glyphs
from dreamerrl.models.aux_objectives import AUX_OBJECTIVES
from dreamerrl.utils.types import make_default_config


def minihack_config(seed, logdir, novelty=False):
    cfg = make_default_config()
    cfg.env.env_id = "MiniHack-Room-5x5-v0"
    cfg.env.num_envs = 2
    cfg.env.max_episode_steps = 6
    cfg.env.parallel = False
    cfg.env.seed = seed
    cfg.world.disable_aux_losses = not novelty
    cfg.world.aux_objectives = [AUX_OBJECTIVES["novelty"]] if novelty else []
    cfg.world.deter_size = 16
    cfg.world.num_classes = 3
    cfg.world.stoch_size = 4
    cfg.world.hidden_size = 16
    cfg.world.imagination_horizon = 3
    cfg.ac.actor_hidden = cfg.ac.critic_hidden = 16
    cfg.train.seed = seed
    cfg.train.cuda = False
    cfg.train.use_amp = False
    cfg.train.enable_fused_adamw_kernels = False
    cfg.train.deterministic_env = False
    cfg.train.deterministic_imagination = False
    cfg.train.freeze_actor_critic_steps = 0
    cfg.train.random_exploration_steps = 0
    cfg.train.batch_size = 2
    cfg.train.collect_steps = 12
    cfg.train.seq_len = 4
    cfg.log.enable_wandb = False
    cfg.log.tb_logdir = str(logdir)
    cfg.log.run_name = f"seed{seed}_{'novelty' if novelty else 'baseline'}"
    return cfg


@torch.no_grad()
def evaluate_minihack(cfg, world, actor, episode_seeds, random_policy=False):
    """Whole episode returns on held-out seeds, including the final reward.

    Use a separate, single environment to avoid vector autoreset ambiguities.
    Evaluation restores training modes and torch RNG state. Latent modes and
    greedy actions give a repeatable policy evaluation; random actions use the
    independently seeded environment action space.
    """
    device = next(world.parameters()).device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    env = TimeLimit(gym.make(cfg.env.env_id), max_episode_steps=cfg.env.max_episode_steps)
    world_training, actor_training = world.training, actor.training
    deterministic_posterior = world.posterior.deterministic_latent_for_tests
    returns = []
    try:
        world.eval()
        actor.eval()
        world.posterior.deterministic_latent_for_tests = True
        with torch.random.fork_rng(devices=devices):
            for seed in episode_seeds:
                obs, _ = env.reset(seed=seed)
                env.action_space.seed(seed)
                state = world.init_state(1)
                action_dim = world.net_cfg.action_dim
                if action_dim is None:
                    raise ValueError("Evaluation requires a discrete action dimension")
                action = torch.zeros(1, action_dim, device=device)
                total = 0.0
                for t in range(cfg.env.max_episode_steps):
                    if random_policy:
                        action_id = env.action_space.sample()
                    else:
                        observation = torch.as_tensor(
                            extract_glyphs(obs).reshape(1, -1), dtype=torch.float32, device=device
                        )
                        if cfg.env.finite_horizon:
                            observation = torch.cat(
                                [
                                    observation,
                                    torch.tensor([[1 - t / cfg.env.max_episode_steps]], device=device),
                                ],
                                -1,
                            )
                        state = world.observe_step(
                            state,
                            observation,
                            action,
                            is_first=torch.tensor([t == 0], device=device),
                        )["post"]
                        action_id = int(actor(state.h, state.z).argmax(-1).item())
                    obs, reward, terminated, truncated, _ = env.step(action_id)
                    total += float(reward)
                    if terminated or truncated:
                        break
                    action = torch.nn.functional.one_hot(
                        torch.tensor([action_id], device=device),
                        action_dim,
                    ).float()
                returns.append(total)
    finally:
        env.close()
        world.posterior.deterministic_latent_for_tests = deterministic_posterior
        world.train(world_training)
        actor.train(actor_training)
    return returns
