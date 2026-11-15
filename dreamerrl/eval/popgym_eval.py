"""Whole-episode PopGym evaluation and a small training harness.

Returns use the environment's native reward scale. Cross-seed loss CV and policy
KL on unrelated states are intentionally not treated as learning evidence.
"""

from typing import TypedDict

import numpy as np
import torch
from numpy.typing import NDArray


class PopGymTrainingResult(TypedDict):
    wm_loss: NDArray[np.float64]
    actor_loss: NDArray[np.float64]
    critic_loss: NDArray[np.float64]
    returns: NDArray[np.float64]
    env_steps: int


@torch.no_grad()
def evaluate_popgym(env, world, actor, episodes=10, device=None, seed=10000, task_horizon=None):
    """Evaluate episodes per vector row, latching each row's first episode end."""
    if task_horizon is not None and task_horizon < 1:
        raise ValueError("task_horizon must be positive")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    model_device = next(world.parameters()).device
    if device is not None and torch.device(device) != model_device:
        raise ValueError("Evaluation device must match the world model")
    action_dim = world.net_cfg.action_dim
    if action_dim is None:
        raise ValueError("Evaluation requires a discrete action dimension")
    devices = [model_device.index or 0] if model_device.type == "cuda" else []
    world_training, actor_training = world.training, actor.training
    posterior_mode = world.posterior.deterministic_latent_for_tests
    returns = []
    try:
        world.eval()
        actor.eval()
        world.posterior.deterministic_latent_for_tests = True
        with torch.random.fork_rng(devices=devices):
            for episode in range(episodes):
                obs = env.reset(seed=seed + episode * env.batch_size)
                state = world.init_state(env.batch_size)
                action = torch.zeros(env.batch_size, action_dim, device=model_device)
                done = torch.zeros(env.batch_size, dtype=torch.bool, device=model_device)
                total = torch.zeros(env.batch_size, device=model_device)
                elapsed = torch.zeros(env.batch_size, device=model_device)
                while not done.all():
                    observation = obs["state"].to(model_device)
                    if task_horizon is not None:
                        clock = (1 - elapsed / task_horizon).clamp_min(0).unsqueeze(-1)
                        observation = torch.cat([observation, clock], -1)
                    state = world.observe_step(state, observation, action, is_first=obs["is_first"].to(model_device))[
                        "post"
                    ]
                    ids = actor(state.h, state.z).argmax(-1)
                    obs = env.step(ids)
                    elapsed.add_(1)
                    elapsed[obs["is_first"].to(model_device)] = 0
                    total += obs["reward"].to(model_device) * (~done)
                    done |= obs["is_last"].to(model_device)
                    if task_horizon is not None:
                        done |= elapsed >= task_horizon
                    action = torch.nn.functional.one_hot(ids, action_dim).float()
                returns.append(total.cpu())
    finally:
        world.posterior.deterministic_latent_for_tests = posterior_mode
        world.train(world_training)
        actor.train(actor_training)
    values = torch.stack(returns)
    return dict(
        mean=values.mean().item(),
        std=values.std(unbiased=False).item(),
        per_env=values.mean(0).tolist(),
        episode_returns=values.tolist(),
    )


def train_popgym_seed(trainer, steps: int = 1000) -> PopGymTrainingResult:
    """Train without changing the caller's configuration; record full episodes."""
    if steps < 1:
        raise ValueError("steps must be positive")
    losses = []
    returns = []
    running = torch.zeros(trainer.env.batch_size, device=trainer.device)
    original_step = trainer.env.step

    def recording_step(actions):
        out = original_step(actions)
        running.add_(out["reward"].to(trainer.device))
        for i in out["is_last"].nonzero(as_tuple=True)[0].tolist():
            returns.append(running[i].item())
            running[i] = 0
        return out

    trainer.env.step = recording_step
    try:
        # At most one configured episode is needed to populate replay on these
        # fixed-length memory tasks. Do not silently spin on ineligible episodes.
        for _ in range(
            max(
                1,
                (trainer.cfg.env.max_episode_steps + trainer.cfg.train.collect_steps - 1)
                // trainer.cfg.train.collect_steps,
            )
        ):
            trainer.collect_env_steps()
        for update in range(steps):
            trainer.collect_env_steps()
            batch = trainer.replay.sample(trainer.cfg.train.batch_size)
            metrics = trainer.update_world_model(batch, update)
            a, c = trainer.update_actor_critic(batch, update)
            row = [metrics.total_loss.item(), a, c]
            if not np.isfinite(row).all():
                raise ValueError("Non-finite training loss")
            losses.append(row)
            if update % 100 == 0:
                print(f"[popgym] update={update}/{steps} env_steps={trainer.total_env_steps}", flush=True)
    finally:
        trainer.env.step = original_step
    values = np.asarray(losses, dtype=np.float64)
    return {
        "wm_loss": values[:, 0],
        "actor_loss": values[:, 1],
        "critic_loss": values[:, 2],
        "returns": np.asarray(returns, dtype=np.float64),
        "env_steps": trainer.total_env_steps,
    }


def aggregate_popgym_results(results: list[PopGymTrainingResult]) -> dict[str, float]:
    """Descriptive training metrics; held-out evaluation is reported separately."""
    if not results:
        raise ValueError("At least one seed is required")
    return dict(
        wm_loss_mean=float(np.mean([r["wm_loss"].mean() for r in results])),
        actor_loss_mean=float(np.mean([r["actor_loss"].mean() for r in results])),
        critic_loss_mean=float(np.mean([r["critic_loss"].mean() for r in results])),
        mean_return=float(np.mean([r["returns"].mean() for r in results])),
    )
