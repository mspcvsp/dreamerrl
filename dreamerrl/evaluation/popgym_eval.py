import torch
from dreamerrl.models.world_model import WorldModelState
from dreamerrl.utils.twohot import value_from_logits


@torch.no_grad()
def evaluate_popgym(env, world, actor, episodes=10, device="cpu"):
    """
    Deterministic Dreamer-V3 evaluation on PopGym environments.
    Uses latent state + actor policy (no exploration noise).
    """
    returns = []
    batch_size = env.batch_size

    for _ in range(episodes):
        obs = env.reset()["state"]
        world_state = world.init_state(batch_size).to(device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        ep_return = torch.zeros(batch_size, device=device)

        while not torch.all(done):
            # Update latent state
            out = world.observe_step(world_state, obs)
            world_state = out["post"]

            # Greedy action from actor
            logits = actor(world_state.h, world_state.z)
            action = torch.argmax(logits, dim=-1)

            # Step environment
            env_out = env.step(action)
            obs = env_out["state"]
            reward = env_out["reward"]
            done = env_out["is_terminal"]

            ep_return += reward * (~done)

        returns.append(ep_return.cpu())

    returns = torch.stack(returns, dim=0)  # (episodes, batch)
    return {
        "mean": returns.mean().item(),
        "std": returns.std().item(),
        "per_env": returns.mean(dim=0).tolist(),
    }
