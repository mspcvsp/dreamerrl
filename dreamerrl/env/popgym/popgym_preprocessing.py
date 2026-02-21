import numpy as np
import torch
import gymnasium as gym


def get_flat_obs_dim(space: gym.Space) -> int:
    """
    Compute flattened observation dimension for any PopGym observation space.
    PopGym uses Box, Tuple, or Dict spaces.
    """
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))

    elif isinstance(space, gym.spaces.Dict):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces.values())

    elif isinstance(space, gym.spaces.Tuple):
        return sum(get_flat_obs_dim(sub) for sub in space.spaces)

    else:
        raise NotImplementedError(f"Unsupported observation space: {space}")


def flatten_obs(obs, space: gym.Space) -> np.ndarray:
    """
    Flatten a PopGym observation into a (B, flat_dim) numpy array.
    PopGym vectorized envs return obs with shape (B, ...).
    """
    if isinstance(space, gym.spaces.Box):
        return obs.reshape(obs.shape[0], -1)

    elif isinstance(space, gym.spaces.Dict):
        parts = []
        for key, subspace in space.spaces.items():
            parts.append(flatten_obs(obs[key], subspace))
        return np.concatenate(parts, axis=-1)

    elif isinstance(space, gym.spaces.Tuple):
        parts = []
        for i, subspace in enumerate(space.spaces):
            parts.append(flatten_obs(obs[i], subspace))
        return np.concatenate(parts, axis=-1)

    else:
        raise NotImplementedError(f"Unsupported observation type: {type(obs)}")


def to_tensor(obs, device):
    """
    Convert flattened numpy obs → torch tensor.
    """
    return torch.tensor(obs, dtype=torch.float32, device=device)
