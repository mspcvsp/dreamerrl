from .replay_buffer import DreamerReplayBuffer

# Alias for compatibility with tests expecting ReplayBuffer
ReplayBuffer = DreamerReplayBuffer

__all__ = ["DreamerReplayBuffer", "ReplayBuffer"]
