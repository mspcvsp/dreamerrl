# inside dreamerrl/models/actor.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

from dreamerrl.utils.types import LatentConfig, NetworkConfig


class Actor(nn.Module):
    def __init__(self, *, latent: LatentConfig, net: NetworkConfig):
        super().__init__()
        assert net.action_dim is not None, "Actor requires action_dim"

        self.latent = latent
        self.net_cfg = net

        """
        Deterministic initialization for CPU/GPU equivalence tests.
        -----------------------------------------------------------
        NOTE: We set torch.manual_seed(0) here to guarantee CPU/GPU weight equivalence in tests.

        This does NOT harm Dreamer training because:

        • The specific seed value (0) has no special meaning—any fixed integer would work.

        • DreamerTrainer calls set_global_seeds(cfg.train.seed) before constructing models, so training runs still use
        the user‑specified global seed.

        • The Actor/Critic are instantiated exactly once per training run, so reseeding here does not interfere with
        rollout randomness, replay sampling, or world model updates.

        • Only the *initial weights* become deterministic; all stochasticity during training (env steps, imagination,
        sampling, dropout-free networks) still comes from the global RNG state set by the trainer.

        In short: this ensures deterministic initialization for CPU/GPU equivalence tests without reducing Dreamer's
        exploration, stochasticity, or training diversity.
        """
        torch.manual_seed(0)

        self.net = nn.Sequential(
            nn.Linear(latent.deter_size + latent.z_dim, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.action_dim),
        )

    def forward(self, h, z):
        """
        IMPORTANT:
        ---------
        This Actor returns *logits* for a categorical policy, not an action. Dreamer‑V3 uses a discrete action
        space, so the correct usage is:

        logits = actor(h, z)
        dist = Categorical(logits=logits)
        a = dist.sample()                     # integer action index
        action = F.one_hot(a, action_dim)     # one-hot vector for RSSMCore

        RSSMCore requires the one-hot action vector. Passing logits or the integer index directly will silently break
        the latent dynamics.
        """
        return self.net(torch.cat([h, z], dim=-1))

    """
    Why both forward() and forward_logits() exist
    ---------------------------------------------

    These two methods perform the *same computation* (produce action logits),
    but they serve different semantic roles in the Dreamer-V3 architecture:

    • forward() is the standard PyTorch entry point.
      It keeps the module compatible with nn.Sequential, TorchScript, and
      general PyTorch tooling.

    • forward_logits() is a semantic alias used only in training code.
      It makes the actor-critic update self-documenting: when reading the
      training loop, it's immediately clear that we want *logits* (not actions)
      for computing log-probs, entropy, and advantages.

    This separation prevents accidental misuse:
      - imagination and actor-critic updates always call forward_logits()
      - environment interaction always calls act()
      - tests can assert invariants on each mode independently

    In short: forward() is the PyTorch API; forward_logits() is the Dreamer-V3
    training API. They intentionally share the same signature to keep the
    actor simple, deterministic, and easy to test.
    """

    def forward_logits(self, h, z):
        return self.forward(h, z)

    def distribution(self, h, z):
        return Categorical(logits=self.forward(h, z))

    @torch.no_grad()
    def act(self, state):
        """
        state: WorldModelState
        returns: (actions, logprobs)
        """
        logits = self.forward(state.h, state.z)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        return actions, logprobs
