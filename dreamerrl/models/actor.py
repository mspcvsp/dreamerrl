# inside dreamerrl/models/actor.py

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, deter_size, stoch_size, hidden_size, action_dim):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(deter_size + stoch_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))

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
