import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from dreamerrl.models.categorical import mixed_probs


class Actor(nn.Module):
    """
    Dreamer‑V3 policy network.
    Consumes factored z and deterministic h.
    Outputs logits over discrete actions.
    """

    def __init__(self, *, latent, net, probe=None):
        super().__init__()

        self.latent = latent
        self.net_cfg = net
        self.probe = probe
        self.register_buffer("return_low", torch.tensor(0.0))
        self.register_buffer("return_high", torch.tensor(1.0))
        self.register_buffer("return_norm_initialized", torch.tensor(False))

        self.z_embed = nn.Linear(latent.z_dim, net.hidden_size)
        self.h_embed = nn.Linear(latent.deter_size, net.hidden_size)

        self.net = nn.Sequential(
            nn.Linear(net.hidden_size, net.hidden_size),
            nn.SiLU(),
            nn.Linear(net.hidden_size, net.action_dim),
        )

    def forward(self, h, z):
        """
        h: (B, deter_size)
        z: (B, K, C)
        """
        # Preserve the identity of every categorical factor.
        z_e = self.z_embed(z.reshape(z.shape[0], -1))
        h_e = self.h_embed(h)
        features = h_e + z_e
        logits = mixed_probs(self.net(features), self.net_cfg.unimix).log()

        if self.probe:
            self.probe.actor(logits)

        return logits  # logits over actions

    @torch.no_grad()
    def act(self, state, deterministic: bool = False):
        """
        Sample an action from the policy.

        state: RSSMState with fields:
            - h: (B, deter_size)
            - z: (B, K, C)

        Returns:
            actions: (B,) int64
            logits:  (B, action_dim)
        """
        h = state.h
        z = state.z

        logits = self.forward(h, z)  # (B, action_dim)

        if deterministic:
            # Greedy action (used in evaluation or reproducibility tests)
            actions = torch.argmax(logits, dim=-1)
        else:
            # Categorical sampling
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1).to(logits.device)

        return actions, logits


def act_in_imagination(logits, deterministic_imagination: bool = False):
    if deterministic_imagination:
        a = logits.argmax(dim=-1)
    else:
        dist = Categorical(logits=logits)
        a = dist.sample().to(logits.device)

    return a
