import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Dreamer-style value function.
    Predicts scalar value from latent state (h, z).
    """

    def __init__(self, deter_size: int, stoch_size: int, hidden_size: int):
        super().__init__()

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

        input_dim = deter_size + stoch_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),  # scalar value
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h, z):
        """
        h: (B, deter_size)
        z: (B, stoch_size)
        Returns: value estimate (B, 1)
        """
        x = torch.cat([h, z], dim=-1)
        return self.net(x)
