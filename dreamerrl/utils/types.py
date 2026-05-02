from dataclasses import dataclass, field
from typing import Optional

import torch

from .transforms import symexp


# ---------------------------------------------------------
# 1. World Model Config (RSSM + Encoder/Decoder)
# ---------------------------------------------------------
@dataclass
class WorldModelConfig:
    # deter_size:
    #   Size of the deterministic hidden state h_t in the RSSM.
    #   Larger values improve model capacity but increase compute.
    #
    # stoch_size, num_classes:
    #   Dreamer‑V3 uses a *factored discrete latent*:
    #       z_t has stoch_size categorical factors,
    #       each with num_classes categories.
    #   This replaces the Gaussian latent used in Dreamer‑V2/Lite.
    #   Tests MUST NOT assume Gaussian shapes or continuous z_t.
    #
    # hidden_size:
    #   Shared MLP hidden size for RSSMCore, Prior, Posterior, Decoder.
    #
    # free_bits:
    #   Minimum KL contribution (in nats) to prevent posterior collapse.
    #   If KL < free_bits, the gradient is clamped.
    #   Dreamer‑V3 typically uses a small positive value (e.g., 1.0).
    #
    # value_bins:
    #   Number of symlog‑spaced bins used for distributional reward/value
    #   prediction. Tests MUST NOT assume scalar reward/value outputs.
    reward_hidden: int = 256
    deter_size: int = 200
    stoch_size: int = 30
    hidden_size: int = 200

    kl_scale: float = 1.0
    free_nats: float = 3.0
    kl_balance: float = 0.8

    # Dreamer-Lite uses a simpler encoder/decoder
    encoder_hidden: int = 256
    decoder_hidden: int = 256

    imagination_horizon: int = 15

    num_classes: int = 32
    value_bins: int = 41
    free_bits: float = 0.0


# ---------------------------------------------------------
# 2. Actor/Critic Config
# ---------------------------------------------------------
@dataclass
class ActorCriticConfig:
    actor_hidden: int = 256
    critic_hidden: int = 256

    # discount:
    #   Standard RL discount factor.
    #
    # lambda_:
    #   GAE‑style smoothing factor used in Dreamer‑V3’s λ‑returns.
    #   Tests MUST NOT assume V2‑style bootstrapping or TD(0).
    discount: float = 0.99
    lambda_: float = 0.95  # GAE-like smoothing for returns in Dreamer


# ---------------------------------------------------------
# 3. Training Config
# ---------------------------------------------------------
@dataclass
class TrainingConfig:
    # warmup_steps:
    #   Number of steps for linear LR warmup before cosine decay.
    #   Dreamer‑V3 is extremely sensitive to LR warmup; tests MUST NOT
    #   assume constant LR from step 0.
    #
    # random_exploration_steps:
    #   Number of steps where actions are sampled uniformly at random.
    #   Tests MUST NOT assume the actor is used from step 0.
    batch_size: int = 50
    seq_len: int = 50

    model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    grad_clip: float = 100.0
    updates_per_step: int = 1
    warmup_steps: int = 1000
    random_exploration_steps: int = 2500
    replay_capacity: int = 10000

    cuda: bool = True
    seed: int = 0


# ---------------------------------------------------------
# 4. Environment Config
# ---------------------------------------------------------
@dataclass
class EnvironmentConfig:
    env_id: str = "popgym-RepeatPreviousEasy-v0"
    num_envs: int = 64
    max_episode_steps: Optional[int] = None


# ---------------------------------------------------------
# 5. Logging Config
# ---------------------------------------------------------
@dataclass
class LoggingConfig:
    tb_logdir: str = "./tb_logs"
    checkpoint_dir: str = "./checkpoints"
    run_name: str = ""


# ---------------------------------------------------------
# 6. Top-level Dreamer Config
# ---------------------------------------------------------
@dataclass
class DreamerConfig:
    mode: str = "lite"

    # Feature toggles
    use_stochastic_latent: bool = False
    use_kl_balance: bool = False
    use_free_nats: bool = False
    use_overshooting: bool = False
    use_value_bootstrap: bool = False

    world: WorldModelConfig = field(default_factory=WorldModelConfig)
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)

    def init_run_name(self):
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log.run_name = f"{self.env.env_id}__dreamer__{ts}"

    def __post_init__(self):
        if self.mode == "lite":
            self.use_stochastic_latent = False
            self.use_kl_balance = False
            self.use_free_nats = False
            self.use_overshooting = False
            self.use_value_bootstrap = False

        elif self.mode == "full":
            self.use_stochastic_latent = True
            self.use_kl_balance = True
            self.use_free_nats = True
            self.use_overshooting = True
            self.use_value_bootstrap = True


def make_default_config() -> DreamerConfig:
    cfg = DreamerConfig()
    cfg.init_run_name()
    return cfg


@dataclass(frozen=True)
class LatentConfig:
    """
    Shared Dreamer-V3 latent configuration.
    Prevents silent shape bugs by centralizing latent geometry.

    z_dim:
       Flattened dimension of the discrete latent:
           z_dim = stoch_size * num_classes
       Tests MUST NOT assume Gaussian latent shapes (mean, std).
    """

    deter_size: int
    stoch_size: int
    num_classes: int

    @property
    def z_dim(self) -> int:
        return self.stoch_size * self.num_classes


@dataclass(frozen=True)
class NetworkConfig:
    """
    Shared network configuration for Actor/Critic/Heads.
    Prevents accidental swapping of hidden_size, action_dim, value_bins.

    value_bins:
       Number of symlog‑spaced bins for distributional value/reward heads. Dreamer‑V3 predicts logits over bins, NOT
       scalar values. Tests MUST NOT expect shape (B, 1) for reward/value predictions.

    make_bins():
        Produces symlog‑spaced bin centers, then applies symexp. These bins define the support of the distributional
        heads.
    """

    hidden_size: int
    action_dim: int | None = None
    value_bins: int | None = None
    bin_min: float = -10.0
    bin_max: float = 10.0

    def make_bins(self, device=None):
        symlog_bins = torch.linspace(self.bin_min, self.bin_max, steps=self.value_bins)
        bins = symexp(symlog_bins)
        return bins if device is None else bins.to(device)


@dataclass(frozen=True)
class LRScheduleConfig:
    # Linear warmup + cosine decay schedule.
    #
    # Dreamer‑V3 requires a SINGLE shared LR schedule for world model,
    # actor, and critic. Tests MUST NOT create separate LR schedules or
    # assume constant LR. Warmup is essential for stability because the
    # actor/critic depend on a partially‑trained world model early on.
    base_lr: float
    warmup_steps: int
    total_steps: int

    # lr_floor:
    #   Fraction of base_lr to preserve during cosine decay.
    #   Prevents the learning rate from collapsing to zero, which would
    #   freeze the actor/critic/world model and halt improvement.
    #
    #   Dreamer‑V3 benefits from a small but non‑zero LR late in training:
    #     • stabilizes long‑horizon value estimates,
    #     • prevents the actor from becoming overconfident,
    #     • avoids world‑model stagnation.
    #
    #   Typical values: 0.05–0.20 of base_lr.
    lr_floor: float = 0.1
