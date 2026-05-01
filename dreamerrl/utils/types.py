from dataclasses import dataclass, field
from typing import Optional

import torch

from .transforms import symexp


# ---------------------------------------------------------
# 1. World Model Config (RSSM + Encoder/Decoder)
# ---------------------------------------------------------
@dataclass
class WorldModelConfig:
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


# ---------------------------------------------------------
# 2. Actor/Critic Config
# ---------------------------------------------------------
@dataclass
class ActorCriticConfig:
    actor_hidden: int = 256
    critic_hidden: int = 256

    discount: float = 0.99
    lambda_: float = 0.95  # GAE-like smoothing for returns in Dreamer


# ---------------------------------------------------------
# 3. Training Config
# ---------------------------------------------------------
@dataclass
class TrainingConfig:
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
