from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------
# 1. World Model Config (RSSM + Encoder/Decoder)
# ---------------------------------------------------------
@dataclass
class WorldModelConfig:
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
    world: WorldModelConfig = field(default_factory=WorldModelConfig)
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)

    def init_run_name(self):
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log.run_name = f"{self.env.env_id}__dreamer__{ts}"
